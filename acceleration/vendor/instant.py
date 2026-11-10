"""Contiguous K-block generation for the finite threshold sweep.

This is a new, measured generator family. K is independent of output length.
K=1, mu=0 retains the frozen unpruned dual-cache update/selection semantics,
with an explicit per-block round cap. K>1 uses contiguous groups instead of the
historical bidirectional/scatter layouts. No model is loaded by this module.
"""
import math
import random
import time


def block_groups(gen_length, block_length=32, parallel_k=1):
    """Partition each generated block once, retaining a shorter final group."""
    for value, label in ((gen_length, "gen_length"), (block_length, "block_length"), (parallel_k, "parallel_k")):
        if type(value) is not int or value < 1:
            raise ValueError(label + " must be a positive integer")
    if gen_length % block_length:
        raise ValueError("gen_length must be divisible by block_length")
    count = gen_length // block_length
    return [list(range(start, min(start + parallel_k, count))) for start in range(0, count, parallel_k)]


def estimated_forward_matmul_flops(spec, batch_size, query_length, key_length):
    """Dense matmul estimate: every linear map plus attention QK and AV.

    One multiply-add is two FLOPs. Excludes softmax, nonlinearities, normalization,
    rotary embeddings, selection/BERM, KV copies, kernel padding and memory work.
    This is an analytical executed-shape estimate, never a hardware measurement.
    """
    if spec is None:
        return None
    if any(type(v) is not int or v < 1 for v in (batch_size, query_length, key_length)):
        raise ValueError("Positive integer forward dimensions required")
    linear = 2 * batch_size * query_length * spec["linear_weight_elements_per_query"]
    attention = 4 * batch_size * query_length * key_length * spec["attention_width_times_layers"]
    return {"linear_flops": linear, "attention_flops": attention, "total_flops": linear + attention}


def model_matmul_spec(model):
    """Use actual frozen model Linear weight shapes; add a tied embedding LM head."""
    import torch
    linears = [(name, module) for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)]
    if not linears:
        raise ValueError("Cannot estimate FLOPs: model has no inspectable dense Linear modules")
    config = getattr(model, "config", None)
    width = getattr(config, "d_model", getattr(config, "hidden_size", None))
    layers = getattr(config, "n_layers", getattr(config, "num_hidden_layers", None))
    if not isinstance(width, int) or not isinstance(layers, int) or width < 1 or layers < 1:
        raise ValueError("Cannot estimate FLOPs: model width/layer count absent")
    linear_weights = sum(int(module.in_features) * int(module.out_features) for _, module in linears)
    extra_head = 0
    output = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    if output is not None and not isinstance(output, torch.nn.Linear):
        weight = getattr(output, "weight", None)
        if weight is None or weight.ndim != 2 or int(weight.shape[1]) != width:
            raise ValueError("Cannot identify tied output projection")
        extra_head = int(weight.numel())
    if output is None:
        raise ValueError("Output projection must be explicitly inspectable")
    return {"linear_weight_elements_per_query": linear_weights + extra_head,
            "linear_module_weight_elements": linear_weights, "tied_head_weight_elements_added": extra_head,
            "linear_module_count": len(linears), "attention_width_times_layers": width * layers,
            "hidden_width": width, "layers": layers, "multiply_add_flops": 2,
            "scope": "Dense linear maps including vocabulary head plus QK/AV; excludes non-matmul and memory work"}


def _select(logits, x, mask, threshold, torch):
    """Frozen temperature-zero, low-confidence threshold selection, plus counts.

    Confidence uses float64 softmax, exactly as the frozen get_transfer_index.
    The highest-confidence masked position is ORed into each row, even if it is
    already above threshold. Additional fallback counts mean below-threshold
    positions selected only by this OR operation, not all forced maxima.
    """
    proposals = torch.argmax(logits, dim=-1)
    probabilities = torch.nn.functional.softmax(logits.to(torch.float64), dim=-1)
    proposed_confidence = torch.gather(probabilities, -1, proposals.unsqueeze(-1)).squeeze(-1)
    proposals = torch.where(mask, proposals, x)
    negative = torch.tensor(torch.finfo(proposed_confidence.dtype).min,
                            dtype=proposed_confidence.dtype, device=proposed_confidence.device)
    confidence = torch.where(mask, proposed_confidence, negative)
    above = mask & (confidence >= threshold)
    maxima = torch.argmax(confidence, dim=1, keepdim=True)
    forced = torch.zeros_like(above).scatter_(1, maxima, True) & mask.any(dim=1, keepdim=True)
    selected = (above | forced) & mask
    return proposals, selected, confidence, {
        "threshold_selected_positions": int(above.sum().item()),
        "fallback_added_positions": int((forced & ~above).sum().item()),
        "fallback_candidate_rows": int(mask.any(dim=1).sum().item()),
        "selected_positions": int(selected.sum().item()),
    }


def _cache_length(cache):
    if not isinstance(cache, (tuple, list)) or not cache:
        raise ValueError("Expected frozen model tuple/list KV cache")
    lengths = []
    for layer in cache:
        if not isinstance(layer, (tuple, list)) or len(layer) != 2:
            raise ValueError("Expected key/value pair in every layer")
        if layer[0].shape[-2] != layer[1].shape[-2]:
            raise ValueError("Key/value lengths differ")
        lengths.append(int(layer[0].shape[-2]))
    if len(set(lengths)) != 1:
        raise ValueError("Layer cache lengths differ")
    return lengths[0]


def _sum_selection(totals, update):
    for name, value in update.items():
        totals[name] = totals.get(name, 0) + value


def generate_generic_k(model, prompt, *, gen_length=256, block_length=32, parallel_k=1,
                       threshold=0.9, berm_mu=0, max_rounds_per_group=64,
                       temperature=0.0, remasking="low_confidence", mask_id=126336,
                       seed=None, refresh_interval=0, kv_delta_gamma=0.0,
                       adaptive_alpha=None, eos_token_id=None, return_diagnostics=True,
                       matmul_spec=None):
    """Return (full_ids,nfe,metrics), or (full_ids,nfe) if diagnostics=False.

    prompt is batch1 [1,L]. Frozen model cache replacement occurs in place.
    No EOS early stop or late fill is performed: each configured budget is kept.
    Seed=None leaves caller RNG state alone; seed sets RNGs once before this call.
    """
    import torch
    groups = block_groups(gen_length, block_length, parallel_k)
    if parallel_k not in (1, 2, 3, 4, 8):
        raise ValueError("Campaign supports K=1,2,3,4,8")
    if type(max_rounds_per_group) is not int or max_rounds_per_group < 1:
        raise ValueError("max_rounds_per_group must be a positive integer including warmup")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("Finite confidence threshold in (0,1] required")
    if berm_mu not in (0, 16):
        raise ValueError("Campaign supports mu=0 or16")
    if temperature != 0 or remasking != "low_confidence":
        raise ValueError("Frozen campaign requires temperature0 and low_confidence")
    if refresh_interval != 0 or kv_delta_gamma != 0 or adaptive_alpha is not None or eos_token_id is not None:
        raise ValueError("Campaign disables refresh, injection, adaptive threshold, and EOS early stop")
    if prompt.ndim != 2 or int(prompt.shape[0]) != 1 or int(prompt.shape[1]) < 1:
        raise ValueError("A nonempty batch1 prompt tensor is required")
    if prompt.dtype != torch.long or bool((prompt == mask_id).any().item()):
        raise ValueError("Prompt must contain integer tokens and no MASK token")
    if seed is not None:
        torch.manual_seed(int(seed))
        random.seed(int(seed))
        try:
            import numpy as np
            np.random.seed(int(seed))
        except ImportError:
            pass
    device = prompt.device
    batch = int(prompt.shape[0])
    prompt_length = int(prompt.shape[1])
    sequence_length = prompt_length + gen_length
    flop_spec = model_matmul_spec(model) if matmul_spec is None else matmul_spec
    counters = {"observed_forward_calls": 0, "observed_query_positions": 0}
    forward_records, group_records, selection = [], [], {}
    berm_total = 0

    def observed_forward(module, arguments):
        if not arguments or not hasattr(arguments[0], "shape"):
            raise ValueError("Unmapped model forward input")
        counters["observed_forward_calls"] += 1
        counters["observed_query_positions"] += int(arguments[0].shape[0] * arguments[0].shape[1])

    def forward(tokens, group_index, round_index, phase, **kwargs):
        qlen = int(tokens.shape[1])
        result = model(tokens, use_cache=True, **kwargs)
        key_length = _cache_length(result.past_key_values)
        if key_length != sequence_length or tuple(result.logits.shape[:2]) != (batch, qlen):
            raise ValueError("Frozen model returned an unexpected cache/logit shape")
        flops = estimated_forward_matmul_flops(flop_spec, batch, qlen, key_length)
        forward_records.append({"index": len(forward_records), "group_index": group_index,
                                "round_index": round_index, "phase": phase, "batch_size": batch,
                                "query_length": qlen, "query_positions": batch * qlen,
                                "key_length": key_length, "estimated_main_matmul_flops": flops["total_flops"],
                                "estimated_linear_flops": flops["linear_flops"],
                                "estimated_attention_flops": flops["attention_flops"]})
        return result

    handle = model.register_forward_pre_hook(observed_forward)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    try:
        with torch.no_grad():
            x = torch.full((batch, sequence_length), mask_id, dtype=torch.long, device=device)
            x[:, :prompt_length] = prompt
            for group_index, block_ids in enumerate(groups):
                ranges = [(prompt_length + block * block_length, prompt_length + (block + 1) * block_length) for block in block_ids]
                current_k = len(ranges)
                group_start, group_end = ranges[0][0], ranges[-1][1]
                before_nfe = len(forward_records)
                before_selection = selection.copy()
                group_berm = 0
                warmup_prior_mask_updates = 0
                output = forward(x, group_index, 0, "full_warmup")
                past = output.past_key_values
                # K1 matches original dual-cache global warmup selection through
                # the current block. Normally prior positions are fully decoded;
                # slice just that block to avoid a redundant full-vocabulary
                # float64 softmax over already-decoded context. Rowwise choices
                # and ascending argmax tie order are unchanged by this slice.
                if parallel_k == 1:
                    selection_start = 0 if bool((x[:, :group_start] == mask_id).any().item()) else group_start
                    target = x[:, selection_start:group_end]
                    mask = target == mask_id
                    proposal, take, confidence, counts = _select(
                        output.logits[:, selection_start:group_end], target, mask, threshold, torch)
                    new_target = torch.where(take, proposal, target)
                    if selection_start == 0:
                        warmup_prior_mask_updates = int((take[:, :group_start] & (proposal[:, :group_start] != mask_id)).sum().item())
                    x[:, selection_start:group_end] = new_target
                    _sum_selection(selection, counts)
                else:
                    for start_pos, end_pos in ranges:
                        target = x[:, start_pos:end_pos]
                        proposal, take, confidence, counts = _select(output.logits[:, start_pos:end_pos], target, target == mask_id, threshold, torch)
                        x[:, start_pos:end_pos] = torch.where(take, proposal, target)
                        _sum_selection(selection, counts)
                del output, proposal, take, confidence
                replace = torch.zeros_like(x, dtype=torch.bool)
                replace[:, group_start:group_end] = True
                positions = torch.arange(group_start, group_end, device=device).unsqueeze(0)
                rounds = 1
                for round_index in range(1, max_rounds_per_group):
                    if not bool((x[:, group_start:group_end] == mask_id).any().item()):
                        break
                    call_options = {"past_key_values": past, "replace_position": replace}
                    # Original K1 dual-cache uses implicit contiguous RoPE.
                    if parallel_k != 1:
                        call_options["position_ids"] = positions
                    output = forward(x[:, group_start:group_end], group_index, round_index, "cached_group_refine", **call_options)
                    entries = []
                    for block_index, (start_pos, end_pos) in enumerate(ranges):
                        logits = output.logits[:, block_index * block_length:(block_index + 1) * block_length]
                        target = x[:, start_pos:end_pos]
                        pre_mask = target == mask_id
                        proposal, take, confidence, counts = _select(logits, target, pre_mask, threshold, torch)
                        replacement = torch.where(take, proposal, target)
                        if berm_mu:
                            newly_decoded = pre_mask & (replacement != mask_id)
                            for offset in newly_decoded[0].nonzero(as_tuple=True)[0].tolist():
                                entries.append((float(confidence[0, offset].item()), start_pos + offset))
                        x[:, start_pos:end_pos] = replacement
                        _sum_selection(selection, counts)
                    if berm_mu and entries:
                        # Original BERM protects a net gain of the actual group
                        # size, including shorter final groups, not requested K.
                        remask_count = min(len(entries) // berm_mu, max(0, len(entries) - current_k))
                        entries.sort(key=lambda entry: entry[0])
                        for _, position in entries[:remask_count]:
                            x[0, position] = mask_id
                        group_berm += remask_count
                        berm_total += remask_count
                    rounds += 1
                    del output, proposal, take, confidence
                remaining = int((x[:, group_start:group_end] == mask_id).sum().item())
                group_records.append({"group_index": group_index, "block_ids": block_ids,
                                      "requested_k": parallel_k, "actual_k": current_k,
                                      "absolute_start": group_start, "absolute_end": group_end,
                                      "rounds_including_warmup": rounds,
                                      "nfe": len(forward_records) - before_nfe,
                                      "round_cap_reached": rounds == max_rounds_per_group,
                                      "residual_masks_at_group_exit": remaining,
                                      "converged_at_group_exit": remaining == 0,
                                      "berm_remasked_tokens": group_berm,
                                      "warmup_prior_mask_updates": warmup_prior_mask_updates,
                                      "selection": {key: value - before_selection.get(key, 0) for key, value in selection.items()}})
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
    finally:
        handle.remove()
    nfe = len(forward_records)
    query_positions = sum(record["query_positions"] for record in forward_records)
    if nfe != counters["observed_forward_calls"] or query_positions != counters["observed_query_positions"]:
        raise ValueError("Recorded model calls and independent forward hook differ")
    if not torch.equal(x[:, :prompt_length], prompt):
        raise ValueError("Generator changed prompt tokens")
    raw = x[:, prompt_length:]
    mask_positions = (raw[0] == mask_id).nonzero(as_tuple=True)[0].tolist()
    metrics = {"schema": "generic_contiguous_k_generation_v1", "nfe": nfe,
               "actual_forward_calls": counters["observed_forward_calls"], "query_positions": query_positions,
               "observed_query_positions": counters["observed_query_positions"],
               "estimated_main_matmul_flops": sum(record["estimated_main_matmul_flops"] for record in forward_records),
               "estimated_linear_flops": sum(record["estimated_linear_flops"] for record in forward_records),
               "estimated_attention_flops": sum(record["estimated_attention_flops"] for record in forward_records),
               "matmul_estimate_spec": flop_spec, "forward_records": forward_records, "groups": group_records,
               "selection": selection, "fallback_added_positions": selection.get("fallback_added_positions", 0),
               "fallback_tokens": selection.get("fallback_added_positions", 0), "berm_remasked_tokens": berm_total,
               "residual_masks": len(mask_positions), "residual_mask_positions": mask_positions,
               "output_token_budget": gen_length, "block_length": block_length, "parallel_k": parallel_k,
               "threshold": threshold, "berm_mu": berm_mu, "max_rounds_per_group": max_rounds_per_group,
               "generation_seconds": elapsed, "prompt_prefix_unchanged": True,
               "key_lengths": sorted(set(record["key_length"] for record in forward_records)),
               "rng_seed_this_call": seed,
               "equivalence_scope": ("K1 mu0 retains original unpruned dual-cache token/cache update semantics at the same round cap; "
                                     "K1 warmup skips confidence computation for decoded rows, which does not change selection. "
                                     "K>1 is a new contiguous grouping family; it is not historical bidirectional/scatter grouping. "
                                     "Analytical matmul counts exclude selection/BERM and memory work; generation timing includes diagnostics.")}
    return (x, nfe, metrics) if return_diagnostics else (x, nfe)


def generate_grouped(model, prompt_tensor, gen_length, block_length=32, max_rounds_per_group=64,
                     parallel_k=1, threshold=0.9, berm_mu=0, seed=None, **kwargs):
    """Runner convenience API: return (full_sequence_ids, metrics)."""
    if "return_diagnostics" in kwargs:
        raise ValueError("generate_grouped always returns diagnostics")
    ids, _, metrics = generate_generic_k(model, prompt_tensor, gen_length=gen_length,
                                        block_length=block_length, max_rounds_per_group=max_rounds_per_group,
                                        parallel_k=parallel_k, threshold=threshold, berm_mu=berm_mu,
                                        seed=seed, return_diagnostics=True, **kwargs)
    return ids, metrics
