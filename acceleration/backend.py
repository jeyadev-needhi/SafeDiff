"""Batch-one LLaDA generation with shared weights and timing metrics.

load() imports the model dependencies and loads one checkpoint and tokenizer
for all three modes. The caller selects the mode or policy.
"""

from collections import Counter
from copy import deepcopy
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import random
import threading
import time

from .presets import describe_config, mode_for_policy


def _json_hash(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def source_provenance():
    """Verify packaged decoder/model bytes against the committed manifest."""
    root = Path(__file__).resolve().parent / "vendor"
    record = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    for rel, expected in record["packaged_files"].items():
        path = (root / rel).resolve()
        if root not in path.parents:
            raise RuntimeError("Invalid vendor source path: " + rel)
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected["sha256"]:
            raise RuntimeError("Pinned acceleration source changed: " + rel)
    vanilla = json.loads((root / "llada_provenance.json").read_text(encoding="utf-8"))
    for rel, expected in vanilla["files"].items():
        path = (root / rel).resolve()
        if root not in path.parents:
            raise RuntimeError("Invalid LLaDA source path: " + rel)
        raw = path.read_bytes()
        if (hashlib.sha256(raw).hexdigest() != expected["sha256"]
                or len(raw) != expected["bytes"]):
            raise RuntimeError("Pinned LLaDA source changed: " + rel)
    record["vanilla_llada"] = vanilla
    return record


class _ForwardObserver:
    """Count forward calls and record input and cache lengths."""

    def __init__(self):
        self.shapes = Counter()

    def __call__(self, module, args, kwargs, output):
        tokens = args[0] if args else kwargs.get("input_ids")
        if tokens is None or tokens.ndim != 2:
            raise RuntimeError("Unexpected model inputs/cache in the forward observer")
        cache = getattr(output, "past_key_values", None)
        full = kwargs.get("past_key_values") is None
        if cache is None:
            if not full or kwargs.get("use_cache") is not False:
                raise RuntimeError("Missing KV output outside an explicitly uncached forward")
            key_length = int(tokens.shape[1])
        else:
            if not isinstance(cache, (tuple, list)) or not cache:
                raise RuntimeError("Unexpected KV output in the forward observer")
            key_lengths = {int(pair[0].shape[-2]) for pair in cache}
            if len(key_lengths) != 1 or any(pair[0].shape[-2] != pair[1].shape[-2] for pair in cache):
                raise RuntimeError("Inconsistent layer KV lengths")
            key_length = key_lengths.pop()
        self.shapes[(int(tokens.shape[0]), int(tokens.shape[1]), key_length, full)] += 1

    def result(self, nfe):
        actual = sum(self.shapes.values())
        if int(nfe) != actual:
            raise RuntimeError("Decoder NFE does not match actual model forward calls")
        return {
            "nfe": actual,
            "actual_forward_calls": actual,
            "query_positions": sum(b * q * n for (b, q, k, full), n in self.shapes.items()),
            "full_forward_calls": sum(n for (b, q, k, full), n in self.shapes.items() if full),
            "cached_forward_calls": sum(n for (b, q, k, full), n in self.shapes.items() if not full),
            "forward_shape_counts": [
                {"batch": b, "query_length": q, "key_length": k, "full_forward": full, "count": n}
                for (b, q, k, full), n in sorted(self.shapes.items())
            ],
        }


class _UncachedModel:
    """Disable KV caching for vanilla LLaDA."""

    def __init__(self, model):
        self._model = model
        self.device = model.device

    def __call__(self, tokens):
        result = self._model(tokens, use_cache=False)
        if getattr(result, "past_key_values", None) is not None:
            raise RuntimeError("Vanilla LLaDA forward unexpectedly created a KV cache")
        return result


class LLaDABackend:
    """Shared LLaDA-8B-Instruct checkpoint and tokenizer for three modes.

    Accepts a local checkpoint or a Hugging Face ID and revision. Downloads are
    opt-in. generate() applies the chat template and returns JSON-ready results.
    Inference runs on CUDA; calls on one backend are serialized.
    """

    def __init__(self, model_path, *, revision=None, device="cuda:0",
                 local_files_only=True, dtype="bfloat16", flash_attention=True):
        if not isinstance(model_path, (str, Path)) or not str(model_path).strip():
            raise ValueError("model_path must be a checkpoint directory or model repository ID")
        if dtype not in ("bfloat16", "float16", "float32"):
            raise ValueError("dtype must be bfloat16, float16, or float32")
        if device != "cuda" and not (isinstance(device, str) and device.startswith("cuda:")
                                     and device[5:].isdigit()):
            raise ValueError("Only a CUDA device is supported")
        self.model_path = str(model_path)
        self.revision = revision
        self.device = device
        self.local_files_only = bool(local_files_only)
        self.dtype = dtype
        self.flash_attention = bool(flash_attention)
        self._model = None
        self._tokenizer = None
        self._runtime = None
        self._provenance = None
        self._matmul_spec = None
        self._lock = threading.RLock()

    def load(self):
        """Load the model once; also called by generate() on first use."""
        with self._lock:
            if self._model is not None:
                return self
            started = time.perf_counter()
            provenance = source_provenance()
            import torch
            import transformers
            from transformers import AutoTokenizer
            from .vendor.model.configuration_llada import LLaDAConfig, ModelConfig
            from .vendor.model.modeling_llada import LLaDAModelLM
            from .vendor.instant import model_matmul_spec

            if not torch.cuda.is_available():
                raise RuntimeError("A CUDA GPU is required; no CPU inference fallback")
            device = torch.device(self.device)
            dtype = getattr(torch, self.dtype)
            if self.dtype == "bfloat16":
                with torch.cuda.device(device):
                    if not torch.cuda.is_bf16_supported():
                        raise RuntimeError("Selected GPU does not support bfloat16")
            options = {"local_files_only": self.local_files_only}
            if self.revision is not None:
                options["revision"] = self.revision
            checkpoint_config, unused = LLaDAConfig.get_config_dict(self.model_path, **options)
            config = LLaDAConfig.from_pretrained(self.model_path, **options)
            if int(config.d_model) != 4096 or int(config.n_layers) != 32 or int(config.vocab_size) <= 126336:
                raise ValueError("Checkpoint configuration is not compatible with LLaDA-8B and mask ID 126336")
            # The FlashAttention CUDA kernel requires fp16 or bf16 inputs.
            config.flash_attention = self.flash_attention and self.dtype != "float32"
            missing = {f.name: getattr(config, f.name) for f in fields(ModelConfig)
                       if f.name not in checkpoint_config}
            tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True, **options)
            if (tokenizer.convert_tokens_to_ids("<|mdm_mask|>") != 126336
                    or tokenizer.convert_ids_to_tokens(126336) != "<|mdm_mask|>"):
                raise ValueError("Checkpoint tokenizer must map <|mdm_mask|> to token ID 126336")
            if not tokenizer.chat_template:
                raise ValueError("The checkpoint must supply the LLaDA-Instruct chat template")
            model = LLaDAModelLM.from_pretrained(
                self.model_path, config=config, torch_dtype=dtype, **options
            ).to(device).eval()
            torch.cuda.synchronize(device)
            modules = [m for m in model.modules() if hasattr(m, "flash_attn_func")]
            revision = getattr(config, "_commit_hash", None)
            local = Path(self.model_path).expanduser()
            if revision is None and local.is_dir() and local.parent.name == "snapshots":
                revision = local.name
            metadata_hashes = {}
            if local.is_dir():
                for name in ("config.json", "tokenizer_config.json", "tokenizer.json",
                             "special_tokens_map.json", "model.safetensors.index.json",
                             "pytorch_model.bin.index.json"):
                    path = local / name
                    if path.is_file():
                        metadata_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            self._runtime = {
                "checkpoint": str(local.resolve()) if local.is_dir() else self.model_path,
                "requested_revision": self.revision,
                "resolved_revision": revision,
                "checkpoint_config_sha256": _json_hash(checkpoint_config),
                "local_metadata_sha256": metadata_hashes,
                "weight_identity_note": "Revision and metadata are recorded; full weight files are not rehashed here",
                "dtype": self.dtype,
                "device": str(device),
                "gpu_name": torch.cuda.get_device_name(device),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "transformers": transformers.__version__,
                "flash_attention_requested": self.flash_attention,
                "attention_modules_using_flash_attn": sum(callable(m.flash_attn_func) for m in modules),
                "attention_modules_using_sdpa": sum(m.flash_attn_func is None for m in modules),
                "model_implementation": "shared model with explicit-position RoPE support",
                "defaults_added_for_missing_checkpoint_fields": missing,
                "model_load_seconds": time.perf_counter() - started,
                "chat_template_sha256": _json_hash(tokenizer.chat_template),
                "verified_mask_token": {"content": "<|mdm_mask|>", "id": 126336},
            }
            self._matmul_spec = model_matmul_spec(model)
            self._model, self._tokenizer, self._provenance = model, tokenizer, provenance
            return self

    def runtime_info(self):
        """Return loaded runtime metadata, or None before the first load."""
        with self._lock:
            return deepcopy(self._runtime)

    def generate_for_policy(self, prompt, *, policy, gen_length=256, seed=42,
                            system_prompt=None):
        """Generate a response using the selected policy."""
        mode = mode_for_policy(policy)
        result = self.generate(prompt, mode=mode, gen_length=gen_length,
                               seed=seed, system_prompt=system_prompt)
        return {**result, "policy": policy}

    def generate(self, prompt, *, mode="instant", gen_length=256, seed=42, system_prompt=None):
        """Generate one response using the requested fixed mode.

        generation_seconds covers the CUDA-synchronized decoder and observers.
        request_seconds also covers lazy loading (if any), tokenization, seed
        setup and decoding.
        """
        config = describe_config(mode, gen_length=gen_length, seed=seed)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a nonempty string")
        if system_prompt is not None and not isinstance(system_prompt, str):
            raise ValueError("system_prompt must be a string or None")
        with self._lock:
            started = time.perf_counter()
            was_loaded = self._model is not None
            self.load()
            import numpy as np
            import torch
            from .vendor import fastdllm, instant, llada

            messages = []
            if system_prompt is not None:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            inputs = self._tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to(self._model.device)
            if inputs.ndim != 2 or int(inputs.shape[0]) != 1 or inputs.shape[1] == 0:
                raise ValueError("The chat template must produce one nonempty prompt")
            if bool((inputs == config["mask_id"]).any().item()):
                raise ValueError("Prompt contains a MASK token; masked prompts are unsupported")
            max_context = int(self._model.config.max_sequence_length)
            if inputs.shape[1] + gen_length > max_context:
                raise ValueError("Prompt plus generation budget exceeds checkpoint context length")
            prompt_ids = inputs[0].tolist()
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            observer = _ForwardObserver()
            hook = self._model.register_forward_hook(observer, with_kwargs=True)
            device = self._model.device
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            generation_start = time.perf_counter()
            details = None
            try:
                with torch.no_grad():
                    if mode == "instant":
                        full, nfe, details = instant.generate_generic_k(
                            self._model, inputs, gen_length=gen_length, block_length=32,
                            parallel_k=8, threshold=0.9, berm_mu=16, max_rounds_per_group=64,
                            temperature=0.0, mask_id=126336, matmul_spec=self._matmul_spec,
                        )
                    elif mode == "fastdllm":
                        full, nfe = fastdllm.generate_with_dual_cache(
                            self._model, inputs, steps=gen_length, gen_length=gen_length,
                            block_length=32, temperature=0.0, remasking="low_confidence",
                            mask_id=126336, threshold=0.9, factor=None,
                        )
                    else:
                        full = llada.generate(
                            _UncachedModel(self._model), inputs,
                            steps=gen_length, gen_length=gen_length,
                            block_length=gen_length, temperature=0.0,
                            cfg_scale=0.0, remasking="low_confidence", mask_id=126336,
                        )
                        nfe = gen_length
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - generation_start
                peak_memory = int(torch.cuda.max_memory_allocated(device))
            finally:
                hook.remove()
            metrics = observer.result(nfe)
            if mode == "llada" and (metrics["cached_forward_calls"] != 0
                                    or metrics["query_positions"] != gen_length * (len(prompt_ids) + gen_length)):
                raise RuntimeError("Vanilla LLaDA must recompute the full sequence at every step")
            if not torch.equal(full[:, :inputs.shape[1]], inputs):
                raise RuntimeError("Generation changed the prompt prefix")
            raw_ids = full[0, inputs.shape[1]:].tolist()
            if len(raw_ids) != gen_length:
                raise RuntimeError("Generator did not preserve the fixed output canvas")
            eos_ids = set()
            if isinstance(self._tokenizer.eos_token_id, int):
                eos_ids.add(self._tokenizer.eos_token_id)
            eot = self._tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if isinstance(eot, int) and eot != self._tokenizer.unk_token_id:
                eos_ids.add(eot)
            stop = next((i for i, token in enumerate(raw_ids) if token in eos_ids), len(raw_ids))
            response_ids = raw_ids[:stop]
            masks = sum(token == 126336 for token in raw_ids)
            result = {
                "schema": "safediff_acceleration_result_v1",
                "mode": mode,
                "prompt": prompt,
                "system_prompt": system_prompt,
                "input_sha256": _json_hash(messages),
                "prompt_token_ids": prompt_ids,
                "prompt_token_ids_sha256": _json_hash(prompt_ids),
                "response": self._tokenizer.decode(response_ids, skip_special_tokens=True),
                "raw_text": self._tokenizer.decode(raw_ids, skip_special_tokens=False),
                "generated_token_ids": raw_ids,
                "response_token_ids": response_ids,
                "first_eos_index": stop if stop < len(raw_ids) else None,
                "eos_token_ids": sorted(eos_ids),
                "residual_masks": masks,
                "residual_masks_before_first_eos": sum(token == 126336 for token in response_ids),
                "completion_status": "complete" if not masks else "residual_masks",
                "generation_seconds": elapsed,
                "peak_cuda_memory_allocated_bytes": peak_memory,
                "model_was_loaded_before_request": was_loaded,
                "load_seconds_this_request": 0.0 if was_loaded else self._runtime["model_load_seconds"],
                "timing_definition": {
                    "generation_seconds": "CUDA-synchronized decoder wall time including observers and decoder diagnostics; excludes loading/tokenization/decoding/scoring",
                    "request_seconds": "wall time for this call including lazy loading if needed, tokenization, seeding, generation and text decoding; excludes caller I/O and judging",
                    "comparison_note": "Generation time includes each decoder's observers and diagnostics",
                },
                "config": config,
                "runtime": deepcopy(self._runtime),
                "provenance": deepcopy(self._provenance),
                **metrics,
            }
            if details is not None:
                if details["nfe"] != metrics["nfe"] or details["query_positions"] != metrics["query_positions"]:
                    raise RuntimeError("Instant diagnostics disagree with the common observer")
                result["instant_diagnostics"] = {
                    key: details[key] for key in ("groups", "selection", "berm_remasked_tokens",
                                                  "estimated_main_matmul_flops", "residual_mask_positions")
                }
            result["request_seconds"] = time.perf_counter() - started
            return result

    def close(self):
        """Release this backend's loaded model; the object can be loaded again."""
        with self._lock:
            if self._model is not None:
                device = self._model.device
                self._model = None
                self._tokenizer = None
                self._runtime = None
                self._matmul_spec = None
                self._provenance = None
                import torch
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
