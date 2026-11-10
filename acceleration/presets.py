"""Frozen Plan A/B generation presets, independent of the risk controller.

                   FAST                  NORMAL                SAFE
Plan A:            ALL8                  Fast-dLLM 0.9         vanilla LLaDA
Plan B:            Fast-dLLM 0.7          Fast-dLLM 0.8         Fast-dLLM 0.9

A changes the generation strategy; B changes only token confidence threshold.
The controller supplies a policy; this module neither classifies risk nor reads
dataset labels. SAFE names a compute preset, not a demonstrated safety property.
A NORMAL and B SAFE resolve to the same decoder settings. Six named slots are
therefore five distinct configurations. Default plan A preserves the prior API.
"""

MODES = ("instant", "fastdllm", "llada")
PLANS = ("A", "B")
POLICY_TO_MODE = {"FAST": "instant", "NORMAL": "fastdllm", "SAFE": "llada"}
POLICIES = tuple(POLICY_TO_MODE)
_PLAN_B_THRESHOLDS = {"FAST": 0.7, "NORMAL": 0.8, "SAFE": 0.9}


def mode_for_policy(policy, plan="A"):
    """Map a policy in a fixed experiment plan to its generation backend."""
    if not isinstance(plan, str) or plan not in PLANS:
        raise ValueError("plan must be one of: " + ", ".join(PLANS))
    if not isinstance(policy, str) or policy not in POLICY_TO_MODE:
        raise ValueError("policy must be one of: " + ", ".join(POLICIES))
    return POLICY_TO_MODE[policy] if plan == "A" else "fastdllm"


def describe_policy_config(policy, plan="A", gen_length=256, seed=42):
    """Resolve and record all settings; never infer a policy from prompt data."""
    mode = mode_for_policy(policy, plan=plan)
    config = describe_config(mode, gen_length=gen_length, seed=seed)
    if plan == "B":
        # Confidence is the decoder's token-acceptance criterion, not a risk
        # score. Cache, block length, seed and generation budget stay fixed.
        config["threshold"] = _PLAN_B_THRESHOLDS[policy]
    return {**config, "plan": plan, "policy": policy}


def describe_config(mode, gen_length=256, seed=42):
    """Return a new JSON-compatible configuration, validating CPU-only inputs."""
    if mode not in MODES:
        raise ValueError("mode must be one of: " + ", ".join(MODES))
    if mode == "llada":
        if type(gen_length) is not int or gen_length <= 0:
            raise ValueError("gen_length must be a positive integer")
    elif type(gen_length) is not int or gen_length < 32 or gen_length % 32:
        raise ValueError("gen_length must be a positive multiple of 32")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if mode == "llada":
        # Plan A SAFE: official uncached full-canvas reference. With G output
        # positions this uses one G-token block and G scheduled forward calls.
        # It intentionally differs from Plan B SAFE's cached 32-token blocks.
        return {
            "mode": mode,
            "model_family": "LLaDA-8B-Instruct",
            "generator": "official_llada_vanilla",
            "batch_size": 1,
            "gen_length": gen_length,
            "block_length": gen_length,
            "steps": gen_length,
            "temperature": 0.0,
            "cfg_scale": 0.0,
            "remasking": "low_confidence",
            "mask_id": 126336,
            "seed": seed,
            "use_cache": False,
            "dual_cache": False,
            "threshold": None,
            "factor": None,
            "parallel_k": 1,
            "berm_mu": None,
            "max_rounds_per_group": None,
            "eos_early_stop": False,
            "output_policy": "fixed_canvas; decode only the prefix before first EOS/EOT",
            "residual_mask_policy": "retain and report; no repair",
            "stopping_rule": "native upstream loop: gen_length forwards over the full generation canvas",
        }
    # Both accelerated strategies use 32-token blocks. The fixed output canvas
    # is completed even if EOS appears; only displayed text is trimmed at EOS.
    common = {
        "mode": mode,
        "model_family": "LLaDA-8B-Instruct",
        "batch_size": 1,
        "gen_length": gen_length,
        "block_length": 32,
        "temperature": 0.0,
        "remasking": "low_confidence",
        "mask_id": 126336,
        "seed": seed,
        "threshold": 0.9,
        "dual_cache": True,
        "factor": None,
        "eos_early_stop": False,
        "output_policy": "fixed_canvas; decode only the prefix before first EOS/EOT",
        "residual_mask_policy": "retain and report; no repair",
    }
    if mode == "instant":
        # Plan A FAST: process up to eight adjacent blocks together, with BERM
        # mu=16 and threshold=0.9. Historical speed gains include quality loss.
        common.update(
            generator="contiguous_all8",
            parallel_k=8,
            berm_mu=16,
            max_rounds_per_group=64,
            steps=None,
            group_layout="contiguous nonoverlapping groups; keep shorter final group",
            refresh_interval=0,
            kv_delta_gamma=0.0,
            adaptive_alpha=None,
            stopping_rule="each group ends when unmasked or after 64 forwards including warmup",
        )
    else:
        # Plan A NORMAL and all Plan B tiers: native sequential dual-cache
        # Fast-dLLM. `steps` is the whole-sequence budget, not a per-block budget.
        common.update(
            generator="official_fastdllm_v1_dual_cache",
            parallel_k=1,
            berm_mu=0,
            steps=gen_length,
            max_rounds_per_group=None,
            stopping_rule="native upstream loop: at most 32 forwards per block including warmup",
        )
    return common
