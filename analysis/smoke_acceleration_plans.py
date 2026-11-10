"""Check both LLaDA acceleration plans on two benign prompts with one GPU model."""

from __future__ import annotations

import argparse
from contextlib import closing
import json
from pathlib import Path
import sys

from analysis.run_acceleration import ResultWriter


PROMPTS = {
    "arithmetic": "What is 17 + 25? Give a short answer.",
    "explanation": "In one sentence, explain why the daytime sky usually looks blue.",
}
POLICIES = ("FAST", "NORMAL", "SAFE")
FIXED_ARMS = tuple((plan, policy) for plan in ("A", "B") for policy in POLICIES)
MIXED_POLICIES = {
    "arithmetic": ("SAFE", "FAST", "NORMAL"),
    "explanation": ("NORMAL", "SAFE", "FAST"),
}
SEED = 42


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="LLaDA-8B-Instruct checkpoint or cached model ID")
    parser.add_argument("--revision", help="Checkpoint revision")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gen-length", type=int, default=256)
    parser.add_argument("--no-flash-attention", action="store_true")
    parser.add_argument("--allow-download", action="store_true", help="Allow fetching weights; off by default")
    parser.add_argument("--output", type=Path, required=True, help="New JSONL file with every generation")
    parser.add_argument("--summary", type=Path, required=True, help="New JSON file with smoke check outcomes")
    return parser


def _record(writer, summary, backend, expected, *, phase, prompt_id, plan, policy):
    record = {"phase": phase, "prompt_id": prompt_id, "plan": plan, "policy": policy}
    try:
        # Both phases exercise only our backend API. Mixed order checks model
        # reuse across presets; it does not test or modify the legacy router.
        generation = backend.generate_for_policy(
            PROMPTS[prompt_id], plan=plan, policy=policy,
            gen_length=expected["gen_length"], seed=expected["seed"],
        )
    except Exception as exc:
        writer.write({**record, "status": "error", "error": str(exc)})
        raise
    token_ids = generation.get("generated_token_ids")
    nfe = generation.get("nfe")
    checks = {
        "matching_config": generation.get("config") == expected,
        "matching_plan_policy_and_mode": (
            (generation.get("plan"), generation.get("policy"), generation.get("mode"))
            == (plan, policy, expected["mode"])
        ),
        "recorded_input_hashes": bool(generation.get("input_sha256") and generation.get("prompt_token_ids_sha256")),
        "fixed_canvas": isinstance(token_ids, list) and len(token_ids) == expected["gen_length"],
        "completed_without_masks": (
            generation.get("completion_status") == "complete" and generation.get("residual_masks") == 0
        ),
        "matching_forward_count": type(nfe) is int and generation.get("actual_forward_calls") == nfe > 0,
        "warm_model": (
            generation.get("model_was_loaded_before_request") is True
            and generation.get("load_seconds_this_request") == 0
        ),
    }
    status = "passed" if all(checks.values()) else "failed"
    writer.write({**record, "status": status, "checks": checks, "generation": generation})
    summary["calls"].append({
        **record, "status": status, "checks": checks, "nfe": nfe,
        "generation_seconds": generation.get("generation_seconds"),
    })
    return generation


def _matching_generation(left, right):
    return {
        "equal_generated_tokens": left.get("generated_token_ids") == right.get("generated_token_ids"),
        "equal_nfe": left.get("nfe") == right.get("nfe"),
    }


def run_smoke(args, *, backend_factory=None):
    from acceleration import LLaDABackend
    from acceleration.presets import describe_policy_config

    if args.output.suffix.lower() != ".jsonl" or args.summary.suffix.lower() != ".json":
        raise ValueError("--output must be .jsonl and --summary must be .json")
    for path in (args.output, args.summary):
        if path.exists():
            raise FileExistsError(path)
    configs = {
        (plan, policy): describe_policy_config(policy, plan=plan, gen_length=args.gen_length, seed=SEED)
        for plan, policy in FIXED_ARMS
    }
    backend_factory = LLaDABackend if backend_factory is None else backend_factory
    summary = {
        "status": "failed",
        "scope": "Integration smoke only; two benign prompts, no quality, safety or speedup estimate.",
        "routing_scope": "Explicit preset calls only; no Guard classification or legacy adaptive runner.",
        "gen_length": args.gen_length,
        "seed": SEED,
        "output": str(args.output),
        "calls": [],
        "comparisons": [],
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("x", encoding="utf-8", newline="\n") as report:
        try:
            with closing(ResultWriter(str(args.output))) as writer:
                backend = backend_factory(
                    args.model_path, revision=args.revision, device=args.device,
                    local_files_only=not args.allow_download, flash_attention=not args.no_flash_attention,
                )
                with closing(backend):
                    backend.load()
                    summary["runtime"] = backend.runtime_info()
                    _run_calls(writer, summary, backend, configs)
        except Exception as exc:
            summary["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        else:
            if (all(call["status"] == "passed" for call in summary["calls"])
                    and all(all(item["checks"].values()) for item in summary["comparisons"])):
                summary["status"] = "passed"
        finally:
            json.dump(summary, report, ensure_ascii=False, indent=2, allow_nan=False)
            report.write("\n")
    return 0 if summary["status"] == "passed" else 2


def _run_calls(writer, summary, backend, configs):
    fixed = {}
    for plan, policy in FIXED_ARMS:
        for prompt_id in PROMPTS:
            fixed[prompt_id, plan, policy] = _record(
                writer, summary, backend, configs[plan, policy],
                phase="fixed", prompt_id=prompt_id, plan=plan, policy=policy,
            )

    for prompt_id, policies in MIXED_POLICIES.items():
        baseline = fixed[prompt_id, "A", "NORMAL"]
        arm_results = [fixed[prompt_id, plan, policy] for plan, policy in FIXED_ARMS]
        summary["comparisons"].append({
            "prompt_id": prompt_id, "comparison": "A NORMAL and B SAFE 0.9 bridge",
            "checks": _matching_generation(baseline, fixed[prompt_id, "B", "SAFE"]),
        })
        for plan in ("A", "B"):
            for policy in policies:
                repeated = _record(
                    writer, summary, backend, configs[plan, policy],
                    phase="mixed", prompt_id=prompt_id, plan=plan, policy=policy,
                )
                arm_results.append(repeated)
                summary["comparisons"].append({
                    "prompt_id": prompt_id, "comparison": f"{plan} {policy} fixed and mixed",
                    "checks": _matching_generation(fixed[prompt_id, plan, policy], repeated),
                })
        summary["comparisons"].append({
            "prompt_id": prompt_id, "comparison": "all arms use the same input",
            "checks": {
                "equal_input_hashes": len({row.get("input_sha256") for row in arm_results}) == 1,
                "equal_prompt_token_hashes": len({row.get("prompt_token_ids_sha256") for row in arm_results}) == 1,
            },
        })


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return run_smoke(args)
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
