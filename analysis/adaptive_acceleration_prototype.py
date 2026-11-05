#!/usr/bin/env python3
"""Route dataset risk labels to FAST, NORMAL or SAFE generation.

This oracle experiment reads risk labels from the input records.
"""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acceleration import LLaDABackend, POLICIES, describe_config, mode_for_policy
from analysis.run_acceleration import ResultWriter


def route(record):
    """Map a dataset risk label to (policy, backend mode)."""
    value = record.get("risk_level", record.get("label"))
    if value is None or isinstance(value, bool):
        raise ValueError("Oracle routing requires an explicit integer risk_level or label")
    level = int(value)
    policy = "FAST" if level <= 0 else "NORMAL" if level == 1 else "SAFE"
    return policy, mode_for_policy(policy)


def _prompt(record):
    values = [record[key] for key in ("prompt", "text", "vanilla prompt") if key in record]
    if not values or any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("Each record requires a nonempty prompt, text or vanilla prompt")
    if any(value != values[0] for value in values[1:]):
        raise ValueError("Conflicting prompt fields")
    return values[0]


def _plan_records(records, *, gen_length, seed, system_prompt):
    if system_prompt is not None and not isinstance(system_prompt, str):
        raise ValueError("system_prompt must be a string or None")
    planned = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Record {index} must be a JSON object")
        policy, mode = route(record)
        describe_config(mode, gen_length=gen_length, seed=seed)
        planned.append((record, _prompt(record), policy, mode))
    return planned


def run_records(records, backend, *, gen_length=256, seed=42, system_prompt=None,
                on_result=None):
    """Use one caller-owned backend, persisting each result through the callback."""
    planned = _plan_records(records, gen_length=gen_length, seed=seed,
                            system_prompt=system_prompt)
    results = []
    for index, (record, prompt, policy, mode) in enumerate(planned):
        result = {
            "schema": "safediff.oracle_policy_prototype.v1",
            "record_index": index, "id": record.get("id", index),
            "source_record": dict(record), "adaptive_policy": policy,
            "policy": policy, "mode": mode,
            "routing_source": "oracle_dataset_risk_label",
        }
        try:
            generation = backend.generate_for_policy(
                prompt, policy=policy, gen_length=gen_length, seed=seed,
                system_prompt=system_prompt,
            )
            if generation.get("policy") != policy or generation.get("mode") != mode:
                raise RuntimeError("Backend result disagrees with the public policy mapping")
        except Exception as exc:
            result.update(status="error", error={"type": type(exc).__name__, "message": str(exc)})
            if on_result is not None:
                on_result(result)
            raise
        incomplete = generation.get("residual_masks", 0) > 0 or generation.get("completion_status") == "residual_masks"
        result.update(status="incomplete" if incomplete else "ok", generation=generation)
        if on_result is not None:
            on_result(result)
        results.append(result)
    return results


def main(argv=None, *, backend_factory=LLaDABackend):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="analysis/datasets/safety_eval_v1.json")
    parser.add_argument("--model_path", "--model-path", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--revision")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--gen_length", "--gen-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--system-prompt")
    parser.add_argument("--out_dir", "--out-dir", default="results/adaptive_prototype")
    parser.add_argument("--output", help="New merged .json/.jsonl output; default: out_dir/adaptive_results.json")
    args = parser.parse_args(argv)
    records = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Input must be a JSON list of oracle-labelled records")
    _plan_records(records, gen_length=args.gen_length, seed=args.seed,
                  system_prompt=args.system_prompt)
    out_dir = Path(args.out_dir)
    merged_path = Path(args.output) if args.output else out_dir / "adaptive_results.json"
    started = time.perf_counter()
    with ExitStack() as stack:
        merged_writer = ResultWriter(str(merged_path))
        stack.callback(merged_writer.close)
        policy_writers = {}
        for policy in POLICIES:
            writer = ResultWriter(str(out_dir / f"{policy.lower()}_output.json"))
            stack.callback(writer.close)
            policy_writers[policy] = writer

        def persist(result):
            merged_writer.write(result)
            policy_writers[result["policy"]].write(result)

        backend = backend_factory(
            args.model_path, revision=args.revision, device=args.device,
            local_files_only=not args.allow_download,
        )
        stack.callback(backend.close)
        results = run_records(records, backend, gen_length=args.gen_length,
                              seed=args.seed, system_prompt=args.system_prompt,
                              on_result=persist)
    elapsed = time.perf_counter() - started
    print("Routing source: dataset risk labels (oracle)")
    for policy in POLICIES:
        rows = [row for row in results if row["policy"] == policy]
        nfe = sum(row["generation"]["nfe"] for row in rows)
        print(f"{policy}: mode={mode_for_policy(policy)}, examples={len(rows)}, measured NFE={nfe}")
    print(f"Total prototype wall time, including lazy model loading: {elapsed:.2f}s")
    print(f"Merged output: {merged_path}")
    return results


if __name__ == "__main__":
    main()
