"""Run LLaDA generation by mode or policy name.

Input/output helpers use the Python standard library. The backend imports
model dependencies when generation starts.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import uuid

from acceleration.presets import MODES, POLICIES, describe_config, mode_for_policy


PROMPT_FIELDS = ("text", "prompt", "vanilla prompt")
LEGACY_COMPARE_MODES = ("instant", "fastdllm")
RUN_SCHEMA = "safediff.acceleration.run.v1"


def extract_prompt(record: dict, *, location: str = "record") -> str:
    """Read a nonempty prompt from the supported input fields."""
    if not isinstance(record, dict):
        raise ValueError(f"{location}: expected a JSON object")
    values = {}
    for field in PROMPT_FIELDS:
        if field not in record:
            continue
        if not isinstance(record[field], str):
            raise ValueError(f"{location}: {field!r} must be a string")
        values[field] = record[field]
    if not values or not any(value.strip() for value in values.values()):
        raise ValueError(f"{location}: no nonempty prompt in {PROMPT_FIELDS}")
    if len(set(values.values())) != 1:
        raise ValueError(f"{location}: conflicting prompt fields: {', '.join(values)}")
    return next(iter(values.values()))


def load_records(*, input_path: str | None = None, prompt: str | None = None) -> list[dict]:
    """Validate the entire input before loading a model or creating results."""
    if (input_path is None) == (prompt is None):
        raise ValueError("provide exactly one of --prompt or --input")
    if prompt is not None:
        raw_records = [{"id": 0, "prompt": prompt}]
    else:
        path = Path(input_path)
        if path.suffix.lower() == ".jsonl":
            raw_records = []
            for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    raw_records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
        else:
            content = json.loads(path.read_text(encoding="utf-8-sig"))
            raw_records = content if isinstance(content, list) else [content]
    if not raw_records:
        raise ValueError("input contains no records")
    records = []
    for index, record in enumerate(raw_records):
        normalized = extract_prompt(record, location=f"record {index}")
        records.append({"index": index, "id": record.get("id", index),
                        "prompt": normalized, "source_record": record})
    return records


class ResultWriter:
    """Create a new result file and save each completed record.

    JSON uses an atomic snapshot replacement. JSONL flushes and fsyncs each line.
    Opening an existing output raises FileExistsError.
    """

    def __init__(self, output: str):
        self.path = Path(output)
        if self.path.suffix.lower() not in (".json", ".jsonl"):
            raise ValueError("--output must end in .json or .jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows = []
        self.stream = self.path.open("x", encoding="utf-8", newline="\n")
        if self.path.suffix.lower() == ".json":
            self.stream.write("[]\n")
            self.stream.flush()
            os.fsync(self.stream.fileno())
            self.stream.close()

    def write(self, record: dict) -> None:
        if self.path.suffix.lower() == ".jsonl":
            self.stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            self.stream.flush()
            os.fsync(self.stream.fileno())
            return
        self.rows.append(record)
        temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(self.rows, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def close(self) -> None:
        self.stream.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--revision", help="Model revision or commit (recorded by the backend)")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--mode", choices=(*MODES, "compare", "compare-all"), default="instant",
                           help="compare retains the two acceleration modes; compare-all also includes vanilla LLaDA")
    selection.add_argument("--policy", choices=POLICIES,
                           help="Explicit preset: FAST=instant, NORMAL=fastdllm, SAFE=vanilla llada")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--prompt", help="One literal user prompt")
    source.add_argument("--input", help="JSON object/list or JSONL, with text/prompt/vanilla prompt")
    parser.add_argument("--output", help="New .json or .jsonl result file; existing files are rejected")
    parser.add_argument("--gen-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--allow-download", action="store_true",
                        help="Allow model download; default is local files/cache only")
    parser.add_argument("--no-flash-attention", action="store_true")
    parser.add_argument("--describe-config", action="store_true",
                        help="Print fixed settings and exit without model loading")
    return parser


def selected_modes(args):
    if args.policy is not None:
        return (mode_for_policy(args.policy),)
    if args.mode == "compare":
        return LEGACY_COMPARE_MODES
    if args.mode == "compare-all":
        return MODES
    return (args.mode,)


def run_jobs(args, records: list[dict], *, backend_factory=None) -> int:
    if not args.output:
        raise ValueError("--output is required for generation")
    modes = selected_modes(args)
    for mode in modes:
        describe_config(mode, gen_length=args.gen_length, seed=args.seed)
    if backend_factory is None:
        from acceleration import LLaDABackend
        backend_factory = LLaDABackend
    # Open the output before loading the model to catch path conflicts early.
    writer = ResultWriter(args.output)
    backend = None
    run_id = uuid.uuid4().hex
    exit_code = 0
    try:
        backend = backend_factory(
            args.model_path, revision=args.revision, device=args.device,
            local_files_only=not args.allow_download, dtype=args.dtype,
            flash_attention=not args.no_flash_attention,
        )
        for record in records:
            for mode in modes:
                result = {
                    "schema": RUN_SCHEMA, "run_id": run_id,
                    "input_index": record["index"], "input_id": record["id"],
                    "input": record["source_record"], "prompt": record["prompt"],
                    "mode": mode, "seed": args.seed, "gen_length": args.gen_length,
                    "system_prompt": args.system_prompt,
                }
                if args.policy is not None:
                    result["policy"] = args.policy
                try:
                    options = {"gen_length": args.gen_length, "seed": args.seed,
                               "system_prompt": args.system_prompt}
                    if args.policy is not None:
                        generation = backend.generate_for_policy(record["prompt"], policy=args.policy, **options)
                    else:
                        generation = backend.generate(record["prompt"], mode=mode, **options)
                except Exception as exc:
                    result.update(status="error", error={"type": type(exc).__name__, "message": str(exc)})
                    writer.write(result)
                    raise
                incomplete = (generation.get("completion_status") == "residual_masks"
                              or generation.get("residual_masks", 0) > 0)
                status = "incomplete" if incomplete else "ok"
                if incomplete:
                    exit_code = 2
                result.update(status=status, generation=generation)
                writer.write(result)
                print(f"[{record['index'] + 1}/{len(records)}] {mode}: {status}, saved", file=sys.stderr)
    finally:
        try:
            if backend is not None:
                backend.close()
        finally:
            writer.close()
    return exit_code


def main(argv=None, *, backend_factory=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.gen_length <= 0:
            raise ValueError("--gen-length must be positive")
        # describe_config validates backend restrictions before opening an output.
        modes = selected_modes(args)
        configs = {mode: describe_config(mode, gen_length=args.gen_length, seed=args.seed)
                   for mode in modes}
        if args.describe_config:
            description = configs if args.policy is None else {"policy": args.policy, "mode": modes[0], "config": configs[modes[0]]}
            print(json.dumps(description, ensure_ascii=False, indent=2, allow_nan=False))
            return 0
        records = load_records(input_path=args.input, prompt=args.prompt)
        return run_jobs(args, records, backend_factory=backend_factory)
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
