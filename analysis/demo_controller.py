#!/usr/bin/env python3
"""Interactive generation with Qwen guard routing or a supplied policy.

Dry runs use a supplied policy or guard output. Models are loaded on first use.
"""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acceleration import LLaDABackend, POLICIES, describe_config, mode_for_policy
from analysis.run_acceleration import ResultWriter

MODEL_ID = "Qwen/Qwen3Guard-Gen-0.6B"


class GuardClassifier:
    """Load the Qwen guard on first use; downloads are opt-in."""

    def __init__(self, model_path=MODEL_ID, *, allow_download=False):
        self.model_path = model_path
        self.local_files_only = not allow_download
        self.tokenizer = self.model = None

    def classify(self, prompt):
        import torch
        if self.model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, local_files_only=self.local_files_only)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, torch_dtype="auto", device_map="auto",
                local_files_only=self.local_files_only,
            ).eval()
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=32, do_sample=False)
        generated = output[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


def get_policy(result):
    """Map guard output to a risk label, policy and backend mode."""
    text = result.lower()
    if "unsafe" in text:
        risk, policy = "HIGH", "SAFE"
    elif "controversial" in text:
        risk, policy = "MEDIUM", "NORMAL"
    elif "safe" in text:
        risk, policy = "LOW", "FAST"
    else:
        risk, policy = "UNKNOWN", "NORMAL"
    return risk, policy, mode_for_policy(policy)


class DemoController:
    def __init__(self, model_path="GSAI-ML/LLaDA-8B-Instruct", *, revision=None,
                 device="cuda:0", allow_download=False, guard_model_path=MODEL_ID,
                 backend_factory=LLaDABackend, classifier=None):
        self.model_path = model_path
        self.options = dict(revision=revision, device=device,
                            local_files_only=not allow_download)
        self.backend_factory = backend_factory
        self.backend = None
        self.classifier = classifier
        self.guard_model_path = guard_model_path
        self.allow_download = allow_download

    def run(self, prompt, *, policy=None, guard_output=None, dry_run=False,
            gen_length=256, seed=42, system_prompt=None):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a nonempty string")
        if policy is not None and guard_output is not None:
            raise ValueError("Choose either an explicit policy or supplied guard output")
        modes = ([mode_for_policy(policy)] if policy is not None else
                 [get_policy(guard_output)[2]] if guard_output is not None else
                 [mode_for_policy(name) for name in POLICIES])
        for mode in modes:
            describe_config(mode, gen_length=gen_length, seed=seed)
        if system_prompt is not None and not isinstance(system_prompt, str):
            raise ValueError("system_prompt must be a string or None")
        if policy is not None:
            risk, mode = None, mode_for_policy(policy)
            routing_source = "explicit_policy"
        else:
            routing_source = "supplied_guard_output"
            if guard_output is None:
                if dry_run:
                    raise ValueError("A model-free dry run needs --policy or --guard-output")
                if self.classifier is None:
                    self.classifier = GuardClassifier(
                        self.guard_model_path, allow_download=self.allow_download)
                guard_output = self.classifier.classify(prompt)
                routing_source = "existing_qwen_guard_output_heuristic"
            risk, policy, mode = get_policy(guard_output)
        result = {"prompt": prompt, "policy": policy, "mode": mode,
                  "routing_source": routing_source, "risk_level": risk,
                  "guard_output": guard_output, "dry_run": dry_run,
                  "routing_note": "Policy mapping: FAST=instant, NORMAL=fastdllm, SAFE=llada"}
        if not dry_run:
            if self.backend is None:
                self.backend = self.backend_factory(self.model_path, **self.options)
            result["generation"] = self.backend.generate_for_policy(
                prompt, policy=policy, gen_length=gen_length, seed=seed,
                system_prompt=system_prompt,
            )
            if result["generation"].get("policy") != policy or result["generation"].get("mode") != mode:
                raise RuntimeError("Backend result disagrees with the public policy mapping")
            generation = result["generation"]
            incomplete = generation.get("residual_masks", 0) > 0 or generation.get("completion_status") == "residual_masks"
            result["status"] = "incomplete" if incomplete else "ok"
        else:
            result["status"] = "dry_run"
        return result

    def close(self):
        if self.backend is not None:
            self.backend.close()
            self.backend = None


def main(argv=None, *, backend_factory=LLaDABackend):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", "--model_path", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--guard-model-path", default=MODEL_ID)
    parser.add_argument("--revision")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--prompt")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--policy", choices=POLICIES)
    selection.add_argument("--guard-output")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gen-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--system-prompt")
    parser.add_argument("--output", help="Optional new .json/.jsonl output; each result is persisted")
    args = parser.parse_args(argv)
    if args.dry_run and args.prompt is None:
        print(json.dumps({policy: mode_for_policy(policy) for policy in POLICIES}, indent=2))
        return
    if args.dry_run and args.policy is None and args.guard_output is None:
        parser.error("--dry-run with --prompt requires --policy or --guard-output")
    controller = DemoController(
        args.model_path, revision=args.revision, device=args.device,
        allow_download=args.allow_download, guard_model_path=args.guard_model_path,
        backend_factory=backend_factory,
    )
    with ExitStack() as stack:
        stack.callback(controller.close)
        writer = ResultWriter(args.output) if args.output else None
        if writer is not None:
            stack.callback(writer.close)
        while True:
            if args.prompt is not None:
                prompt = args.prompt
            else:
                try:
                    prompt = input("Enter prompt (or 'exit'): ")
                except EOFError:
                    break
                if prompt.strip().lower() == "exit":
                    break
                if not prompt.strip():
                    continue
            try:
                result = controller.run(
                    prompt, policy=args.policy, guard_output=args.guard_output,
                    dry_run=args.dry_run, gen_length=args.gen_length, seed=args.seed,
                    system_prompt=args.system_prompt,
                )
            except Exception as exc:
                if writer is not None:
                    writer.write({"prompt": prompt, "policy": args.policy,
                                  "status": "error", "error": {"type": type(exc).__name__, "message": str(exc)}})
                raise
            if writer is not None:
                writer.write(result)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            if args.prompt is not None:
                return result


if __name__ == "__main__":
    main()
