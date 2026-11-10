import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from acceleration.presets import describe_policy_config
from analysis.smoke_acceleration_plans import build_parser, run_smoke


class FakeBackend:
    def __init__(self, model_path, **options):
        self.options = options
        self.loaded = False
        self.closed = False
        self.calls = 0

    def load(self):
        self.loaded = True

    def runtime_info(self):
        return {"gpu_name": "fake"}

    def generate_for_policy(self, prompt, *, plan, policy, gen_length, seed, system_prompt=None):
        self.calls += 1
        config = describe_policy_config(policy, plan=plan, gen_length=gen_length, seed=seed)
        mode = config["mode"]
        token = round((config["threshold"] or 0) * 100)
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        return {
            "plan": plan, "policy": policy, "mode": mode, "config": config,
            "generated_token_ids": [token] * gen_length,
            "completion_status": "complete", "residual_masks": 0,
            "input_sha256": digest, "prompt_token_ids_sha256": digest,
            "nfe": token + 1, "actual_forward_calls": token + 1,
            "model_was_loaded_before_request": self.loaded, "load_seconds_this_request": 0,
            "generation_seconds": 0.01,
        }

    def close(self):
        self.closed = True


class PlanSmokeTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        folder = temporary.name
        self.output = Path(folder) / "calls.jsonl"
        self.summary = Path(folder) / "summary.json"
        self.args = build_parser().parse_args([
            "--model-path", "fake", "--output", str(self.output),
            "--summary", str(self.summary), "--gen-length", "32",
        ])
        self.backends = []

    def run_fake(self, backend_type):
        def backend_factory(*args, **kwargs):
            backend = backend_type(*args, **kwargs)
            self.backends.append(backend)
            return backend

        return run_smoke(self.args, backend_factory=backend_factory)

    def read_results(self):
        rows = [json.loads(line) for line in self.output.read_text(encoding="utf-8").splitlines()]
        summary = json.loads(self.summary.read_text(encoding="utf-8"))
        return rows, summary

    def test_checks_all_arms_and_repeats_on_one_loaded_backend(self):
        self.assertEqual(self.run_fake(FakeBackend), 0)
        rows, summary = self.read_results()
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(len(rows), 24)
        self.assertEqual(len(summary["comparisons"]), 16)
        self.assertEqual(len(self.backends), 1)
        self.assertEqual(self.backends[0].calls, 24)
        self.assertTrue(self.backends[0].options["local_files_only"])
        self.assertTrue(self.backends[0].closed)

    def test_records_mismatched_outputs_and_checks(self):
        class MismatchedBackend(FakeBackend):
            def generate_for_policy(self, prompt, **kwargs):
                result = super().generate_for_policy(prompt, **kwargs)
                if self.calls == 11:
                    result["generated_token_ids"] = [0] * kwargs["gen_length"]
                    result["nfe"] -= 1
                if self.calls == 13:
                    result["residual_masks"] = 1
                    result["input_sha256"] = "wrong"
                    result["config"] = {}
                    result["generated_token_ids"] = [1] * kwargs["gen_length"]
                return result

        self.assertEqual(self.run_fake(MismatchedBackend), 2)
        rows, summary = self.read_results()
        self.assertEqual(len(rows), 24)
        self.assertEqual(summary["status"], "failed")
        self.assertFalse(rows[10]["checks"]["matching_forward_count"])
        self.assertFalse(rows[12]["checks"]["completed_without_masks"])
        self.assertFalse(rows[12]["checks"]["matching_config"])
        comparisons = {
            item["comparison"]: item["checks"] for item in summary["comparisons"]
            if item["prompt_id"] == "arithmetic"
        }
        self.assertFalse(comparisons["A NORMAL and B SAFE 0.9 bridge"]["equal_generated_tokens"])
        self.assertFalse(comparisons["A NORMAL and B SAFE 0.9 bridge"]["equal_nfe"])
        self.assertFalse(comparisons["A SAFE fixed and mixed"]["equal_generated_tokens"])
        self.assertFalse(comparisons["all arms use the same input"]["equal_input_hashes"])

    def test_preserves_completed_rows_on_error_and_refuses_overwrite(self):
        class FailingBackend(FakeBackend):
            def generate_for_policy(self, prompt, **kwargs):
                if self.calls == 1:
                    raise RuntimeError("test failure")
                return super().generate_for_policy(prompt, **kwargs)

        with self.assertRaisesRegex(RuntimeError, "test failure"):
            self.run_fake(FailingBackend)
        rows, summary = self.read_results()
        self.assertEqual([row["status"] for row in rows], ["passed", "error"])
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["error"]["type"], "RuntimeError")
        with self.assertRaises(FileExistsError):
            self.run_fake(FakeBackend)
        self.assertEqual(self.read_results(), (rows, summary))
        self.assertTrue(self.backends[0].closed)


if __name__ == "__main__":
    unittest.main()
