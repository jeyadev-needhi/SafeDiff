from types import SimpleNamespace
import unittest
from unittest.mock import DEFAULT, Mock, patch

import torch

from acceleration import (
    LLaDABackend, PLANS, POLICIES, describe_config, describe_policy_config, mode_for_policy, vendor,
)


class FakeModel(torch.nn.Module):
    device = torch.device("cpu")
    config = SimpleNamespace(max_sequence_length=1024)

    def forward(self, tokens, *, use_cache):
        cache = None
        if use_cache:
            kv = torch.empty(1, 1, tokens.shape[1], 1)
            cache = ((kv, kv),)
        return SimpleNamespace(past_key_values=cache)


class FakeTokenizer:
    eos_token_id = 2
    unk_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return torch.tensor([[10, 11]])

    def convert_tokens_to_ids(self, token):
        return self.unk_token_id

    def decode(self, ids, **kwargs):
        return " ".join(str(token) for token in ids)


def fake_canvas(inputs, gen_length, token):
    return torch.cat((inputs, torch.full((1, gen_length), token)), dim=1)


def fake_fastdllm(model, inputs, *, threshold, gen_length, **kwargs):
    canvas = fake_canvas(inputs, gen_length, int(round(threshold * 10)))
    model(canvas, use_cache=True)
    return canvas, 1


def fake_instant(model, inputs, *, gen_length, **kwargs):
    canvas = fake_canvas(inputs, gen_length, 8)
    model(canvas, use_cache=True)
    return canvas, 1, {
        "nfe": 1, "query_positions": canvas.shape[1], "groups": [], "selection": [],
        "berm_remasked_tokens": 0, "estimated_main_matmul_flops": None, "residual_mask_positions": [],
    }


def fake_llada(model, inputs, *, gen_length, steps, **kwargs):
    canvas = fake_canvas(inputs, gen_length, 5)
    for _ in range(steps):
        model(canvas)
    return canvas


class PlanTests(unittest.TestCase):
    def test_policy_configs_keep_plan_a_defaults_and_resolve_plan_b(self):
        self.assertEqual(PLANS, ("A", "B"))
        for policy, mode in zip(POLICIES, ("instant", "fastdllm", "llada"), strict=True):
            with self.subTest(policy=policy):
                self.assertEqual(mode_for_policy(policy), mode)
                self.assertEqual(describe_policy_config(policy), {
                    **describe_config(mode), "plan": "A", "policy": policy,
                })
                self.assertEqual(mode_for_policy(policy, plan="B"), "fastdllm")
        self.assertEqual(describe_policy_config("NORMAL"), {
            **describe_policy_config("SAFE", plan="B"), "plan": "A", "policy": "NORMAL",
        })

    def test_invalid_policy_settings_are_rejected_before_loading(self):
        backend = LLaDABackend("unused")
        for overrides, message in (
            ({"plan": "C"}, "plan"), ({"policy": "unknown"}, "policy"),
            ({"plan": "B", "gen_length": 33}, "gen_length"), ({"seed": -1}, "seed"),
        ):
            with self.subTest(overrides=overrides), patch.object(backend, "load") as load:
                with self.assertRaisesRegex(ValueError, message):
                    backend.generate_for_policy("prompt", **{"policy": "SAFE", **overrides})
                load.assert_not_called()

    def test_selected_thresholds_reach_decoder_and_results(self):
        backend = LLaDABackend("unused")
        backend._model = FakeModel()
        backend._tokenizer = FakeTokenizer()
        backend._runtime = {}
        fastdllm = Mock(side_effect=fake_fastdllm)
        instant = Mock(side_effect=fake_instant)
        llada = Mock(side_effect=fake_llada)

        with patch.multiple(
            vendor, create=True, fastdllm=SimpleNamespace(generate_with_dual_cache=fastdllm),
            instant=SimpleNamespace(generate_generic_k=instant), llada=SimpleNamespace(generate=llada),
        ), patch.multiple(
            torch.cuda, manual_seed_all=DEFAULT, reset_peak_memory_stats=DEFAULT,
            synchronize=DEFAULT, max_memory_allocated=Mock(return_value=0),
        ):
            for policy, threshold in zip(POLICIES, (0.7, 0.8, 0.9), strict=True):
                with self.subTest(policy=policy):
                    result = backend.generate_for_policy("prompt", policy=policy, plan="B", gen_length=64)
                    self.assertEqual((result["plan"], result["policy"], result["mode"]), ("B", policy, "fastdllm"))
                    self.assertEqual(result["config"], describe_policy_config(policy, plan="B", gen_length=64))
                    self.assertEqual(result["generated_token_ids"], [int(round(threshold * 10))] * 64)
                    self.assertEqual(fastdllm.call_args.kwargs, {
                        "gen_length": 64, "block_length": 32, "temperature": 0.0, "remasking": "low_confidence",
                        "mask_id": 126336, "steps": 64, "threshold": threshold, "factor": None,
                    })
            self.assertEqual(fastdllm.call_count, 3)
            instant.assert_not_called()
            llada.assert_not_called()

            normal = backend.generate_for_policy("prompt", policy="NORMAL", gen_length=64)
            self.assertEqual(fastdllm.call_args.kwargs["threshold"], 0.9)
            direct = backend.generate("prompt", mode="fastdllm", gen_length=64)
            self.assertEqual(normal["generated_token_ids"], direct["generated_token_ids"])
            self.assertEqual(direct["config"], describe_config("fastdllm", gen_length=64))
            self.assertNotIn("plan", direct)
            self.assertNotIn("policy", direct)

            fast = backend.generate_for_policy("prompt", policy="FAST", gen_length=64)
            self.assertEqual((fast["plan"], fast["mode"]), ("A", "instant"))
            self.assertEqual(instant.call_args.kwargs, {
                "gen_length": 64, "block_length": 32, "temperature": 0.0, "remasking": "low_confidence",
                "mask_id": 126336, "parallel_k": 8, "threshold": 0.9, "berm_mu": 16,
                "max_rounds_per_group": 64, "matmul_spec": None,
            })
            safe = backend.generate_for_policy("prompt", policy="SAFE", gen_length=33)
            self.assertEqual((safe["plan"], safe["mode"], safe["nfe"]), ("A", "llada", 33))
            self.assertEqual(llada.call_args.kwargs, {
                "gen_length": 33, "block_length": 33, "temperature": 0.0, "remasking": "low_confidence",
                "mask_id": 126336, "steps": 33, "cfg_scale": 0.0,
            })
