from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

from utility.generate_function_llada import generate


MASK, ANSWER, ALTERNATIVE, PROMPT = range(4)


class RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_parameter("anchor", torch.nn.Parameter(torch.zeros(1)))
        self.inputs = []

    def forward(self, tokens, **kwargs):
        self.inputs.append(tokens.clone())
        logits = torch.zeros((*tokens.shape, 4))
        # Prefer later positions so a future block would be filled first if eligible.
        logits[..., ANSWER] = 8 + torch.arange(tokens.shape[1])
        logits[..., ALTERNATIVE] = 5
        return SimpleNamespace(logits=logits)


class BlockwiseTests(unittest.TestCase):
    def setUp(self):
        self.model = RecordingModel()
        self.tokenizer = lambda *args, **kwargs: SimpleNamespace(input_ids=[ALTERNATIVE])

    def decode(self, prompt, **kwargs):
        return generate(
            self.model,
            self.tokenizer,
            torch.tensor([prompt]),
            gen_length=kwargs.pop("gen_length", 4),
            steps=kwargs.pop("steps", 8),
            block_length=2,
            temperature=0,
            mask_id=MASK,
            **kwargs,
        )[0].tolist()

    def test_only_active_block_changes_and_empty_steps_skip_forward(self):
        self.assertEqual(self.decode([PROMPT], steps=9), [PROMPT, ANSWER, ANSWER, ANSWER, ANSWER])
        self.assertEqual([row[0].tolist() for row in self.model.inputs], [
            [PROMPT, MASK, MASK, MASK, MASK],
            [PROMPT, MASK, ANSWER, MASK, MASK],
            [PROMPT, ANSWER, ANSWER, MASK, MASK],
            [PROMPT, ANSWER, ANSWER, MASK, ANSWER],
        ])

    def test_first_block_fills_inline_masks_but_preserves_protected_positions(self):
        result = self.decode([PROMPT, MASK, MASK], protected_index=torch.tensor([[0, 1, 0]]))
        self.assertEqual(result, [PROMPT, MASK, ANSWER, ANSWER, ANSWER, ANSWER, ANSWER])
        self.assertTrue(all(row[0, -2:].tolist() == [MASK, MASK] for row in self.model.inputs[:3]))
        self.assertEqual(self.model.inputs[3][0, :-2].tolist(), result[:-2])

    def test_prefilled_first_block_skips_to_second_block(self):
        self.tokenizer = lambda *args, **kwargs: SimpleNamespace(input_ids=[ALTERNATIVE, ALTERNATIVE])
        result = self.decode(
            [PROMPT], attack_method="pad", pad_anchors=["anchor"], pad_positions=[0],
        )
        self.assertEqual(result, [PROMPT, ALTERNATIVE, ALTERNATIVE, ANSWER, ANSWER])
        self.assertEqual([row[0].tolist() for row in self.model.inputs], [
            [PROMPT, ALTERNATIVE, ALTERNATIVE, MASK, MASK],
            [PROMPT, ALTERNATIVE, ALTERNATIVE, MASK, ANSWER],
        ])

    def test_explicit_whole_sequence_mode_skips_completed_canvas(self):
        result = self.decode([PROMPT, MASK], fill_all_masks=True, steps=2)
        self.assertEqual(result, [PROMPT, ANSWER, ANSWER, ANSWER, ANSWER, ANSWER])
        self.assertEqual(len(self.model.inputs), 1)

    def test_inline_only_and_empty_prompt_modes(self):
        self.assertEqual(self.decode([PROMPT, MASK, PROMPT], gen_length=0), [PROMPT, ANSWER, PROMPT])
        self.assertEqual(len(self.model.inputs), 1)
        self.assertEqual(self.decode([PROMPT], gen_length=0, sp_mode="logit", sp_threshold=0), [PROMPT])
        self.assertEqual(len(self.model.inputs), 1)

    def test_injection_still_runs_after_the_last_transfer(self):
        self.tokenizer = Mock(return_value=SimpleNamespace(input_ids=[ALTERNATIVE]))
        self.assertEqual(
            self.decode([PROMPT], injection_step=3, accept_cues=(), refuse_cues=()),
            [PROMPT, ALTERNATIVE, ANSWER, ANSWER, ANSWER],
        )
        self.assertEqual(len(self.model.inputs), 4)
        self.assertEqual(self.tokenizer.call_count, 2)
