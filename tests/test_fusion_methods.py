"""Focused tests for the native Cross-Attention baseline."""

import unittest

import torch

from model.fusion_methods import (
    CrossAttentionFusion, CrossAttentionFusionConfig, FusionInput,
)


class CrossAttentionFusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.visual = torch.randn(2, 4, 6)
        self.question = torch.randn(2, 5, 7)
        self.visual_mask = torch.tensor([
            [False, False, False, True],
            [False, False, False, False],
        ])
        self.question_mask = torch.tensor([
            [False, False, False, True, True],
            [False, False, False, False, True],
        ])
        self.config = CrossAttentionFusionConfig(
            layers=2, heads=2, ffn_hidden=16, dropout=0,
        )

    def test_shapes_masks_diagnostics_and_gradients(self):
        module = CrossAttentionFusion(6, 7, 8, self.config)
        output = module(
            FusionInput(
                self.visual, self.question, self.visual_mask, self.question_mask,
            ),
            return_diagnostics=True,
        )
        self.assertEqual(output.memory.shape, (2, 5, 8))
        self.assertEqual(output.attention_weights.shape, (2, 2, 5, 4))
        self.assertTrue((output.memory[self.question_mask] == 0).all())
        self.assertTrue(
            (output.attention_weights[0, :, :, self.visual_mask[0]] == 0).all()
        )
        output.memory.square().mean().backward()
        self.assertIsNotNone(module.layers[-1].q.weight.grad)

    def test_padded_values_cannot_change_valid_outputs(self):
        module = CrossAttentionFusion(6, 7, 8, self.config).eval()
        first = module(FusionInput(
            self.visual, self.question, self.visual_mask, self.question_mask,
        )).memory
        changed_visual = self.visual.clone()
        changed_question = self.question.clone()
        changed_visual[self.visual_mask] = 1e4
        changed_question[self.question_mask] = -1e4
        second = module(FusionInput(
            changed_visual, changed_question, self.visual_mask, self.question_mask,
        )).memory
        torch.testing.assert_close(first[~self.question_mask], second[~self.question_mask])

    def test_configuration_round_trip(self):
        restored = CrossAttentionFusionConfig.from_dict(self.config.to_dict())
        self.assertEqual(restored, self.config)

    def test_invalid_mask_is_rejected(self):
        module = CrossAttentionFusion(6, 7, 8, self.config)
        with self.assertRaisesRegex(ValueError, "Boolean"):
            module(FusionInput(
                self.visual, self.question, self.visual_mask.float(),
                self.question_mask,
            ))


if __name__ == "__main__":
    unittest.main()
