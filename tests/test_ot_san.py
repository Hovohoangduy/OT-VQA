"""Unit tests for masked OT-SAN aggregation."""

import unittest

import torch

from model.ot_san import OTSAN, OTSANConfig


class OTSANTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.config = OTSANConfig(
            hidden_dim=7, num_layers=2, dropout=0.0, gate_init=-2.0
        )
        self.module = OTSAN(model_dim=6, config=self.config)
        self.tokens = torch.randn(2, 5, 6, requires_grad=True)
        self.mask = torch.tensor([
            [False, False, False, True, True],
            [False, False, False, False, False],
        ])

    def test_shapes_masks_weights_and_gate(self):
        output = self.module(self.tokens, self.mask, return_diagnostics=True)
        self.assertEqual(output.memory.shape, (2, 6, 6))
        self.assertEqual(output.memory_padding_mask.shape, (2, 6))
        self.assertEqual(output.summary.shape, (2, 6))
        self.assertEqual(output.attention_weights.shape, (2, 2, 5))
        self.assertEqual(output.attention_entropy.shape, (2, 2))
        self.assertEqual(output.summary_norm.shape, (2,))
        self.assertFalse(output.memory_padding_mask[:, 0].any())
        torch.testing.assert_close(output.memory_padding_mask[:, 1:], self.mask)
        self.assertTrue((output.attention_weights[0, :, 3:] == 0).all())
        torch.testing.assert_close(
            output.attention_weights.sum(-1), torch.ones(2, 2)
        )
        torch.testing.assert_close(
            output.gate, torch.sigmoid(torch.tensor(-2.0))
        )

    def test_padding_values_cannot_change_summary(self):
        first = self.module(self.tokens, self.mask).summary
        changed = self.tokens.detach().clone()
        changed[0, 3:] = torch.randn_like(changed[0, 3:]) * 1000
        second = self.module(changed, self.mask).summary
        torch.testing.assert_close(first, second)

    def test_gradients_are_finite(self):
        output = self.module(self.tokens, self.mask, return_diagnostics=True)
        output.memory.square().mean().backward()
        self.assertTrue(torch.isfinite(self.tokens.grad).all())
        self.assertIsNotNone(self.module.gate_logit.grad)
        for parameter in self.module.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_diagnostics_do_not_change_eval_output(self):
        self.module.eval()
        without = self.module(self.tokens, self.mask, return_diagnostics=False)
        with_diagnostics = self.module(self.tokens, self.mask, return_diagnostics=True)
        torch.testing.assert_close(without.memory, with_diagnostics.memory)
        self.assertIsNone(without.attention_weights)
        self.assertIsNotNone(with_diagnostics.attention_weights)

    def test_invalid_inputs_raise(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            self.module(self.tokens[:, 0], self.mask)
        with self.assertRaisesRegex(ValueError, "match"):
            self.module(self.tokens, self.mask[:, :-1])
        with self.assertRaisesRegex(ValueError, "boolean"):
            self.module(self.tokens, self.mask.float())
        with self.assertRaisesRegex(ValueError, "valid token"):
            self.module(self.tokens, torch.ones_like(self.mask))
        with self.assertRaises(ValueError):
            OTSANConfig(hidden_dim=0)
        with self.assertRaises(ValueError):
            OTSANConfig(num_layers=3)
        with self.assertRaises(ValueError):
            OTSANConfig(dropout=1.0)


if __name__ == "__main__":
    unittest.main()
