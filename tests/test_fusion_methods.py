"""Unit tests for native and transport-augmented token fusion modules."""

import unittest

import torch

from model.fusion_methods import (
    AlignedCrossAttentionConfig,
    BANConfig,
    CrossAttentionFusionConfig,
    FusionInput,
    MUTANConfig,
    QFormerConfig,
    build_fusion_module,
)
from model.optimal_transport import TransportOutput


class FusionMethodTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.visual = torch.randn(2, 4, 6)
        self.question = torch.randn(2, 3, 7)
        self.visual_mask = torch.tensor([[False, False, False, True],
                                         [False, False, True, True]])
        self.question_mask = torch.tensor([[False, False, True],
                                           [False, False, False]])
        plan = torch.rand(2, 4, 3)
        plan = plan.masked_fill(self.visual_mask.unsqueeze(-1), 0)
        plan = plan.masked_fill(self.question_mask.unsqueeze(1), 0)
        self.transport = TransportOutput(
            plan=plan,
            cost=None,
            visual_marginal=None,
            question_marginal=None,
            fused_tokens=torch.randn(2, 3, 8).masked_fill(
                self.question_mask.unsqueeze(-1), 0
            ),
            memory_padding_mask=self.question_mask,
            transport_cost=torch.ones(2),
            entropy=torch.ones(2),
            matched_mass=torch.ones(2),
            unmatched_mass=torch.zeros(2),
            excess_mass=torch.zeros(2),
            residual=torch.zeros(2),
            iterations=torch.ones(2, dtype=torch.long),
            converged=torch.ones(2, dtype=torch.bool),
        )

    def _inputs(self, transport=None):
        return FusionInput(
            self.visual, self.question, self.visual_mask,
            self.question_mask, transport,
        )

    def test_all_methods_obey_output_contract_and_backpropagate(self):
        cases = {
            "ban": BANConfig(glimpses=2, hidden_dim=5, dropout=0),
            "mutan": MUTANConfig(rank=3, factor_dim=5, dropout=0),
            "cross_attention": CrossAttentionFusionConfig(
                layers=2, heads=2, ffn_hidden=16, dropout=0
            ),
            "aligned_cross_attention": AlignedCrossAttentionConfig(
                layers=2, heads=2, ffn_hidden=16, dropout=0, gate_init=-2
            ),
            "qformer": QFormerConfig(
                query_tokens=2, layers=2, heads=2, ffn_hidden=16, dropout=0
            ),
        }
        for method, config in cases.items():
            for transport in (None, self.transport):
                with self.subTest(method=method, uses_ot=transport is not None):
                    module = build_fusion_module(method, 6, 7, 8, config)
                    output = module(self._inputs(transport), return_diagnostics=True)
                    expected_length = 2 if method == "qformer" else 3
                    self.assertEqual(output.memory.shape, (2, expected_length, 8))
                    self.assertEqual(output.memory_padding_mask.shape, (2, expected_length))
                    self.assertEqual(output.memory_padding_mask.dtype, torch.bool)
                    self.assertTrue(torch.isfinite(output.memory).all())
                    self.assertIsNotNone(output.diagnostics)
                    output.memory.square().mean().backward()
                    gradients = [
                        value.grad for value in module.parameters()
                        if value.requires_grad and value.grad is not None
                    ]
                    self.assertTrue(gradients)
                    self.assertTrue(all(torch.isfinite(value).all() for value in gradients))

    def test_padded_inputs_do_not_change_valid_outputs(self):
        configs = {
            "ban": BANConfig(glimpses=2, hidden_dim=5, dropout=0),
            "mutan": MUTANConfig(rank=3, factor_dim=5, dropout=0),
            "cross_attention": CrossAttentionFusionConfig(
                layers=1, heads=2, ffn_hidden=16, dropout=0
            ),
            "aligned_cross_attention": AlignedCrossAttentionConfig(
                layers=1, heads=2, ffn_hidden=16, dropout=0, gate_init=-2
            ),
            "qformer": QFormerConfig(
                query_tokens=3, layers=1, heads=2, ffn_hidden=16, dropout=0
            ),
        }
        changed_visual = self.visual.clone()
        changed_question = self.question.clone()
        changed_visual[self.visual_mask] = 1000
        changed_question[self.question_mask] = -1000
        changed = FusionInput(
            changed_visual, changed_question, self.visual_mask,
            self.question_mask, None,
        )
        for method, config in configs.items():
            with self.subTest(method=method):
                module = build_fusion_module(method, 6, 7, 8, config).eval()
                first = module(self._inputs())
                second = module(changed)
                valid = ~first.memory_padding_mask
                torch.testing.assert_close(first.memory[valid], second.memory[valid])

    def test_diagnostics_do_not_change_eval_memory(self):
        config = CrossAttentionFusionConfig(
            layers=1, heads=2, ffn_hidden=16, dropout=0.3
        )
        module = build_fusion_module("cross_attention", 6, 7, 8, config).eval()
        without = module(self._inputs(self.transport), False)
        with_diagnostics = module(self._inputs(self.transport), True)
        torch.testing.assert_close(without.memory, with_diagnostics.memory)

    def test_every_module_strictly_reloads_its_state(self):
        configs = {
            "ban": BANConfig(glimpses=2, hidden_dim=5, dropout=0),
            "mutan": MUTANConfig(rank=3, factor_dim=5, dropout=0),
            "cross_attention": CrossAttentionFusionConfig(
                layers=1, heads=2, ffn_hidden=16, dropout=0
            ),
            "aligned_cross_attention": AlignedCrossAttentionConfig(
                layers=1, heads=2, ffn_hidden=16, dropout=0, gate_init=-2
            ),
            "qformer": QFormerConfig(
                query_tokens=3, layers=1, heads=2, ffn_hidden=16, dropout=0
            ),
        }
        for method, config in configs.items():
            with self.subTest(method=method):
                original = build_fusion_module(method, 6, 7, 8, config)
                restored = build_fusion_module(method, 6, 7, 8, config)
                restored.load_state_dict(original.state_dict(), strict=True)
                first = original.eval()(self._inputs(self.transport)).memory
                second = restored.eval()(self._inputs(self.transport)).memory
                torch.testing.assert_close(first, second)

    def test_aligned_cross_attention_gate_controls_ot_interpolation(self):
        config = AlignedCrossAttentionConfig(
            layers=1, heads=2, ffn_hidden=16, dropout=0, gate_init=-30
        )
        module = build_fusion_module(
            "aligned_cross_attention", 6, 7, 8, config, uses_ot=True
        ).eval()
        without_ot = module(self._inputs(), return_diagnostics=True)
        near_zero_gate = module(self._inputs(self.transport), return_diagnostics=True)
        torch.testing.assert_close(
            without_ot.memory, near_zero_gate.memory, atol=1e-5, rtol=1e-5
        )
        self.assertLess(
            near_zero_gate.diagnostics["ot_gate_mean"].max().item(), 1e-10
        )

        with torch.no_grad():
            module.ot_gate.bias.fill_(30)
        first = module(self._inputs(self.transport), return_diagnostics=True)
        changed_transport = TransportOutput(
            **{
                **self.transport.__dict__,
                "fused_tokens": self.transport.fused_tokens + 2.0,
            }
        )
        second = module(self._inputs(changed_transport), return_diagnostics=True)
        self.assertFalse(torch.allclose(first.memory, second.memory))
        self.assertGreater(first.diagnostics["ot_gate_mean"].min().item(), 0.999)

    def test_aligned_cross_attention_masks_gate_diagnostics(self):
        config = AlignedCrossAttentionConfig(
            layers=1, heads=2, ffn_hidden=16, dropout=0, gate_init=-2
        )
        module = build_fusion_module(
            "aligned_cross_attention", 6, 7, 8, config, uses_ot=True
        ).eval()
        changed = TransportOutput(
            **{
                **self.transport.__dict__,
                "fused_tokens": self.transport.fused_tokens.clone(),
            }
        )
        changed.fused_tokens[self.question_mask] = 1000
        first = module(self._inputs(self.transport), return_diagnostics=True)
        second = module(self._inputs(changed), return_diagnostics=True)
        valid = ~self.question_mask
        torch.testing.assert_close(first.memory[valid], second.memory[valid])
        torch.testing.assert_close(
            first.diagnostics["ot_gate_mean"],
            second.diagnostics["ot_gate_mean"],
        )


if __name__ == "__main__":
    unittest.main()
