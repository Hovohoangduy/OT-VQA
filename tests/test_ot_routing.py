"""Numerical and integration tests for OT evidence routing."""

import unittest

import torch

from model.ot_routing import (
    OTEvidenceRouter, OTEvidenceRoutingConfig,
    _masked_sparsemax,
    independent_softmax_transport, semi_relaxed_sinkhorn,
)


class SemiRelaxedTransportTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.cost = torch.randn(2, 3, 5)
        self.a = torch.full((2, 3), 1 / 3)
        self.b = torch.tensor([
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.3, 0.2, 0.2, 0.2, 0.1],
        ])
        self.mask = torch.tensor([
            [False, False, False, True, False],
            [False, False, False, False, False],
        ])
        self.b[0, 3] = 0
        self.b[0] /= self.b[0].sum()

    def test_plan_is_finite_masked_and_respects_hard_rows(self):
        result = semi_relaxed_sinkhorn(
            self.cost, self.a, self.b, self.mask,
            epsilon=0.2, tau=0.5, iterations=50,
        )
        self.assertTrue(torch.isfinite(result.plan).all())
        self.assertTrue((result.plan >= 0).all())
        self.assertTrue((result.plan[0, :, 3] == 0).all())
        torch.testing.assert_close(
            result.plan.sum(-1), self.a, atol=2e-6, rtol=2e-6,
        )

    def test_tau_zero_is_exact_independent_softmax_limit(self):
        ot = semi_relaxed_sinkhorn(
            self.cost, self.a, self.b, self.mask,
            epsilon=0.2, tau=0, iterations=20,
        )
        control = independent_softmax_transport(
            self.cost, self.a, self.mask, epsilon=0.2,
        )
        torch.testing.assert_close(ot.plan, control.plan)

    def test_single_slot_matches_closed_form_reference(self):
        cost = torch.tensor([[[0.2, 0.8, -0.1]]], dtype=torch.float64)
        a = torch.ones(1, 1, dtype=torch.float64)
        b = torch.tensor([[0.2, 0.5, 0.3]], dtype=torch.float64)
        mask = torch.zeros(1, 3, dtype=torch.bool)
        epsilon, tau = 0.3, 0.7
        result = semi_relaxed_sinkhorn(
            cost, a, b, mask, epsilon=epsilon, tau=tau,
            iterations=200, tolerance=1e-7,
        )
        logits = (
            tau * b.log() - cost[:, 0]
        ) / (epsilon + tau)
        expected = torch.softmax(logits, dim=-1).unsqueeze(1).float()
        torch.testing.assert_close(result.plan, expected, atol=2e-6, rtol=2e-6)

    def test_gradients_reach_cost_and_preference(self):
        cost = self.cost.clone().requires_grad_(True)
        preference_logits = torch.randn(2, 5, requires_grad=True)
        preference = torch.softmax(
            preference_logits.masked_fill(self.mask, float("-inf")), dim=-1
        )
        result = semi_relaxed_sinkhorn(
            cost, self.a, preference, self.mask,
            epsilon=0.2, tau=0.5, iterations=10,
        )
        (result.plan.square().sum()).backward()
        self.assertTrue(torch.isfinite(cost.grad).all())
        self.assertTrue(torch.isfinite(preference_logits.grad).all())
        self.assertGreater(cost.grad.abs().sum().item(), 0)
        self.assertGreater(preference_logits.grad.abs().sum().item(), 0)


class EvidenceRouterTests(unittest.TestCase):
    def config(self, **changes):
        values = dict(
            slots=3, reasoning_steps=2, routing_dim=12, heads=3,
            dropout=0, epsilon=0.2, tau=0.5, sinkhorn_iterations=10,
        )
        values.update(changes)
        return OTEvidenceRoutingConfig(**values)

    def inputs(self):
        torch.manual_seed(3)
        visual = torch.randn(2, 4, 8)
        question = torch.randn(2, 5, 10)
        visual_mask = torch.tensor([
            [False, False, False, True],
            [False, False, False, False],
        ])
        question_mask = torch.tensor([
            [False, False, False, True, True],
            [False, False, False, False, True],
        ])
        return visual, question, visual_mask, question_mask

    def test_shapes_diagnostics_and_gradients(self):
        module = OTEvidenceRouter(8, 10, 16, self.config(), routing_mode="ot")
        output = module(*self.inputs(), grid_size=(2, 2), return_diagnostics=True)
        self.assertEqual(output.memory.shape, (2, 3, 16))
        self.assertEqual(output.memory_padding_mask.shape, (2, 3))
        self.assertEqual(output.attention_weights.shape, (2, 3, 6))
        self.assertIn("routing_null", output.diagnostics)
        self.assertIn("routing_query_similarity", output.diagnostics)
        self.assertTrue(torch.isfinite(output.auxiliary_loss))
        output.memory.square().mean().backward()
        self.assertIsNotNone(module.cost_query.weight.grad)
        self.assertIsNotNone(module.preference_score.weight.grad)
        self.assertGreater(module.cost_query.weight.grad.abs().sum().item(), 0)

    def test_slot_initialization_preserves_distinct_roles(self):
        module = OTEvidenceRouter(8, 10, 16, self.config(), routing_mode="ot")
        torch.manual_seed(9)
        common_question = 20 * torch.randn(2, self.config().routing_dim)
        slots = module._initialize_slots(common_question)
        normalized = torch.nn.functional.normalize(slots, dim=-1)
        similarities = torch.matmul(normalized, normalized.transpose(1, 2))
        off_diagonal = ~torch.eye(3, dtype=torch.bool).unsqueeze(0)
        self.assertLess(similarities.masked_select(off_diagonal).mean().item(), 0.9)

    def test_query_diversity_loss_trains_routing_queries(self):
        module = OTEvidenceRouter(8, 10, 16, self.config(), routing_mode="ot")
        output = module(*self.inputs(), grid_size=(2, 2))
        output.auxiliary_loss.backward()
        self.assertGreater(module.cost_query.weight.grad.abs().sum().item(), 0)
        self.assertGreater(module.slot_embeddings.grad.abs().sum().item(), 0)

    def test_padded_values_cannot_change_output(self):
        module = OTEvidenceRouter(8, 10, 16, self.config(), routing_mode="ot").eval()
        visual, question, visual_mask, question_mask = self.inputs()
        first = module(
            visual, question, visual_mask, question_mask, grid_size=(2, 2)
        ).memory
        visual[visual_mask] = 1e5
        question[question_mask] = -1e5
        second = module(
            visual, question, visual_mask, question_mask, grid_size=(2, 2)
        ).memory
        torch.testing.assert_close(first, second, atol=1e-5, rtol=1e-5)

    def test_softmax_and_tau_zero_router_agree(self):
        config = self.config(tau=0)
        ot = OTEvidenceRouter(8, 10, 16, config, routing_mode="ot").eval()
        control = OTEvidenceRouter(8, 10, 16, config, routing_mode="softmax").eval()
        control.load_state_dict(ot.state_dict())
        inputs = self.inputs()
        first = ot(*inputs, grid_size=(2, 2), return_diagnostics=True)
        second = control(*inputs, grid_size=(2, 2), return_diagnostics=True)
        torch.testing.assert_close(first.memory, second.memory)
        torch.testing.assert_close(first.attention_weights, second.attention_weights)

    def test_diagnostics_do_not_change_eval_output(self):
        module = OTEvidenceRouter(8, 10, 16, self.config(), routing_mode="ot").eval()
        inputs = self.inputs()
        plain = module(*inputs, grid_size=(2, 2), return_diagnostics=False)
        diagnosed = module(*inputs, grid_size=(2, 2), return_diagnostics=True)
        torch.testing.assert_close(plain.memory, diagnosed.memory)

    def test_grid_mismatch_is_rejected(self):
        module = OTEvidenceRouter(8, 10, 16, self.config(), routing_mode="ot")
        with self.assertRaisesRegex(ValueError, "does not match"):
            module(*self.inputs(), grid_size=(1, 3))

    def test_null_only_evidence_is_finite(self):
        module = OTEvidenceRouter(8, 10, 16, self.config(), routing_mode="ot").eval()
        visual, question, visual_mask, question_mask = self.inputs()
        visual_mask[:] = True
        output = module(
            visual, question, visual_mask, question_mask,
            grid_size=(2, 2), return_diagnostics=True,
        )
        self.assertTrue(torch.isfinite(output.memory).all())
        torch.testing.assert_close(
            output.attention_weights[:, :, -1],
            torch.full((2, 3), 1 / 3),
        )

    def test_outer_autocast_keeps_transport_gradients_finite(self):
        module = OTEvidenceRouter(8, 10, 16, self.config(), routing_mode="ot")
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            output = module(
                *self.inputs(), grid_size=(2, 2), return_diagnostics=True,
            )
            loss = output.memory.square().mean()
        loss.backward()
        self.assertEqual(output.memory.dtype, torch.bfloat16)
        self.assertEqual(output.attention_weights.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(module.cost_query.weight.grad).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_forward_backward_is_finite(self):
        module = OTEvidenceRouter(8, 10, 16, self.config(), routing_mode="ot").cuda()
        inputs = tuple(value.cuda() for value in self.inputs())
        output = module(*inputs, grid_size=(2, 2))
        output.memory.square().mean().backward()
        self.assertTrue(torch.isfinite(output.memory).all())
        self.assertTrue(torch.isfinite(module.cost_query.weight.grad).all())

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable")
    def test_mps_forward_backward_is_finite(self):
        device = torch.device("mps")
        module = OTEvidenceRouter(8, 10, 16, self.config(), routing_mode="ot").to(device)
        inputs = tuple(value.to(device) for value in self.inputs())
        output = module(*inputs, grid_size=(2, 2))
        output.memory.square().mean().backward()
        self.assertTrue(torch.isfinite(output.memory).all().cpu())
        self.assertTrue(torch.isfinite(module.cost_query.weight.grad).all().cpu())

    def test_spatial_permutation_without_positions_changes_output(self):
        module = OTEvidenceRouter(8, 10, 16, self.config(), routing_mode="ot").eval()
        visual, question, visual_mask, question_mask = self.inputs()
        first = module(
            visual, question, visual_mask, question_mask, grid_size=(2, 2)
        ).memory
        permutation = torch.tensor([2, 0, 3, 1])
        # Position is part of evidence, so this test uses direct spatial-token
        # permutation only to assert that changing positions changes semantics.
        # Exact permutation invariance belongs to the lower-level solver.
        changed = module(
            visual[:, permutation], question, visual_mask[:, permutation],
            question_mask, grid_size=(2, 2),
        ).memory
        self.assertFalse(torch.allclose(first, changed))

    def test_sparsemax_is_normalized_masked_and_sparse(self):
        logits = torch.tensor([
            [3.0, 1.0, 0.0, -2.0],
            [0.1, 0.2, 0.3, 9.0],
        ])
        mask = torch.tensor([
            [False, False, False, False],
            [False, False, False, True],
        ])
        output = _masked_sparsemax(logits, mask)
        torch.testing.assert_close(output.sum(-1), torch.ones(2))
        self.assertEqual(output[1, 3].item(), 0.0)
        self.assertGreater((output == 0).sum().item(), 2)

    def test_v2_routed_patch_memory_preserves_tokens_and_gradients(self):
        config = self.config(
            memory_mode="routed_patches",
            preference_transform="sparsemax",
            preference_smoothing=0.001,
            cost_scale_mode="learned",
            cost_scale=4.0,
            question_conditioned_keys=True,
        )
        module = OTEvidenceRouter(8, 10, 16, config, routing_mode="ot")
        visual, question, visual_mask, question_mask = self.inputs()
        output = module(
            visual, question, visual_mask, question_mask,
            grid_size=(2, 2), return_diagnostics=True,
        )
        expected_length = question.size(1) + config.slots + visual.size(1)
        self.assertEqual(output.memory.shape, (2, expected_length, 16))
        torch.testing.assert_close(
            output.memory_padding_mask[:, :question.size(1)], question_mask
        )
        torch.testing.assert_close(
            output.memory_padding_mask[:, -visual.size(1):], visual_mask
        )
        self.assertIn("routing_gate_mean", output.diagnostics)
        self.assertIn("routing_preference_effective_support", output.diagnostics)
        output.memory.square().mean().backward()
        self.assertGreater(module.cost_query.weight.grad.abs().sum().item(), 0)
        self.assertGreater(module.raw_cost_scale.grad.abs().item(), 0)

    def test_v2_sparse_preference_is_safe_under_outer_autocast(self):
        config = self.config(
            memory_mode="routed_patches",
            preference_transform="sparsemax",
            preference_smoothing=0.001,
            cost_scale_mode="learned",
            cost_scale=4.0,
            question_conditioned_keys=True,
        )
        module = OTEvidenceRouter(8, 10, 16, config, routing_mode="ot")
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            output = module(
                *self.inputs(), grid_size=(2, 2), return_diagnostics=True,
            )
            loss = output.memory.square().mean()
        loss.backward()
        self.assertEqual(output.memory.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(output.memory).all())
        self.assertTrue(torch.isfinite(module.preference_score.weight.grad).all())
        self.assertGreater(module.preference_score.weight.grad.abs().sum().item(), 0)

    def test_runtime_tau_can_warm_from_independent_routing(self):
        module = OTEvidenceRouter(8, 10, 16, self.config(), routing_mode="ot")
        module.set_tau(0.0)
        output = module(
            *self.inputs(), grid_size=(2, 2), return_diagnostics=True,
        )
        torch.testing.assert_close(
            output.diagnostics["routing_runtime_tau"], torch.zeros(2)
        )
        output.memory.square().mean().backward()
        self.assertIsNotNone(module.preference_score.weight.grad)

    def test_plan_intervention_changes_routed_result_and_is_reversible(self):
        config = self.config(memory_mode="routed_patches")
        module = OTEvidenceRouter(8, 10, 16, config, routing_mode="ot").eval()
        inputs = self.inputs()
        original = module(*inputs, grid_size=(2, 2)).memory
        module.set_plan_intervention("shuffle_evidence")
        changed = module(*inputs, grid_size=(2, 2)).memory
        module.set_plan_intervention(None)
        restored = module(*inputs, grid_size=(2, 2)).memory
        self.assertFalse(torch.allclose(original, changed))
        torch.testing.assert_close(original, restored)


if __name__ == "__main__":
    unittest.main()
