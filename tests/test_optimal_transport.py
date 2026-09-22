"""Numerical and integration checks for Optimal-Transport fusion."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from model.optimal_transport import (
    OTConfig, OptimalTransportFusion, TransportOutput, masked_softmax, sinkhorn_transport,
    uniform_marginal,
)
from utils.feature_cache import FeatureCacheDataset, collate_feature_cache, file_fingerprint, write_feature_cache
from utils.transport_visualization import save_transport_diagnostics


class SinkhornTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_masked_marginals_are_normalized_and_zero_on_padding(self):
        mask = torch.tensor([[False, False, True], [False, True, True]])
        uniform = uniform_marginal(mask, torch.float32)
        learned = masked_softmax(torch.tensor([[1.0, 2.0, 99.0], [3.0, 99.0, 99.0]]), mask)
        torch.testing.assert_close(uniform.sum(1), torch.ones(2))
        torch.testing.assert_close(learned.sum(1), torch.ones(2))
        self.assertTrue((uniform[mask] == 0).all())
        self.assertTrue((learned[mask] == 0).all())

    def test_balanced_plan_matches_marginals_and_diagonal_cost(self):
        cost = torch.tensor([[[0.0, 4.0], [4.0, 0.0]]])
        marginal = torch.tensor([[0.4, 0.6]])
        mask = torch.zeros(1, 2, dtype=torch.bool)
        output = sinkhorn_transport(
            cost, marginal, marginal, mask, mask,
            OTConfig(transport_type="balanced", epsilon=0.1,
                     max_iterations=200, tolerance=1e-6),
        )
        torch.testing.assert_close(output.plan.sum(2), marginal, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(output.plan.sum(1), marginal, atol=1e-5, rtol=1e-5)
        self.assertGreater(output.plan.diagonal(dim1=1, dim2=2).sum().item(), 0.99)
        self.assertTrue(output.converged.item())

    def test_uot_can_leave_mass_unmatched_and_padding_gets_no_mass(self):
        cost = torch.full((1, 3, 3), 2.0)
        mask = torch.tensor([[False, False, True]])
        marginal = uniform_marginal(mask, torch.float32)
        output = sinkhorn_transport(
            cost, marginal, marginal, mask, mask,
            OTConfig(transport_type="unbalanced", epsilon=0.1,
                     tau_visual=0.2, tau_question=0.2,
                     max_iterations=300, tolerance=1e-6),
        )
        self.assertLess(output.plan.sum().item(), 1.0)
        self.assertTrue((output.plan[:, 2] == 0).all())
        self.assertTrue((output.plan[:, :, 2] == 0).all())
        self.assertTrue(torch.isfinite(output.plan).all())

    def test_lower_epsilon_concentrates_plan_and_masked_values_do_not_matter(self):
        cost = torch.tensor([[[0.0, 1.0, 999.0], [1.0, 0.0, 999.0]]])
        vm = torch.zeros(1, 2, dtype=torch.bool)
        qm = torch.tensor([[False, False, True]])
        a = uniform_marginal(vm, torch.float32)
        b = uniform_marginal(qm, torch.float32)
        plans = []
        for epsilon in (0.5, 0.05):
            config = OTConfig(transport_type="balanced", epsilon=epsilon,
                              max_iterations=200, tolerance=1e-6)
            plans.append(sinkhorn_transport(cost, a, b, vm, qm, config).plan)
        self.assertGreater(plans[1].square().sum().item(), plans[0].square().sum().item())
        changed = cost.clone()
        changed[:, :, 2] = -999.0
        reference = sinkhorn_transport(
            cost, a, b, vm, qm,
            OTConfig(transport_type="balanced", max_iterations=200),
        ).plan
        actual = sinkhorn_transport(
            changed, a, b, vm, qm,
            OTConfig(transport_type="balanced", max_iterations=200),
        ).plan
        torch.testing.assert_close(reference, actual)

    def test_fusion_gradients_are_finite_and_decoder_memory_is_masked(self):
        torch.manual_seed(3)
        config = OTConfig(transport_type="unbalanced", ot_dim=8,
                          max_iterations=40, tolerance=1e-5)
        fusion = OptimalTransportFusion(6, 7, 10, config)
        visual = torch.randn(2, 4, 6, requires_grad=True)
        question = torch.randn(2, 5, 7, requires_grad=True)
        visual_mask = torch.tensor([[False] * 4, [False, False, False, True]])
        question_mask = torch.tensor([[False, False, False, True, True], [False] * 5])
        output = fusion(visual, question, visual_mask, question_mask, True)
        loss = output.fused_tokens.square().mean() + output.transport_cost.mean()
        loss.backward()
        self.assertTrue(torch.isfinite(output.plan).all())
        self.assertTrue((output.fused_tokens[question_mask] == 0).all())
        for parameter in (fusion.pairwise_cost.learned[0].weight,
                          fusion.visual_marginal.scorer[0].weight,
                          fusion.question_marginal.scorer[0].weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_fast_alignment_profile_is_finite_at_ten_and_twenty_iterations(self):
        profile = OTConfig.from_json("configs/ot_aligned_fast.json")
        self.assertEqual(profile.ot_dim, 128)
        self.assertEqual(profile.max_iterations, 20)
        cost = torch.rand(2, 5, 4)
        visual_mask = torch.zeros(2, 5, dtype=torch.bool)
        question_mask = torch.tensor([
            [False, False, False, True],
            [False, False, False, False],
        ])
        a = uniform_marginal(visual_mask, torch.float32)
        b = uniform_marginal(question_mask, torch.float32)
        for iterations in (10, 20):
            output = sinkhorn_transport(
                cost, a, b, visual_mask, question_mask,
                replace(profile, max_iterations=iterations),
            )
            self.assertTrue(torch.isfinite(output.plan).all())
            self.assertTrue(torch.isfinite(output.residual).all())
            self.assertTrue((output.iterations <= iterations).all())

    def test_invalid_config_and_empty_token_set_raise(self):
        with self.assertRaises(ValueError):
            OTConfig(epsilon=0)
        with self.assertRaisesRegex(ValueError, "valid token"):
            uniform_marginal(torch.ones(1, 2, dtype=torch.bool), torch.float32)


class FeatureCacheTests(unittest.TestCase):
    def test_cache_round_trip_collation_and_provenance_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv = root / "data.csv"
            csv.write_text("image,question,answer\na.jpg,what?,red\n", encoding="utf-8")
            samples = [
                {"anno_id": "a", "image_features": torch.ones(6, 4).half(),
                 "question_features": torch.ones(2, 5).half(),
                 "question_padding_mask": torch.tensor([True, False]),
                 "question": "what?", "answer": "red"},
                {"anno_id": "b", "image_features": torch.zeros(6, 4).half(),
                 "question_features": torch.zeros(3, 5).half(),
                 "question_padding_mask": torch.tensor([True, False, True]),
                 "question": "color?", "answer": "blue"},
            ]
            write_feature_cache(root / "cache", samples, {
                "dataset_fingerprint": file_fingerprint(csv),
                "text_model": "tiny-text", "image_model": "tiny-image",
            })
            dataset = FeatureCacheDataset(root / "cache", csv, "tiny-text", "tiny-image")
            batch = collate_feature_cache([dataset[0], dataset[1]])
            self.assertEqual(batch["question_features"].shape, (2, 3, 5))
            self.assertTrue(batch["question_padding_mask"][0, 2])
            self.assertEqual(batch["image_features"].dtype, torch.float16)
            csv.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "CSV contents"):
                FeatureCacheDataset(root / "cache", csv)

    def test_transport_diagnostic_figure_is_written(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = torch.tensor([[[0.4, 0.1], [0.1, 0.4]]])
            transport = TransportOutput(
                plan=plan, cost=1 - plan,
                visual_marginal=torch.tensor([[0.5, 0.5]]),
                question_marginal=torch.tensor([[0.5, 0.5]]),
                fused_tokens=torch.zeros(1, 2, 4),
                memory_padding_mask=torch.zeros(1, 2, dtype=torch.bool),
                transport_cost=torch.tensor([0.6]), entropy=torch.tensor([1.2]),
                matched_mass=torch.tensor([1.0]), unmatched_mass=torch.tensor([0.0]),
                excess_mass=torch.tensor([0.0]), residual=torch.tensor([1e-6]),
                iterations=torch.tensor([4]), converged=torch.tensor([True]),
            )
            output = Path(directory) / "diagnostics.png"
            save_transport_diagnostics(transport, output, question_tokens=["what", "color"])
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
