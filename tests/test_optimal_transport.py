"""Numerical tests for the training-only Sinkhorn kernel."""

import unittest

import torch

from model.optimal_transport import OTConfig, sinkhorn_transport, uniform_marginal


class SinkhornTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.cost = torch.rand(2, 4, 3)
        self.visual_mask = torch.tensor([
            [False, False, False, True],
            [False, False, False, False],
        ])
        self.question_mask = torch.tensor([
            [False, False, True],
            [False, False, False],
        ])
        self.a = uniform_marginal(self.visual_mask, torch.float32)
        self.b = uniform_marginal(self.question_mask, torch.float32)

    def solve(self):
        return sinkhorn_transport(
            self.cost, self.a, self.b, self.visual_mask, self.question_mask,
            OTConfig(
                epsilon=0.1,
                max_iterations=100,
                tolerance=1e-4,
            ),
        )

    def test_unbalanced_plan_respects_padding(self):
        result = self.solve()
        self.assertTrue((result.plan[0, -1] == 0).all())
        self.assertTrue((result.plan[0, :, -1] == 0).all())

    def test_unbalanced_plan_is_finite_and_can_change_mass(self):
        result = self.solve()
        self.assertTrue(torch.isfinite(result.plan).all())
        self.assertTrue((result.plan >= 0).all())
        self.assertFalse(torch.allclose(result.plan.sum((1, 2)), torch.ones(2)))

    def test_solver_backpropagates_to_cost(self):
        cost = self.cost.clone().requires_grad_(True)
        result = sinkhorn_transport(
            cost, self.a, self.b, self.visual_mask, self.question_mask,
            OTConfig(max_iterations=10),
        )
        (result.plan * cost).sum().backward()
        self.assertIsNotNone(cost.grad)
        self.assertTrue(torch.isfinite(cost.grad).all())


if __name__ == "__main__":
    unittest.main()
