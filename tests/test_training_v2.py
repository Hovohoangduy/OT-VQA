"""Focused tests for V2 training safety and counterfactual batching."""

import math
import unittest

import torch
from torch import nn

from train import (
    _counterfactual_forward, _gradient_health, _sequence_log_probability,
    _stable_gradient_norm, train,
)


class _FeatureEcho(nn.Module):
    def forward(
        self, image_features=None, question_features=None,
        question_padding_mask=None, answers=None,
    ):
        targets = torch.zeros(
            image_features.size(0), 1, dtype=torch.long,
            device=image_features.device,
        )
        return image_features, targets


class _TinyVQA(nn.Module):
    fusion_type = "cross_attention"
    pad_token_id = 0

    def __init__(self):
        super().__init__()
        self.fusion_module = nn.Linear(1, 3)

    def forward(self, image_features=None, answers=None, **kwargs):
        pooled = image_features.float().mean(dim=tuple(range(1, image_features.ndim)))
        logits = self.fusion_module(pooled.unsqueeze(-1)).unsqueeze(1).repeat(1, 2, 1)
        targets = torch.ones(image_features.size(0), 2, dtype=torch.long)
        return logits, targets

    @staticmethod
    def answers_from_ids(ids):
        return [str(int(row[0])) for row in ids.cpu()]


class V2TrainingTests(unittest.TestCase):
    def test_stable_norm_does_not_overflow_on_large_finite_gradients(self):
        gradient = torch.full((10,), 1e20)
        norm = _stable_gradient_norm([gradient])
        self.assertTrue(math.isfinite(norm))
        self.assertAlmostEqual(norm / 1e20, math.sqrt(10), places=5)

    def test_gradient_health_detects_nonfinite_values(self):
        parameter = nn.Parameter(torch.ones(2))
        parameter.grad = torch.tensor([1.0, float("inf")])
        health = _gradient_health([parameter])
        self.assertFalse(health["finite"])
        self.assertEqual(health["nonfinite"], 1)

    def test_counterfactual_batch_uses_different_images_and_answers(self):
        batch = {
            "anno_ids": [10, 11, 12],
            "answers": ["yes", "no", "left"],
            "image_features": torch.tensor([[[0.0]], [[1.0]], [[2.0]]]),
            "question_features": torch.zeros(3, 2, 1),
            "question_padding_mask": torch.zeros(3, 2, dtype=torch.bool),
        }
        result, eligible = _counterfactual_forward(
            _FeatureEcho(), batch, torch.device("cpu")
        )
        self.assertTrue(eligible.all())
        self.assertEqual(result[0][:, 0, 0].tolist(), [1.0, 2.0, 0.0])

    def test_sequence_log_probability_ignores_padding(self):
        logits = torch.tensor([[[0.0, 2.0], [4.0, 0.0], [9.0, -9.0]]])
        targets = torch.tensor([[1, 1, 0]])
        expected = (
            torch.log_softmax(logits, -1)[0, 0, 1]
            + torch.log_softmax(logits, -1)[0, 1, 1]
        ) / 2
        actual = _sequence_log_probability(logits, targets, pad_token_id=0)
        self.assertAlmostEqual(actual.item(), expected.item(), places=5)

    def test_train_supports_accumulation_and_counterfactual_loss(self):
        model = _TinyVQA()
        loader = []
        for offset in (0.0, 2.0):
            loader.append({
                "anno_ids": [int(offset), int(offset + 1)],
                "answers": ["yes", "no"],
                "image_features": torch.tensor([[[offset]], [[offset + 1]]]),
                "question_features": torch.zeros(2, 1, 1),
                "question_padding_mask": torch.zeros(2, 1, dtype=torch.bool),
            })
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        losses, em, f1, diagnostics = train(
            model, loader, 1, optimizer, scheduler, nn.CrossEntropyLoss(),
            device=torch.device("cpu"), gradient_accumulation_steps=2,
            counterfactual_weight=0.1, counterfactual_fraction=1.0,
            return_diagnostics_summary=True,
        )
        self.assertEqual(len(losses), 2)
        self.assertEqual(scheduler.last_epoch, 1)
        self.assertEqual(len(em), 1)
        self.assertEqual(len(f1), 1)
        self.assertEqual(diagnostics, {})


if __name__ == "__main__":
    unittest.main()
