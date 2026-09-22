"""Tests for contrastive UOT alignment and Cross-Attention distillation."""

import unittest

import torch

from model.ot_alignment import (
    AlignmentNegativeQueue,
    OTAlignmentConfig,
    OTContrastiveAligner,
    ot_attention_distillation_loss,
)
from model.fusion_methods import (
    CrossAttentionFusion, CrossAttentionFusionConfig, FusionInput,
)
from utils.ot_alignment_training import teacher_passes_gate


class OTContrastiveAlignmentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        self.config = OTAlignmentConfig(
            ot_dim=8,
            max_iterations=10,
            negative_count=2,
            tolerance=1e-3,
        )

    def make_inputs(self, batch=3):
        visual = torch.randn(batch, 4, 6)
        question = torch.randn(batch, 5, 7)
        visual_mask = torch.zeros(batch, 4, dtype=torch.bool)
        question_mask = torch.tensor(
            [[False, False, False, True, True]] * batch
        )
        return visual, question, visual_mask, question_mask

    def test_positive_and_hard_negative_pairs_have_finite_scores_and_gradients(self):
        aligner = OTContrastiveAligner(6, 7, self.config)
        inputs = self.make_inputs()
        output = aligner(*inputs)
        self.assertEqual(output.positive_transport.plan.shape, (3, 4, 5))
        self.assertEqual(output.image_to_question_scores.shape, (3, 3))
        self.assertEqual(output.question_to_image_scores.shape, (3, 3))
        diagonal = torch.arange(3)[:, None]
        self.assertFalse(
            output.image_to_question_negative_indices.eq(diagonal).any()
        )
        self.assertFalse(
            output.question_to_image_negative_indices.eq(diagonal).any()
        )
        self.assertTrue(torch.isfinite(output.loss))
        self.assertTrue(torch.isfinite(output.positive_transport.score).all())
        output.loss.backward()
        for parameter in (
            aligner.visual_adapter.layers[0].weight,
            aligner.question_adapter.layers[0].weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_padding_and_low_mass_scores_remain_finite(self):
        config = OTAlignmentConfig(
            ot_dim=8,
            epsilon=0.05,
            tau_visual=0.01,
            tau_question=0.01,
            max_iterations=10,
            negative_count=1,
        )
        aligner = OTContrastiveAligner(6, 7, config)
        visual, question, visual_mask, question_mask = self.make_inputs(batch=2)
        visual_mask[0, -1] = True
        output = aligner(visual, question, visual_mask, question_mask)
        self.assertTrue(torch.isfinite(output.positive_transport.score).all())
        self.assertTrue((output.positive_transport.plan[0, -1] == 0).all())
        self.assertTrue((output.positive_transport.plan[:, :, 3:] == 0).all())

    def test_identical_features_trigger_collapse_diagnostic(self):
        aligner = OTContrastiveAligner(6, 7, self.config)
        visual = torch.ones(3, 4, 6)
        question = torch.ones(3, 5, 7)
        visual_mask = torch.zeros(3, 4, dtype=torch.bool)
        question_mask = torch.zeros(3, 5, dtype=torch.bool)
        output = aligner(visual, question, visual_mask, question_mask)
        self.assertTrue(output.collapsed.item())

    def test_queue_round_trip_and_singleton_negative(self):
        queue = AlignmentNegativeQueue(capacity=2)
        inputs = self.make_inputs(batch=2)
        queue.enqueue(*inputs)
        state = queue.state_dict()
        restored = AlignmentNegativeQueue(capacity=1)
        restored.load_state_dict(state)
        self.assertEqual(len(restored), 2)
        for expected, actual in zip(queue.tensors(torch.device("cpu")),
                                    restored.tensors(torch.device("cpu"))):
            torch.testing.assert_close(expected, actual)

        aligner = OTContrastiveAligner(6, 7, self.config)
        singleton = tuple(value[:1] for value in inputs)
        output = aligner(*singleton, queue=restored)
        self.assertEqual(output.image_to_question_scores.shape, (1, 3))

    def test_attention_distillation_detaches_teacher_and_trains_student(self):
        plan = torch.rand(2, 4, 3, requires_grad=True)
        logits = torch.randn(2, 2, 3, 4, requires_grad=True)
        attention = torch.softmax(logits, dim=-1)
        question_mask = torch.tensor([
            [False, False, True],
            [False, False, False],
        ])
        loss = ot_attention_distillation_loss(plan, attention, question_mask)
        loss.backward()
        self.assertIsNone(plan.grad)
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_cross_attention_exposes_final_layer_weights_for_distillation(self):
        fusion = CrossAttentionFusion(
            6, 7, 8,
            CrossAttentionFusionConfig(
                layers=2, heads=2, ffn_hidden=16, dropout=0
            ),
        )
        visual, question, visual_mask, question_mask = self.make_inputs(batch=2)
        output = fusion(
            FusionInput(visual, question, visual_mask, question_mask),
            return_diagnostics=True,
        )
        self.assertEqual(output.attention_weights.shape, (2, 2, 5, 4))
        loss = ot_attention_distillation_loss(
            torch.rand(2, 4, 5),
            output.attention_weights,
            question_mask,
            visual_mask,
        )
        loss.backward()
        self.assertIsNotNone(fusion.layers[-1].q.weight.grad)
        self.assertIsNotNone(fusion.layers[-1].k.weight.grad)

    def test_teacher_gate_rejects_unvalidated_alignment(self):
        valid = {
            'ot_score_margin': 0.1,
            'ot_i2q_accuracy': 0.6,
            'ot_q2i_accuracy': 0.6,
            'ot_candidate_count': 2.0,
            'ot_matched_mass': 0.8,
            'ot_entropy': 1.0,
            'ot_residual': 1e-3,
            'ot_convergence_rate': 1.0,
            'ot_visual_variance': 0.1,
            'ot_question_variance': 0.1,
            'ot_collapsed': 0.0,
        }
        self.assertTrue(teacher_passes_gate(valid, 1)[0])
        invalid = dict(valid, ot_score_margin=-0.1)
        passed, reason = teacher_passes_gate(invalid, 1)
        self.assertFalse(passed)
        self.assertIn('hard negatives', reason)


if __name__ == "__main__":
    unittest.main()
