"""Offline regressions for the simplified Cross-Attention VQA pipeline."""

from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from transformers import (
    BertConfig, BertModel, BertTokenizer,
    DeiTConfig, DeiTImageProcessor, DeiTModel,
)

from model.vqa_model import VQAModel
from model.ot_alignment import (
    AlignmentNegativeQueue, OTAlignmentConfig, OTContrastiveAligner,
)
from utils.checkpoint import (
    load_model, load_student_initialization, read_checkpoint,
    restore_alignment_state, save_checkpoint,
)


class ModelLogicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.text = cls.root / "text"
        cls.text.mkdir()
        vocab = [
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
            "red", "blue", "what", "color", "?", "a", "b",
        ]
        (cls.text / "vocab.txt").write_text("\n".join(vocab) + "\n")
        BertTokenizer(vocab=str(cls.text / "vocab.txt")).save_pretrained(cls.text)
        BertModel(BertConfig(
            vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1,
            num_attention_heads=4, intermediate_size=32,
            hidden_dropout_prob=0, attention_probs_dropout_prob=0,
        )).save_pretrained(cls.text)
        cls.visual = cls.root / "visual"
        DeiTModel(DeiTConfig(
            hidden_size=16, num_hidden_layers=1, num_attention_heads=4,
            intermediate_size=32, image_size=32, patch_size=16,
        )).save_pretrained(cls.visual)
        DeiTImageProcessor(
            size={"height": 32, "width": 32},
            crop_size={"height": 32, "width": 32},
        ).save_pretrained(cls.visual)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def make_model(
        self, skip_encoders=False, embeddings_path=None, freeze=False,
        fusion="cross_attention",
    ):
        return VQAModel(
            text_model=str(self.text), image_model=str(self.visual),
            d_model=16, ffn_hidden=32, num_layers=1,
            num_heads=4, drop_prob=0, fusion=fusion,
            fusion_config={
                "layers": 1, "heads": 4, "ffn_hidden": 32, "dropout": 0,
            } if fusion == "cross_attention" else None,
            routing_config={
                "slots": 2, "reasoning_steps": 2, "routing_dim": 16,
                "heads": 4, "dropout": 0, "epsilon": 0.2, "tau": 0.5,
                "sinkhorn_iterations": 5,
            } if fusion != "cross_attention" else None,
            freeze_answer_embeddings=freeze,
            skip_encoders=skip_encoders,
            embeddings_path=embeddings_path,
        )

    def test_shifted_targets_cross_attention_and_backward(self):
        model = self.make_model()
        logits, targets = model(
            torch.rand(1, 3, 32, 32), ["what color ?"], ["red"], max_len=6,
        )
        self.assertEqual(logits.shape, (1, 5, 12))
        self.assertEqual(targets.tolist(), [[5, 3, 0, 0, 0]])
        loss = nn.functional.cross_entropy(
            logits.transpose(1, 2), targets, ignore_index=model.pad_token_id,
        )
        loss.backward()
        self.assertIsNotNone(model.fusion_module.layers[0].q.weight.grad)
        self.assertIsNotNone(
            model.answer_embedding.token_embeddings.word_embeddings.weight.grad
        )
        self.assertIsNone(next(model.image_model.model.parameters()).grad)
        self.assertFalse(any(
            parameter.requires_grad
            for parameter in model.question_encoder.text_encoder.parameters()
        ))

    def test_raw_and_cached_feature_paths_agree(self):
        model = self.make_model().eval()
        images = torch.rand(2, 3, 32, 32)
        questions = ["what color ?", "color ?"]
        with torch.no_grad():
            online = model.encode(images, questions, return_diagnostics=True)
            image_features, _ = model.image_model(images)
            question_features, question_mask, _ = model.question_encoder.encode_tokens(
                questions
            )
            cached = model.encode_from_features(
                image_features.half(), question_features.half(), question_mask,
                return_diagnostics=True,
            )
        torch.testing.assert_close(online.memory, cached.memory, atol=2e-3, rtol=2e-3)
        self.assertEqual(cached.memory_padding_mask.tolist(), question_mask.tolist())
        self.assertEqual(
            cached.fusion_output.attention_weights.shape[:3], (2, 4, 5)
        )

    def test_generation_stops_at_eos(self):
        model = self.make_model()
        original_decode = model.decode
        calls = []

        def decode(ids, memory, **kwargs):
            calls.append(ids.clone())
            output = torch.zeros(1, ids.size(1), 12)
            output[:, -1, 5 if ids.size(1) == 1 else 3] = 10
            return output

        model.decode = decode
        result = model.generate(torch.rand(1, 3, 32, 32), ["what color ?"], max_len=6)
        model.decode = original_decode
        self.assertEqual(result.tolist(), [[5, 3]])
        self.assertEqual(model.answers_from_ids(result), ["red"])
        self.assertEqual(len(calls), 2)

    def test_checkpoint_round_trip(self):
        model = self.make_model().eval()
        path = self.root / "student.pt"
        save_checkpoint(
            path, model=model, text_model=str(self.text), image_model=str(self.visual),
        )
        restored = load_model(path, torch.device("cpu"))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[name])

    def test_ot_routing_raw_cache_backward_and_checkpoint(self):
        model = self.make_model(fusion="ot_evidence_routing")
        images = torch.rand(2, 3, 32, 32)
        questions = ["what color ?", "color ?"]
        answers = ["red", "blue"]
        logits, targets, output = model(
            images, questions, answers, max_len=6, return_diagnostics=True,
        )
        self.assertEqual(logits.shape, (2, 5, 12))
        self.assertEqual(output.memory.shape, (2, 2, 16))
        self.assertIn("routing_null", output.diagnostics)
        nn.functional.cross_entropy(
            logits.transpose(1, 2), targets, ignore_index=model.pad_token_id,
        ).backward()
        self.assertIsNotNone(model.fusion_module.cost_query.weight.grad)

        with torch.no_grad():
            image_features, _ = model.image_model(images)
            question_features, question_mask, _ = model.question_encoder.encode_tokens(
                questions
            )
            online = model.encode(images, questions).memory
            cached = model.encode_from_features(
                image_features.half(), question_features.half(), question_mask,
            ).memory
        torch.testing.assert_close(online, cached, atol=2e-3, rtol=2e-3)

        path = self.root / "ot-routing.pt"
        save_checkpoint(
            path, model=model, text_model=str(self.text), image_model=str(self.visual),
        )
        payload = read_checkpoint(path, torch.device("cpu"))
        self.assertEqual(payload["architecture"], "ot_evidence_routing_v1")
        restored = load_model(path, torch.device("cpu"))
        self.assertEqual(restored.fusion_type, "ot_evidence_routing")
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[name])
        model.eval()
        with torch.no_grad():
            expected = model.generate(images, questions, max_len=6)
            actual = restored.generate(images, questions, max_len=6)
        torch.testing.assert_close(expected, actual)

    def test_softmax_routing_model_constructs(self):
        model = self.make_model(fusion="softmax_evidence_routing")
        output = model.encode(
            torch.rand(1, 3, 32, 32), ["what color ?"], return_diagnostics=True,
        )
        self.assertEqual(output.memory.shape, (1, 2, 16))
        self.assertEqual(model.fusion_module.routing_mode, "softmax")

    def test_v2_routed_memory_checkpoint_and_matched_control(self):
        ot = self.make_model(fusion="ot_evidence_routing_v2")
        control = self.make_model(fusion="softmax_evidence_routing_v2")
        control.load_state_dict(ot.state_dict())
        images = torch.rand(2, 3, 32, 32)
        questions = ["what color ?", "color ?"]
        ot.eval()
        control.eval()
        with torch.no_grad():
            routed = ot.encode(images, questions, return_diagnostics=True)
        # Five padded/tokenizer positions, two slots, and four spatial DeiT patches.
        self.assertEqual(routed.memory.shape[1], 11)
        self.assertIn("routing_gate_mean", routed.fusion_output.diagnostics)
        self.assertEqual(ot.fusion_module.config.memory_mode, "routed_patches")
        self.assertEqual(ot.fusion_module.config.preference_transform, "sparsemax")

        path = self.root / "ot-routing-v2.pt"
        save_checkpoint(
            path, model=ot, text_model=str(self.text), image_model=str(self.visual),
        )
        payload = read_checkpoint(path, torch.device("cpu"))
        self.assertEqual(payload["architecture"], "ot_evidence_routing_v2")
        load_student_initialization(path, control)
        restored = load_model(path, torch.device("cpu"))
        self.assertEqual(restored.fusion_type, "ot_evidence_routing_v2")

    def test_routing_initialization_can_transfer_between_matched_controls(self):
        source = self.make_model(fusion="ot_evidence_routing")
        path = self.root / "routing-initialization.pt"
        save_checkpoint(
            path, model=source, text_model=str(self.text), image_model=str(self.visual),
        )
        target = self.make_model(fusion="softmax_evidence_routing")
        load_student_initialization(path, target)
        for name, value in source.state_dict().items():
            torch.testing.assert_close(value, target.state_dict()[name])

    def test_removed_fusion_checkpoint_fails_clearly(self):
        path = self.root / "legacy.pt"
        torch.save({
            "format_version": 3,
            "architecture": "cross_attention_only_v1",
            "model_config": {"fusion": "uot_cross_attention"},
        }, path)
        with self.assertRaisesRegex(ValueError, "was removed"):
            load_model(path, torch.device("cpu"))

    def test_common_initialization_restores_identical_weights(self):
        source = self.make_model()
        path = self.root / "initialization.pt"
        save_checkpoint(
            path, model=source, text_model=str(self.text), image_model=str(self.visual),
        )
        target = self.make_model()
        with torch.no_grad():
            next(target.fusion_module.parameters()).add_(1)
        load_student_initialization(path, target)
        for name, value in source.state_dict().items():
            torch.testing.assert_close(value, target.state_dict()[name])

    def test_cached_student_exports_for_online_inference(self):
        full = self.make_model()
        embeddings = self.root / "embeddings.pt"
        torch.save(full.answer_embedding.token_embeddings.state_dict(), embeddings)
        cached = self.make_model(skip_encoders=True, embeddings_path=embeddings)
        path = self.root / "cached.pt"
        save_checkpoint(
            path, model=cached, text_model=str(self.text), image_model=str(self.visual),
        )
        restored = load_model(path, torch.device("cpu"))
        self.assertIsNotNone(restored.image_model.model)
        self.assertIsNotNone(restored.question_encoder.text_encoder)
        for name, value in cached.fusion_module.state_dict().items():
            torch.testing.assert_close(value, restored.fusion_module.state_dict()[name])

    def test_version_four_restores_teacher_queue_and_stage(self):
        model = self.make_model()
        config = OTAlignmentConfig(
            ot_dim=8, max_iterations=5, negative_count=1,
        )
        teacher = OTContrastiveAligner(16, 16, config)
        queue = AlignmentNegativeQueue(2)
        queue.enqueue(
            torch.randn(1, 4, 16), torch.randn(1, 3, 16),
            torch.zeros(1, 4, dtype=torch.bool),
            torch.zeros(1, 3, dtype=torch.bool),
        )
        path = self.root / "training-v4.pt"
        save_checkpoint(
            path, model=model, text_model=str(self.text),
            image_model=str(self.visual), format_version=4,
            alignment_teacher=teacher,
            alignment_config={"teacher": config.to_dict()},
            negative_queue=queue,
            training_stage={"phase": "alignment_warmup"},
        )
        checkpoint = read_checkpoint(path, torch.device("cpu"))
        restored_teacher = OTContrastiveAligner(16, 16, config)
        restored_queue = AlignmentNegativeQueue(1)
        stage = restore_alignment_state(
            checkpoint, restored_teacher, restored_queue
        )
        self.assertEqual(stage["phase"], "alignment_warmup")
        self.assertEqual(len(restored_queue), 1)
        for name, value in teacher.state_dict().items():
            torch.testing.assert_close(value, restored_teacher.state_dict()[name])

    def test_answer_embeddings_can_be_frozen(self):
        model = self.make_model(freeze=True)
        model.train()
        self.assertFalse(model.answer_embedding.token_embeddings.training)
        self.assertFalse(any(
            parameter.requires_grad
            for parameter in model.answer_embedding.token_embeddings.parameters()
        ))


if __name__ == "__main__":
    unittest.main()
