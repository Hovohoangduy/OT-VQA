"""Transport solver and VQA integration tests with locally initialized encoders."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from PIL import Image
import torch
from torch import nn
from transformers import BertConfig, BertModel, BertTokenizer, ViTConfig, ViTImageProcessor, ViTModel

from model.optimal_transport import PartialTransportFusion, log_sinkhorn
from model.vqa_model import VQAModel
from configs.arg_parser import get_args
import train as training_script
from utils.checkpoint import load_model, save_checkpoint


class SinkhornTests(unittest.TestCase):
    def test_rectangular_masked_marginals_and_gradients(self):
        cost = torch.tensor([[[0.2, 0.9, 0.0], [0.8, 0.1, 0.0],
                              [0.3, 0.4, 0.0], [0.0, 0.0, 0.0]]], requires_grad=True)
        rows = torch.tensor([[0.2, 0.3, 0.5, 0.0]])
        columns = torch.tensor([[0.4, 0.6, 0.0]])
        plan = log_sinkhorn(cost, rows, columns, epsilon=0.1, iterations=100)
        self.assertEqual(plan.shape, (1, 4, 3))
        self.assertEqual(plan[0, -1].sum().item(), 0.0)
        self.assertEqual(plan[0, :, -1].sum().item(), 0.0)
        torch.testing.assert_close(plan.sum(-1), rows, atol=1e-4, rtol=0)
        torch.testing.assert_close(plan.sum(-2), columns, atol=1e-4, rtol=0)
        (plan * cost).sum().backward()
        self.assertTrue(torch.isfinite(cost.grad).all())
        self.assertGreater(cost.grad.abs().sum().item(), 0)

    def test_half_precision_inputs_are_solved_in_float32(self):
        rows = torch.tensor([[0.5, 0.5]])
        columns = torch.tensor([[0.25, 0.75]])
        for dtype in (torch.float16, torch.bfloat16):
            plan = log_sinkhorn(torch.ones(1, 2, 2, dtype=dtype), rows, columns)
            self.assertEqual(plan.dtype, torch.float32)
            torch.testing.assert_close(plan.sum(-1), rows, atol=1e-5, rtol=0)
            torch.testing.assert_close(plan.sum(-2), columns, atol=1e-5, rtol=0)

    def test_invalid_marginals_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "equal total mass"):
            log_sinkhorn(torch.ones(1, 2, 2), torch.tensor([[0.5, 0.5]]),
                         torch.tensor([[0.3, 0.3]]))

    def test_full_transport_without_dustbin_capacity(self):
        fusion = PartialTransportFusion(text_dim=8, d_model=8, dustbin_mass=0,
                                        iterations=50)
        _, _, diagnostics = fusion(torch.randn(1, 4, 8), torch.randn(1, 3, 8),
                                    torch.tensor([[True, True, True]]),
                                    return_transport=True)
        self.assertAlmostEqual(diagnostics['matched_mass'].item(), 1.0, places=4)
        self.assertEqual(diagnostics['plan'][0, -1].sum().item(), 0.0)
        self.assertEqual(diagnostics['plan'][0, :, -1].sum().item(), 0.0)


class FusionTests(unittest.TestCase):
    def test_variable_question_lengths_and_dustbin_mass(self):
        torch.manual_seed(5)
        fusion = PartialTransportFusion(text_dim=12, d_model=8, iterations=80)
        images = torch.randn(2, 5, 8, requires_grad=True)
        questions = torch.randn(2, 4, 12, requires_grad=True)
        mask = torch.tensor([[True, True, False, False], [True, True, True, True]])
        memory, blocked, diagnostics = fusion(images, questions, mask, return_transport=True)
        self.assertEqual(memory.shape, (2, 5, 8))
        self.assertEqual(blocked.tolist(), [[False, False, False, True, True],
                                            [False, False, False, False, False]])
        self.assertEqual(diagnostics["plan"].shape, (2, 6, 5))
        self.assertEqual(diagnostics["plan"][0, :, 2:4].sum().item(), 0.0)
        self.assertLess(diagnostics["row_residual"].max().item(), 0.01)
        self.assertLess(diagnostics["column_residual"].max().item(), 0.01)
        memory.square().sum().backward()
        self.assertGreater(images.grad.abs().sum().item(), 0)
        self.assertGreater(questions.grad.abs().sum().item(), 0)
        self.assertIsNotNone(fusion.question_projection.weight.grad)


class OTVQATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.text = cls.root / "text"
        cls.text.mkdir()
        vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "red", "blue", "what", "color", "?"]
        (cls.text / "vocab.txt").write_text("\n".join(vocab) + "\n")
        BertTokenizer(vocab=str(cls.text / "vocab.txt")).save_pretrained(cls.text)
        BertModel(BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1,
                             num_attention_heads=4, intermediate_size=32,
                             hidden_dropout_prob=0, attention_probs_dropout_prob=0)).save_pretrained(cls.text)
        cls.visual = cls.root / "visual"
        ViTModel(ViTConfig(hidden_size=16, num_hidden_layers=1, num_attention_heads=4,
                           intermediate_size=32, image_size=32, patch_size=16)).save_pretrained(cls.visual)
        ViTImageProcessor(size={"height": 32, "width": 32},
                          crop_size={"height": 32, "width": 32}).save_pretrained(cls.visual)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def make_model(self):
        return VQAModel(text_model=str(self.text), image_model=str(self.visual), fusion="ot",
                        output_size=16, d_model=16, ffn_hidden=32, num_layers=1,
                        drop_prob=0, ot_iterations=50)

    def test_ot_generation_and_checkpoint_round_trip(self):
        model = self.make_model().eval()
        images = torch.rand(2, 3, 32, 32)
        memory, blocked, details = model.encode(images, ["what color ?", "red"],
                                                return_transport=True)
        self.assertEqual(memory.shape[0], 2)
        self.assertEqual(blocked.shape, memory.shape[:2])
        self.assertTrue(torch.isfinite(details["plan"]).all())
        logits, targets = model(images, ["what color ?", "red"], ["red", "blue"],
                                max_len=6)
        self.assertEqual(logits.shape, (2, 5, 10))
        nn.functional.cross_entropy(logits.transpose(1, 2), targets,
                                    ignore_index=model.pad_token_id).backward()
        self.assertIsNotNone(model.ot_fusion.question_projection.weight.grad)
        path = self.root / "ot-checkpoint.pt"
        save_checkpoint(path, model=model, text_model=str(self.text), image_model=str(self.visual))
        restored = load_model(path, torch.device("cpu"))
        self.assertEqual(restored.fusion, "ot")
        with torch.no_grad():
            torch.testing.assert_close(restored(images, ["what color ?", "red"],
                                                ["red", "blue"], max_len=6)[0], logits)
            generated = restored.generate(images[:1], ["what color ?"], max_len=6)
        self.assertEqual(generated.shape[0], 1)

    def test_ot_model_learns_then_generates_one_answer(self):
        torch.manual_seed(11)
        model = self.make_model()
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        image = torch.rand(1, 3, 32, 32)
        losses = []
        for _ in range(35):
            logits, target = model(image, ["what color ?"], ["red"], max_len=6)
            loss = nn.functional.cross_entropy(logits.transpose(1, 2), target,
                                               ignore_index=model.pad_token_id)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        self.assertLess(losses[-1], losses[0] * 0.25)
        self.assertEqual(model.answers_from_ids(model.generate(image, ["what color ?"], max_len=6)),
                         ["red"])

    def test_training_entrypoint_writes_ot_checkpoint_and_manifest(self):
        image_root = self.root / "images"
        for split in ("train", "val"):
            folder = image_root / split
            folder.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (32, 32), color="red").save(folder / "a.jpg")
            pd.DataFrame([{"anno_id": f"{split}-a", "image": f"{split}/a.jpg",
                           "question": "what color ?", "answer": "red"}]).to_csv(
                               self.root / f"{split}.csv", index=False)
        output = self.root / "training-output"
        args = get_args([
            "--train_csv_path", str(self.root / "train.csv"),
            "--dev_csv_path", str(self.root / "val.csv"),
            "--img_path", str(image_root), "--model_path", str(output),
            "--text_model", str(self.text), "--image_model", str(self.visual),
            "--d_model", "16", "--ffn_hidden", "32", "--num_layers", "1",
            "--batch_size", "1", "--epochs", "1", "--fusion", "ot",
            "--device", "cpu",
        ])
        class ConstantScorer:
            hash = "test-scorer"

            def score(self, candidates, references):
                return None, None, torch.ones(len(candidates))

        with patch.object(training_script, "get_args", return_value=args), \
             patch.object(training_script, "build_bertscore_scorer", return_value=ConstantScorer()):
            training_script.main()
        self.assertTrue((output / "run_config.json").is_file())
        self.assertTrue((output / "best.pt").is_file())
        self.assertEqual(load_model(output / "best.pt", torch.device("cpu")).fusion, "ot")


if __name__ == "__main__":
    unittest.main()
