"""Small offline checks for the interchangeable fusion implementations."""

import tempfile
import unittest
from pathlib import Path

import torch
from transformers import (BertConfig, BertModel, BertTokenizer, ViTConfig,
                          ViTImageProcessor, ViTModel)

from model.fusion import (BANFusion, CrossAttentionFusion, QFormerFusion,
                          SANFusion)
from model.vqa_model import VQAModel
from utils.checkpoint import load_model, save_checkpoint


class FusionModuleTests(unittest.TestCase):
    def test_padding_shapes_and_gradients(self):
        torch.manual_seed(5)
        modules = [SANFusion(12, 8, 0, 2), BANFusion(12, 8, 0, 2),
                   CrossAttentionFusion(12, 8, 2, 0),
                   QFormerFusion(12, 8, 2, 0, 6, 2)]
        valid = torch.tensor([[True, False, False], [True, True, True]])
        for module in modules:
            with self.subTest(module=type(module).__name__):
                image = torch.randn(2, 4, 8, requires_grad=True)
                question = torch.randn(2, 3, 12, requires_grad=True)
                memory, blocked = module(image, question, valid)
                self.assertEqual(memory.shape[:2], blocked.shape)
                self.assertTrue(torch.isfinite(memory).all())
                if not isinstance(module, QFormerFusion):
                    self.assertEqual(blocked[0].tolist(), [False, False, True, True])
                    self.assertEqual(memory[0, -1].abs().sum().item(), 0)
                else:
                    self.assertFalse(blocked.any())
                    self.assertEqual(memory.size(1), 6)
                memory.square().sum().backward()
                self.assertGreater(image.grad.abs().sum().item(), 0)
                self.assertGreater(question.grad.abs().sum().item(), 0)


class ModelFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.directory = tempfile.TemporaryDirectory()
        root = Path(cls.directory.name)
        cls.text = root / "text"
        cls.text.mkdir()
        vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "red", "what", "color", "?"]
        (cls.text / "vocab.txt").write_text("\n".join(vocab) + "\n")
        BertTokenizer(vocab=str(cls.text / "vocab.txt")).save_pretrained(cls.text)
        BertModel(BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1,
                             num_attention_heads=4, intermediate_size=32)).save_pretrained(cls.text)
        cls.visual = root / "visual"
        ViTModel(ViTConfig(hidden_size=16, num_hidden_layers=1, num_attention_heads=4,
                           intermediate_size=32, image_size=32, patch_size=16)).save_pretrained(cls.visual)
        ViTImageProcessor(size={"height": 32, "width": 32},
                          crop_size={"height": 32, "width": 32}).save_pretrained(cls.visual)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_each_method_trains_generates_and_loads(self):
        image = torch.rand(2, 3, 32, 32)
        for method in ("san", "ban", "cross_attention", "qformer"):
            with self.subTest(method=method):
                model = VQAModel(text_model=str(self.text), image_model=str(self.visual),
                                 output_size=16, d_model=16, ffn_hidden=32, num_layers=1,
                                 num_heads=4, drop_prob=0, fusion=method)
                logits, targets = model(image, ["what color ?", "red"],
                                        ["red", "red"], max_len=5)
                self.assertEqual(logits.shape, (2, 4, 9))
                torch.nn.functional.cross_entropy(
                    logits.transpose(1, 2), targets,
                    ignore_index=model.pad_token_id).backward()
                self.assertTrue(any(p.grad is not None for p in
                                    model.fusion_module.parameters()))
                with self.assertRaisesRegex(ValueError, "only for OT"):
                    model.encode(image, ["red", "red"], return_transport=True)
                path = Path(self.directory.name) / f"{method}.pt"
                model.eval()
                save_checkpoint(path, model=model, text_model=str(self.text),
                                image_model=str(self.visual))
                restored = load_model(path, torch.device("cpu"))
                self.assertEqual(restored.fusion, method)
                with torch.no_grad():
                    torch.testing.assert_close(restored(image, ["red", "red"],
                                                        ["red", "red"], max_len=5)[0],
                                               model(image, ["red", "red"],
                                                     ["red", "red"], max_len=5)[0])
                    self.assertEqual(restored.generate(image, ["red", "red"],
                                                       max_len=5).size(0), 2)


if __name__ == "__main__":
    unittest.main()
