import unittest
from unittest.mock import patch

from configs.arg_parser import get_args
from utils.device import resolve_device


class DeviceSelectionTests(unittest.TestCase):
    def test_cli_defaults_to_auto_and_accepts_mps(self):
        args = get_args([])
        self.assertEqual(args.device, "auto")
        self.assertEqual(args.text_model, "bert-base-uncased")
        self.assertEqual(args.image_model, "google/vit-base-patch16-224-in21k")
        self.assertFalse(hasattr(args, "language"))
        self.assertEqual(args.train_csv_path, "data/gqa_dataset/train.csv")
        self.assertEqual(args.d_model, 384)
        self.assertEqual(args.ffn_hidden, 1024)
        self.assertEqual(args.num_layers, 2)
        self.assertEqual(args.drop_prob, 0.2)
        self.assertEqual(args.weight_decay, 0.05)
        self.assertEqual(args.gradient_clip, 1.0)
        self.assertTrue(args.freeze_answer_embeddings)
        self.assertFalse(
            get_args(["--no-freeze_answer_embeddings"]).freeze_answer_embeddings
        )
        self.assertEqual(get_args(["--device", "mps"]).device, "mps")

    def test_auto_prefers_cuda_then_mps_then_cpu(self):
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.backends.mps.is_available", return_value=True):
            self.assertEqual(resolve_device("auto").type, "cuda")
        with patch("torch.cuda.is_available", return_value=False), \
             patch("torch.backends.mps.is_available", return_value=True):
            self.assertEqual(resolve_device("auto").type, "mps")
        with patch("torch.cuda.is_available", return_value=False), \
             patch("torch.backends.mps.is_available", return_value=False):
            self.assertEqual(resolve_device("auto").type, "cpu")

    def test_explicit_unavailable_mps_fails_clearly(self):
        with patch("torch.backends.mps.is_available", return_value=False), \
             patch("torch.backends.mps.is_built", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "MPS is built but unavailable"):
                resolve_device("mps")


if __name__ == "__main__":
    unittest.main()
