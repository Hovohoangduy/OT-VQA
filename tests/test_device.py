import unittest
from unittest.mock import patch

from configs.arg_parser import get_args
from configs.config import Config
from utils.device import resolve_device


class DeviceSelectionTests(unittest.TestCase):
    def test_cli_defaults_to_auto_and_accepts_mps(self):
        args = get_args([])
        self.assertEqual(args.device, "auto")
        self.assertEqual(args.text_model, "bert-base-uncased")
        self.assertEqual(args.image_model, "google/vit-base-patch16-224-in21k")
        self.assertEqual(args.image_model, Config.image_model)
        self.assertFalse(hasattr(args, "language"))
        self.assertEqual(args.train_csv_path, "data/gqa_dataset/train.csv")
        self.assertEqual(args.d_model, 384)
        self.assertEqual(args.ffn_hidden, 1024)
        self.assertEqual(args.num_layers, 2)
        self.assertEqual(args.drop_prob, 0.2)
        self.assertEqual(args.fusion, "cross_attention")
        self.assertEqual(args.routing_slots, 4)
        self.assertEqual(args.routing_steps, 2)
        self.assertEqual(args.routing_dim, 256)
        self.assertEqual(args.routing_tau, 0.5)
        self.assertFalse(hasattr(args, "ot_profile"))
        self.assertEqual(args.weight_decay, 0.05)
        self.assertEqual(args.gradient_clip, 1.0)
        self.assertTrue(args.freeze_answer_embeddings)
        self.assertFalse(
            get_args(["--no-freeze_answer_embeddings"]).freeze_answer_embeddings
        )
        self.assertEqual(get_args(["--device", "mps"]).device, "mps")

    def test_training_command_compatibility_aliases(self):
        args = get_args([
            "--fusion_method", "ot_evidence_routing",
            "--train_csv", "train.csv",
            "--dev_csv", "val.csv",
            "--test_csv", "test.csv",
            "--save_dir", "results/run",
            "--distributed",
        ])
        self.assertEqual(args.fusion, "ot_evidence_routing")
        self.assertEqual(args.train_csv_path, "train.csv")
        self.assertEqual(args.dev_csv_path, "val.csv")
        self.assertEqual(args.test_csv_path, "test.csv")
        self.assertEqual(args.model_path, "results/run")
        self.assertTrue(args.distributed)

    def test_v2_routing_defaults(self):
        args = get_args(["--fusion", "ot_evidence_routing_v2"])
        self.assertEqual(args.routing_tau, 0.1)
        self.assertEqual(args.routing_query_diversity_weight, 0.0)
        self.assertIsNone(args.routing_preference_transform)
        control = get_args(["--fusion", "softmax_evidence_routing_v2"])
        self.assertEqual(control.routing_tau, 0.1)

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
