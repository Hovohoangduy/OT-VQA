"""Dataset flags select Python defaults while preserving CLI overrides."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from configs.arg_parser import get_args
from utils.metrics import PAPER_METRICS, SCIVQA_METRICS, metrics_for_dataset


class DatasetPresetTests(unittest.TestCase):
    def test_scivqa_flag_selects_paths_lengths_and_metrics(self):
        for evaluation in (False, True):
            args = get_args(["--dataset", "SciVQA"], evaluation=evaluation)
            self.assertEqual(args.train_csv_path, "data/scivqa_dataset/train.csv")
            self.assertEqual(args.dev_csv_path, "data/scivqa_dataset/val.csv")
            self.assertEqual(args.test_csv_path, "data/scivqa_dataset/test.csv")
            self.assertEqual(args.img_path, "data/scivqa_dataset/images")
            self.assertEqual(args.model_path, "data/scivqa_model")
            self.assertEqual((args.max_question_tokens, args.max_answer_tokens), (256, 64))
            self.assertEqual(args.epochs, 30)
            self.assertEqual(args.bertscore_model, "roberta-large")
            self.assertFalse(args.bertscore_rescale)
            self.assertEqual(metrics_for_dataset(args.dataset), SCIVQA_METRICS)
            self.assertFalse(hasattr(args, "config"))

    def test_plantexpert_flag_selects_paths_lengths_and_metrics(self):
        for evaluation in (False, True):
            args = get_args(["--dataset", "plantexpert"], evaluation=evaluation)
            self.assertEqual(args.train_csv_path, "data/plantexpert_dataset/train.csv")
            self.assertEqual(args.dev_csv_path, "data/plantexpert_dataset/val.csv")
            self.assertEqual(args.test_csv_path, "data/plantexpert_dataset/test.csv")
            self.assertEqual(args.img_path, "data/plantexpert_dataset/images")
            self.assertEqual(args.model_path, "data/plantexpert_model")
            self.assertEqual((args.max_question_tokens, args.max_answer_tokens), (32, 112))
            self.assertEqual((args.batch_size, args.epochs), (4, 10))
            self.assertEqual(args.bertscore_model, "bert-base-uncased")
            self.assertTrue(args.bertscore_rescale)
            self.assertEqual(metrics_for_dataset(args.dataset), PAPER_METRICS)

    def test_cli_overrides_dataset_preset_in_any_order(self):
        flags = ["--batch_size", "2", "--epochs", "1", "--max_question_tokens", "512",
                 "--train_csv_path", "custom/train.csv", "--dev_csv_path", "custom/val.csv",
                 "--img_path", "custom/images", "--model_path", "custom/model",
                 "--bertscore_rescale"]
        for argv in (["--dataset", "scivqa", *flags], [*flags, "--dataset", "scivqa"]):
            args = get_args(argv)
            self.assertEqual((args.batch_size, args.epochs, args.max_question_tokens), (2, 1, 512))
            self.assertEqual(args.train_csv_path, "custom/train.csv")
            self.assertEqual(args.dev_csv_path, "custom/val.csv")
            self.assertEqual(args.img_path, "custom/images")
            self.assertEqual(args.model_path, "custom/model")
            self.assertTrue(args.bertscore_rescale)

    def test_entry_point_reads_dataset_from_sys_argv(self):
        with patch("sys.argv", ["train.py", "--dataset", "scivqa", "--epochs", "1"]):
            args = get_args()
        self.assertEqual(args.dataset, "scivqa")
        self.assertEqual(args.train_csv_path, "data/scivqa_dataset/train.csv")
        self.assertEqual(args.epochs, 1)

    def test_python_defaults_do_not_depend_on_working_directory(self):
        original_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                args = get_args(["--dataset", "scivqa"])
            finally:
                os.chdir(original_directory)
        self.assertEqual(args.train_csv_path, "data/scivqa_dataset/train.csv")
        self.assertFalse(hasattr(args, "config"))

    def test_auto_and_gqa_keep_existing_defaults(self):
        for flags in ([], ["--dataset", "auto"], ["--dataset", "gqa"]):
            args = get_args(flags)
            self.assertEqual(args.train_csv_path, "data/gqa_dataset/train.csv")
            self.assertEqual(args.model_path, "data/gqa_model")
            self.assertEqual(args.epochs, 10)
            self.assertFalse(hasattr(args, "config"))


if __name__ == "__main__":
    unittest.main()
