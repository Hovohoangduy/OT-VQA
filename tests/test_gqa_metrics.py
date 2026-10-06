"""Offline checks for literal VQA consensus scoring in GQA runs."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from configs.arg_parser import get_args
from configs.config import Config
from diagnose_training import _history_report
from scripts.compare_fusions import collect_results, write_comparison
from test import evaluation, main as evaluate_main
from train import _run_training
from utils.metrics import (GQA_METRICS, PAPER_METRICS, mean_scores,
                           metrics_for_dataset, resolve_dataset, vqa_accuracy,
                           vqa_score_pairs)
from utils.vqa_dataset import VQADataset


class ConstantAnswerModel(nn.Module):
    """Small trainable stand-in; generation always produces 'red'."""

    pad_token_id = 0
    fusion = "ot"
    model_config = {"fusion": "ot"}

    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(1, 1, 2))

    def forward(self, images, questions, answers, anno_ids):
        count = len(answers)
        return (self.logits.expand(count, 1, 2),
                torch.ones(count, 1, dtype=torch.long, device=images.device))

    def evaluate_batch(self, images, questions, answers, anno_ids):
        logits, targets = self(images, questions, answers, anno_ids)
        return logits, targets, ["red"] * len(answers)

    def answers_from_ids(self, ids):
        return ids


class GQAMetricTests(unittest.TestCase):
    def test_consensus_formula_for_zero_one_two_and_three_or_more_matches(self):
        for matches in range(6):
            with self.subTest(matches=matches):
                self.assertAlmostEqual(
                    vqa_accuracy(["red"] * matches + ["blue"] * (10 - matches), "red"),
                    min(matches / 3, 1),
                )
        self.assertEqual(vqa_accuracy("red", "blue"), 0)
        self.assertAlmostEqual(vqa_accuracy("red", "red"), 1 / 3)
        self.assertAlmostEqual(vqa_accuracy("  RED   leaf ", "red leaf"), 1 / 3)
        # One multi-word string remains one vote.
        self.assertAlmostEqual(vqa_accuracy("red red red", "red red red"), 1 / 3)
        with self.assertRaises(ValueError):
            vqa_accuracy([], "red")
        with self.assertRaises(ValueError):
            vqa_score_pairs(["red"], [])
        with self.assertRaises(ValueError):
            mean_scores([], GQA_METRICS)

    def test_dataset_selection_and_cli_override(self):
        args = get_args([])
        self.assertEqual(resolve_dataset(args.dataset, args.train_csv_path), "gqa")
        self.assertEqual(metrics_for_dataset("gqa"), GQA_METRICS)
        self.assertEqual(metrics_for_dataset("plantexpert"), PAPER_METRICS)
        for path in ("data/gqa_dataset/val.csv", "/custom/GQA/test.csv", "data/gqa-val.csv"):
            self.assertEqual(resolve_dataset("auto", path), "gqa")
        for path in ("data/plantexpert_dataset/val.csv", "data/other/val.csv", "data/notgqa.csv"):
            self.assertEqual(resolve_dataset("auto", path), "plantexpert")
        self.assertEqual(resolve_dataset("gqa", "custom/train.csv"), "gqa")
        self.assertEqual(resolve_dataset("plantexpert", "gqa/train.csv"), "plantexpert")
        self.assertEqual(get_args(["--dataset", "GQA"]).dataset, "gqa")
        with self.assertRaises(ValueError):
            resolve_dataset("unknown")

    def _write_data(self, root):
        Image.new("RGB", (8, 8), "red").save(root / "sample.jpg")
        frame = pd.DataFrame({"anno_id": [0, 1, 2], "image": ["sample.jpg"] * 3,
                              "question": ["color?"] * 3,
                              "answer": ["red", "blue", "red"],
                              "question_type": ["color"] * 3})
        csv_path = root / "gqa_val.csv"
        frame.to_csv(csv_path, index=False)
        return frame, csv_path

    def test_gqa_evaluation_partial_batch_and_predictions_skip_paper_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame, _ = self._write_data(root)
            loader = DataLoader(VQADataset(frame, Config.transforms, root, dataset_name="gqa"),
                                batch_size=2)
            predictions = []
            model = ConstantAnswerModel()
            with patch("test.build_bertscore_scorer") as builder, \
                 patch("test.score_pairs") as paper_scorer:
                result = evaluation(model, loader, nn.CrossEntropyLoss(),
                                    predictions=predictions, measure_performance=True)
            builder.assert_not_called()
            paper_scorer.assert_not_called()
            self.assertEqual(result["examples"], 3)
            self.assertEqual(set(result["metrics"]), {"vqa_accuracy"})
            self.assertAlmostEqual(result["metrics"]["vqa_accuracy"], 2 / 9)
            self.assertEqual([row["vqa_accuracy"] for row in predictions], [1 / 3, 0, 1 / 3])
            self.assertFalse(set(PAPER_METRICS).intersection(predictions[0]))
            self.assertGreater(result["loss"], 0)
            self.assertEqual(result["performance"]["examples"], 3)

    def test_gqa_training_and_cli_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, csv_path = self._write_data(root)
            destination = root / "run"
            args = get_args(["--train_csv_path", str(csv_path), "--dev_csv_path", str(csv_path),
                             "--img_path", str(root), "--model_path", str(destination),
                             "--epochs", "1", "--batch_size", "2", "--device", "cpu"])
            stdout = io.StringIO()
            with patch("train.VQAModel", return_value=ConstantAnswerModel()), \
                 patch("train.save_checkpoint") as save, \
                 patch("train.build_bertscore_scorer") as training_builder, \
                 patch("test.build_bertscore_scorer") as evaluation_builder, \
                 redirect_stdout(stdout):
                _run_training(args, torch.device("cpu"), rank=0, world_size=1)
            training_builder.assert_not_called()
            evaluation_builder.assert_not_called()
            self.assertEqual(save.call_count, 2)
            history = [json.loads(line) for line in
                       (destination / "metrics.jsonl").read_text().splitlines()]
            self.assertAlmostEqual(history[0]["train_vqa_accuracy"], 2 / 9)
            self.assertAlmostEqual(history[0]["val_vqa_accuracy"], 2 / 9)
            self.assertNotIn("train_em", history[0])
            self.assertIn("vqa_accuracy=0.2222", stdout.getvalue())
            self.assertTrue((destination / "evaluation_metrics_plot.png").is_file())
            manifest = json.loads((destination / "run_config.json").read_text())
            self.assertEqual(manifest["dataset"], "gqa")
            self.assertEqual(manifest["generated_metrics"], ["vqa_accuracy"])
            self.assertNotIn("bertscore_hash", manifest)
            self.assertEqual(_history_report(history)["best_generated_metrics"],
                             {"vqa_accuracy": 2 / 9})

            # Custom paths can force GQA on both validation and test splits.
            custom_path = root / "custom.csv"
            custom_path.write_bytes(csv_path.read_bytes())
            for split in ("dev", "test"):
                report_path = destination / f"{split}_report.json"
                predictions_path = destination / f"{split}_predictions.csv"
                eval_args = get_args(["--dataset", "gqa", "--split", split,
                                      f"--{split}_csv_path", str(custom_path),
                                      "--img_path", str(root), "--device", "cpu",
                                      "--batch_size", "2", "--report_json", str(report_path),
                                      "--predictions_csv", str(predictions_path)])
                with patch("test.get_args", return_value=eval_args), \
                     patch("test.load_model", return_value=ConstantAnswerModel()), \
                     patch("test.build_bertscore_scorer") as builder, \
                     redirect_stdout(io.StringIO()):
                    evaluate_main()
                builder.assert_not_called()
                report = json.loads(report_path.read_text())
                self.assertEqual(report["dataset"], "gqa")
                self.assertEqual(report["generated_metrics"], {"vqa_accuracy": 2 / 9})
                self.assertNotIn("bertscore", report)
                predictions = pd.read_csv(predictions_path)
                self.assertEqual(len(predictions), 3)
                self.assertIn("vqa_accuracy", predictions)
                self.assertIn("question_type", predictions)
                self.assertNotIn("em", predictions)

            # Comparisons and diagnostics must also accept the GQA metric schema.
            comparison_root = root / "comparison"
            run = comparison_root / "ot_seed1"
            run.mkdir(parents=True)
            (run / "run_config.json").write_text(json.dumps(manifest))
            (run / "test_report.json").write_text(json.dumps(report))
            rows = collect_results(comparison_root, ["ot", "san"], [1], "test")
            write_comparison(comparison_root, rows)
            self.assertIn("VQA accuracy", (comparison_root / "comparison.md").read_text())
            self.assertIn("mean_vqa_accuracy", (comparison_root / "summary.csv").read_text())
            self.assertNotIn("bertscore", (comparison_root / "comparison.csv").read_text())
            rows.append({**rows[0], "em": 0.5})
            with self.assertRaisesRegex(ValueError, "different generated metrics"):
                write_comparison(comparison_root, rows)


if __name__ == "__main__":
    unittest.main()
