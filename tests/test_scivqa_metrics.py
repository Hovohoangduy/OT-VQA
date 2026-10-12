"""Offline SciVQA scoring, config, training, and evaluation integration checks."""

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
from utils.metrics import (SCIVQA_METRICS, mean_scores, metrics_for_dataset,
                           resolve_dataset, scivqa_score_pairs)
from utils.vqa_dataset import VQADataset


class FixedScorer:
    hash = "offline-scivqa-scorer"

    def score(self, candidates, references):
        self.candidates, self.references = candidates, references
        # Different P/R/F1 values catch swapped components or F1 recomputation.
        return (torch.full((len(candidates),), 0.8),
                torch.full((len(candidates),), 0.5),
                torch.full((len(candidates),), 0.6))


class ChoiceModel(nn.Module):
    pad_token_id = 0
    fusion = "ot"
    model_config = {"fusion": "ot"}

    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(1, 1, 2))

    def forward(self, images, questions, answers, anno_ids):
        return (self.logits.expand(len(answers), 1, 2),
                torch.ones(len(answers), 1, dtype=torch.long, device=images.device))

    def evaluate_batch(self, images, questions, answers, anno_ids):
        logits, targets = self(images, questions, answers, anno_ids)
        return logits, targets, ["A,C"] * len(answers)

    def answers_from_ids(self, ids):
        return ids


class SciVQAMetricTests(unittest.TestCase):
    def test_nine_components_and_macro_averaging(self):
        scorer = FixedScorer()
        rows = scivqa_score_pairs(["red red leaf", "red blue"],
                                  ["red leaf", "blue red"], scorer)
        self.assertEqual(set(rows[0]), set(SCIVQA_METRICS))
        self.assertEqual(rows[0]["rouge_1_precision"], 1)
        self.assertAlmostEqual(rows[0]["rouge_1_recall"], 2 / 3)
        self.assertAlmostEqual(rows[0]["rouge_1_f1"], 0.8)
        self.assertEqual(rows[1]["rouge_1_f1"], 1)
        self.assertEqual(rows[1]["rouge_l_f1"], 0.5)
        self.assertAlmostEqual(rows[0]["bertscore_precision"], 0.8)
        self.assertAlmostEqual(rows[0]["bertscore_recall"], 0.5)
        self.assertAlmostEqual(rows[0]["bertscore_f1"], 0.6)
        scores = mean_scores(rows, SCIVQA_METRICS)
        self.assertAlmostEqual(scores["rouge_1_f1"], 0.9)
        self.assertAlmostEqual(scores["rouge_l_f1"], 0.65)
        self.assertEqual(scorer.candidates, ["red leaf", "blue red"])
        self.assertEqual(scorer.references, ["red red leaf", "red blue"])

    def test_case_punctuation_numbers_labels_and_empty_answers(self):
        scorer = FixedScorer()
        rows = scivqa_score_pairs(["A,C", "0.66", "", "leaf", ""],
                                  ["a, c", "0.66", "", "", "leaf"], scorer)
        self.assertEqual([row["rouge_1_f1"] for row in rows], [1, 1, 0, 0, 0])
        self.assertEqual(scorer.references, ["A,C", "0.66"])
        self.assertEqual(scorer.candidates, ["a, c", "0.66"])
        for component in ("precision", "recall", "f1"):
            self.assertEqual([row[f"bertscore_{component}"] for row in rows[2:]], [1, 0, 0])
        with self.assertRaises(ValueError):
            scivqa_score_pairs(["yes"], [], scorer)
        with self.assertRaises(ValueError):
            scivqa_score_pairs(["yes"], ["yes"], None)
        with self.assertRaisesRegex(RuntimeError, "wrong number"):
            with patch.object(scorer, "score", return_value=([], [], [])):
                scivqa_score_pairs(["yes"], ["yes"], scorer)

    def test_dataset_and_scorer_defaults_use_the_relevant_split(self):
        for path in ("data/scivqa_dataset/train.csv", "/custom/SciVQA/test.csv",
                     "scivqa-val.csv"):
            self.assertEqual(resolve_dataset("auto", path), "scivqa")
        self.assertEqual(resolve_dataset("auto", "data/notscivqa.csv"), "plantexpert")
        self.assertEqual(resolve_dataset("scivqa", "gqa/train.csv"), "scivqa")
        self.assertEqual(metrics_for_dataset("scivqa"), SCIVQA_METRICS)
        args = get_args(["--train_csv_path", "scivqa/train.csv"])
        self.assertEqual(args.bertscore_model, "roberta-large")
        self.assertFalse(args.bertscore_rescale)
        for split in ("dev", "test"):
            args = get_args(["--split", split, f"--{split}_csv_path", "scivqa/data.csv"],
                            evaluation=True)
            self.assertEqual(args.bertscore_model, "roberta-large")
            self.assertFalse(args.bertscore_rescale)
        self.assertEqual(get_args([]).bertscore_model, "bert-base-uncased")
        self.assertTrue(get_args([]).bertscore_rescale)

    def test_python_defaults_and_explicit_overrides(self):
        args = get_args(["--dataset", "scivqa"])
        self.assertEqual(args.dataset, "scivqa")
        self.assertEqual(args.train_csv_path, "data/scivqa_dataset/train.csv")
        self.assertEqual((args.max_question_tokens, args.max_answer_tokens), (256, 64))
        self.assertEqual(args.fusion, "ot")
        self.assertFalse(args.bertscore_rescale)
        args = get_args(["--batch_size", "2", "--dataset", "scivqa",
                         "--epochs", "1", "--bertscore_rescale", "--bertscore_model", "bert-base-uncased"])
        self.assertEqual((args.batch_size, args.epochs), (2, 1))
        self.assertTrue(args.bertscore_rescale)
        self.assertEqual(args.bertscore_model, "bert-base-uncased")
    def _write_data(self, root):
        Image.new("RGB", (8, 8), "red").save(root / "figure.png")
        frame = pd.DataFrame({"anno_id": ["qa1", "qa2", "qa3"],
                              "image": ["figure.png"] * 3,
                              "question": ["Which curves? Answer options: A: Red; C: Green"] * 3,
                              "answer": ["A,C", "A", "B"],
                              "qa_pair_type": ["non-binary visual"] * 3,
                              "figure_type": ["line chart"] * 3})
        csv_path = root / "scivqa_val.csv"
        frame.to_csv(csv_path, index=False)
        return frame, csv_path

    def test_evaluation_partial_batch_predictions_and_default_scorer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame, _ = self._write_data(root)
            loader = DataLoader(VQADataset(frame, Config.transforms, root, dataset_name="scivqa"),
                                batch_size=2)
            predictions = []
            with patch("test.build_bertscore_scorer", return_value=FixedScorer()) as builder, \
                 patch("test.score_pairs") as plant_scorer:
                result = evaluation(ChoiceModel(), loader, nn.CrossEntropyLoss(),
                                    predictions=predictions, measure_performance=True)
            builder.assert_called_once_with(model_type="roberta-large", rescale_with_baseline=False)
            plant_scorer.assert_not_called()
            self.assertEqual(result["examples"], 3)
            self.assertEqual(len(predictions), 3)
            self.assertEqual(predictions[-1]["anno_id"], "qa3")
            self.assertEqual(predictions[0]["reference"], "A,C")
            self.assertEqual(set(result["metrics"]), set(SCIVQA_METRICS))
            self.assertAlmostEqual(result["metrics"]["rouge_1_f1"], (1 + 2 / 3) / 3)
            self.assertGreater(result["loss"], 0)

    def test_one_epoch_training_and_cli_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, csv_path = self._write_data(root)
            destination = root / "run"
            args = get_args(["--dataset", "scivqa", "--train_csv_path", str(csv_path),
                             "--dev_csv_path", str(csv_path), "--img_path", str(root),
                             "--model_path", str(destination), "--epochs", "1",
                             "--batch_size", "2", "--device", "cpu"])
            with patch("train.VQAModel", return_value=ChoiceModel()) as model_builder, \
                 patch("train.save_checkpoint") as save, \
                 patch("train.build_bertscore_scorer", return_value=FixedScorer()) as scorer_builder, \
                 redirect_stdout(io.StringIO()):
                _run_training(args, torch.device("cpu"), rank=0, world_size=1)
            self.assertEqual(save.call_count, 2)
            self.assertEqual(model_builder.call_args.kwargs["max_question_tokens"], 256)
            self.assertEqual(model_builder.call_args.kwargs["max_answer_tokens"], 64)
            self.assertFalse(scorer_builder.call_args.kwargs["rescale_with_baseline"])
            history = [json.loads(line) for line in (destination / "metrics.jsonl").read_text().splitlines()]
            for name in SCIVQA_METRICS:
                self.assertIn(f"train_{name}", history[0])
                self.assertIn(f"val_{name}", history[0])
            self.assertNotIn("val_em", history[0])
            manifest = json.loads((destination / "run_config.json").read_text())
            self.assertEqual(manifest["dataset"], "scivqa")
            self.assertEqual(manifest["generated_metrics"], list(SCIVQA_METRICS))
            self.assertEqual(manifest["bertscore_hash"], FixedScorer.hash)
            self.assertFalse(manifest["rouge"]["use_stemmer"])
            self.assertTrue((destination / "evaluation_metrics_plot.png").is_file())
            self.assertEqual(set(_history_report(history)["best_generated_metrics"]), set(SCIVQA_METRICS))

            for split in ("dev", "test"):
                report_path = destination / f"{split}_report.json"
                predictions_path = destination / f"{split}_predictions.csv"
                eval_args = get_args(["--dataset", "scivqa", "--split", split,
                                      f"--{split}_csv_path", str(csv_path), "--img_path", str(root),
                                      "--model_path", str(destination), "--device", "cpu",
                                      "--batch_size", "2", "--report_json", str(report_path),
                                      "--predictions_csv", str(predictions_path)], evaluation=True)
                with patch("test.get_args", return_value=eval_args), \
                     patch("test.load_model", return_value=ChoiceModel()), \
                     patch("test.build_bertscore_scorer", return_value=FixedScorer()), \
                     redirect_stdout(io.StringIO()):
                    evaluate_main()
                report = json.loads(report_path.read_text())
                self.assertEqual(report["dataset"], "scivqa")
                self.assertEqual(set(report["generated_metrics"]), set(SCIVQA_METRICS))
                self.assertEqual(report["bertscore"]["model"], "roberta-large")
                self.assertFalse(report["bertscore"]["rescale_with_baseline"])
                predictions = pd.read_csv(predictions_path)
                self.assertEqual(len(predictions), 3)
                self.assertTrue(set(SCIVQA_METRICS).issubset(predictions.columns))
                self.assertIn("qa_pair_type", predictions)
                self.assertIn("figure_type", predictions)

            comparison = root / "comparison"
            run = comparison / "ot_seed1"
            run.mkdir(parents=True)
            (run / "run_config.json").write_text(json.dumps(manifest))
            (run / "test_report.json").write_text(json.dumps(report))
            rows = collect_results(comparison, ["ot", "san"], [1], "test")
            write_comparison(comparison, rows)
            self.assertIn("mean_rouge_1_f1", (comparison / "summary.csv").read_text())
            self.assertIn("ROUGE-1 F1", (comparison / "comparison.md").read_text())
            self.assertIn("bertscore_precision", (comparison / "comparison.csv").read_text())


if __name__ == "__main__":
    unittest.main()
