"""Comparison output must reflect reports and reject mismatched data."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.compare_fusions import collect_results, write_comparison
from utils.metrics import PAPER_METRICS


class ComparisonTests(unittest.TestCase):
    def test_report_collection_and_dataset_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for method in ("ot", "san"):
                run = root / f"{method}_seed1"
                run.mkdir()
                (run / "run_config.json").write_text(json.dumps({
                    "train_examples": 3, "dev_examples": 2,
                    "train_csv_sha256": "train", "dev_csv_sha256": "dev"}))
                (run / "test_report.json").write_text(json.dumps({
                    "fusion": method, "split": "test", "checkpoint": str(run / "best.pt"),
                    "evaluation_csv_sha256": "test", "loss": 1.0,
                    "trainable_parameters": 123,
                    "generated_metrics": {name: 0.5 for name in PAPER_METRICS},
                    "performance": {"examples": 2, "examples_per_second": 4.0,
                                    "milliseconds_per_example": 250.0,
                                    "peak_cuda_bytes": None}}))
            rows = collect_results(root, ["ot", "san", "ban"], [1], "test")
            write_comparison(root, rows)
            self.assertIn("ban,1,incomplete", (root / "comparison.csv").read_text())
            self.assertIn("| ot | 1 | complete", (root / "comparison.md").read_text())
            rows[1]["evaluation_csv_sha256"] = "different"
            with self.assertRaisesRegex(ValueError, "evaluation_csv_sha256"):
                write_comparison(root, rows)


if __name__ == "__main__":
    unittest.main()
