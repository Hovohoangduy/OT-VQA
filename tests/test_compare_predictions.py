import tempfile
import unittest
from pathlib import Path

import pandas as pd

from utils.compare_predictions import compare_runs
from utils.metrics import PAPER_METRICS


class ComparisonTests(unittest.TestCase):
    def test_paired_comparison_and_question_types(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shared = {"anno_id": ["a", "b", "c"], "question": ["q1", "q2", "q3"],
                      "reference": ["red", "blue", "green"],
                      "question_type": ["color", "color", "other"]}
            baseline = root / "san.csv"
            ot = root / "ot.csv"
            pd.DataFrame({**shared, **{metric: [0, 0, 1] for metric in PAPER_METRICS}}).to_csv(baseline, index=False)
            pd.DataFrame({**shared, **{metric: [1, 0, 1] for metric in PAPER_METRICS}}).to_csv(ot, index=False)
            result = compare_runs([baseline], [ot], bootstrap_samples=100, seed=1)
            self.assertEqual(result["examples"], 3)
            self.assertAlmostEqual(result["em"]["mean_delta"], 1 / 3)
            for metric in PAPER_METRICS:
                self.assertAlmostEqual(result[metric]["mean_delta"], 1 / 3)
            self.assertAlmostEqual(result["question_type_deltas"]["color"]["em"], 0.5)
            self.assertEqual(set(PAPER_METRICS),
                             set(result["question_type_deltas"]["color"]) - {"examples"})
            altered = pd.read_csv(ot)
            altered.loc[0, "reference"] = "wrong"
            altered.to_csv(ot, index=False)
            with self.assertRaisesRegex(ValueError, "disagree"):
                compare_runs([baseline], [ot], bootstrap_samples=10)


if __name__ == "__main__":
    unittest.main()
