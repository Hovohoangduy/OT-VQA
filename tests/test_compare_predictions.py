import tempfile
import unittest
from pathlib import Path

import pandas as pd

from utils.compare_predictions import compare_runs


class ComparisonTests(unittest.TestCase):
    def test_paired_comparison_and_question_types(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shared = {"anno_id": ["a", "b", "c"], "question": ["q1", "q2", "q3"],
                      "reference": ["red", "blue", "green"],
                      "question_type": ["color", "color", "other"]}
            baseline = root / "san.csv"
            ot = root / "ot.csv"
            pd.DataFrame({**shared, "em": [0, 0, 1], "f1": [0, 0, 1]}).to_csv(baseline, index=False)
            pd.DataFrame({**shared, "em": [1, 0, 1], "f1": [1, 0, 1]}).to_csv(ot, index=False)
            result = compare_runs([baseline], [ot], bootstrap_samples=100, seed=1)
            self.assertEqual(result["examples"], 3)
            self.assertAlmostEqual(result["em"]["mean_delta"], 1 / 3)
            self.assertAlmostEqual(result["question_type_em_delta"]["color"]["delta"], 0.5)
            altered = pd.read_csv(ot)
            altered.loc[0, "reference"] = "wrong"
            altered.to_csv(ot, index=False)
            with self.assertRaisesRegex(ValueError, "disagree"):
                compare_runs([baseline], [ot], bootstrap_samples=10)


if __name__ == "__main__":
    unittest.main()
