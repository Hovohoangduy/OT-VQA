"""Tests for paired no-OT, Balanced-OT, and UOT benchmark comparisons."""

import unittest

from scripts.summarize_fusion_benchmark import paired_comparisons


class BenchmarkSummaryTests(unittest.TestCase):
    def test_all_transport_pairs_are_compared_by_method_and_seed(self):
        rows = [
            {
                "method": "ban", "transport": transport, "seed": 1105,
                "best_val_f1": f1, "best_val_loss": loss,
            }
            for transport, f1, loss in (
                ("none", 0.20, 2.5),
                ("balanced", 0.25, 2.3),
                ("uot", 0.30, 2.2),
            )
        ]
        pairs, aggregate = paired_comparisons(rows)
        self.assertEqual(len(pairs), 3)
        by_name = {row["comparison"]: row for row in pairs}
        self.assertAlmostEqual(
            by_name["balanced_vs_none"]["candidate_minus_baseline_f1"], 0.05
        )
        self.assertAlmostEqual(
            by_name["uot_vs_none"]["candidate_minus_baseline_f1"], 0.10
        )
        self.assertAlmostEqual(
            by_name["uot_vs_balanced"]["candidate_minus_baseline_f1"], 0.05
        )
        self.assertEqual(len(aggregate), 3)
        self.assertTrue(all(row["candidate_wins"] == 1 for row in aggregate))

    def test_missing_transport_only_skips_unavailable_pairs(self):
        rows = [
            {
                "method": "mutan", "transport": "balanced", "seed": 7,
                "best_val_f1": 0.4, "best_val_loss": 2.0,
            },
            {
                "method": "mutan", "transport": "uot", "seed": 7,
                "best_val_f1": 0.35, "best_val_loss": 2.1,
            },
        ]
        pairs, aggregate = paired_comparisons(rows)
        self.assertEqual([row["comparison"] for row in pairs], ["uot_vs_balanced"])
        self.assertEqual(aggregate[0]["candidate_losses"], 1)


if __name__ == "__main__":
    unittest.main()
