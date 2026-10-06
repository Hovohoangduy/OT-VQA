"""Offline regression checks for random GQA subset selection."""

import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

from utils.download_gqa import PAGE_SIZE, fetch_image_records, main, parse_args


class GQASamplingTests(unittest.TestCase):
    @staticmethod
    def image_page(url, cache_dir=None):
        query = parse_qs(urlsplit(url).query)
        offset = int(query["offset"][0])
        length = int(query["length"][0])
        total = 253  # Include a partial final page.
        return {
            "num_rows_total": total,
            "rows": [
                {"row": {"id": str(index), "image": {"src": f"image/{index}"}}}
                for index in range(offset, min(offset + length, total))
            ],
        }

    def test_samples_full_split_without_replacement_and_reuses_pages(self):
        with patch("utils.download_gqa.fetch_json", side_effect=self.image_page) as fetch:
            records = fetch_image_records("train", 45, workers=3, seed=17)
        expected = random.Random(17).sample(range(253), 45)
        self.assertEqual([int(record["id"]) for record in records], expected)
        self.assertEqual(len({record["id"] for record in records}), 45)
        self.assertTrue(any(index >= 200 for index in expected))
        self.assertEqual(records[0]["url"], f"image/{expected[0]}")
        offsets = [
            int(parse_qs(urlsplit(call.args[0]).query)["offset"][0])
            for call in fetch.call_args_list
        ]
        self.assertEqual(len(offsets), len(set(offsets)))
        self.assertEqual(set(offsets), {0} | {index // PAGE_SIZE * PAGE_SIZE for index in expected})

    def test_seed_reproduces_selection_across_worker_counts(self):
        with patch("utils.download_gqa.fetch_json", side_effect=self.image_page):
            first = fetch_image_records("train", 30, workers=1, seed=42)
            repeated = fetch_image_records("train", 30, workers=4, seed=42)
            different = fetch_image_records("train", 30, workers=4, seed=43)
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, different)

    def test_rejects_unavailable_count_before_fetching_more_pages(self):
        with patch("utils.download_gqa.fetch_json", side_effect=self.image_page) as fetch:
            with self.assertRaisesRegex(ValueError, "only 253 are available"):
                fetch_image_records("val", 254, workers=2)
        fetch.assert_called_once()

    def test_rejects_nonpositive_counts_and_workers(self):
        with patch("utils.download_gqa.fetch_json") as fetch:
            for count, workers in [(0, 1), (-1, 1), (1, 0), (1, -1)]:
                with self.subTest(count=count, workers=workers):
                    with self.assertRaises(ValueError):
                        fetch_image_records("train", count, workers)
        fetch.assert_not_called()

    def test_default_sizes_and_seed(self):
        with patch("sys.argv", ["download_gqa"]):
            args = parse_args()
        self.assertEqual((args.train_images, args.test_images, args.val_images), (10000, 2000, 1000))
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.metadata_workers, 1)
        self.assertEqual(args.request_interval, 1.0)

    def test_main_partitions_held_out_images_and_records_seed(self):
        def records(split, count, workers, seed, cache_dir):
            return [{"id": f"{split}-{index}"} for index in range(count)]

        def annotations(split, image_ids, workers, cache_dir):
            return {image_id: {"question": "color?", "answer": "red"} for image_id in image_ids}

        def materialize(name, selected, answers, output_dir, workers):
            return [{"question": "color?", "answer": "red"} for _ in selected]

        with tempfile.TemporaryDirectory() as directory:
            with patch("sys.argv", ["download_gqa", "--output", directory,
                                    "--train-images", "7", "--val-images", "3",
                                    "--test-images", "5", "--seed", "99"]), \
                 patch("utils.download_gqa.fetch_image_records", side_effect=records) as fetch, \
                 patch("utils.download_gqa.fetch_first_questions", side_effect=annotations), \
                 patch("utils.download_gqa.materialize_split", side_effect=materialize) as write, \
                 patch("utils.download_gqa.fetch_json", return_value={"sha": "revision"}):
                main()
            metadata = json.loads((Path(directory) / "metadata.json").read_text())
        self.assertEqual(fetch.call_args_list[0].args, ("train", 7, 1, 99, Path(directory) / ".row_cache"))
        self.assertEqual(fetch.call_args_list[1].args, ("val", 8, 1, 99, Path(directory) / ".row_cache"))
        selected = {call.args[0]: {record["id"] for record in call.args[1]}
                    for call in write.call_args_list}
        self.assertEqual(len(selected["val"]), 3)
        self.assertEqual(len(selected["test"]), 5)
        self.assertFalse(selected["val"] & selected["test"])
        self.assertFalse(selected["train"] & (selected["val"] | selected["test"]))
        self.assertEqual(metadata["seed"], 99)
        self.assertEqual(metadata["counts"]["train_rows"], 7)
        self.assertEqual(metadata["counts"]["val_rows"], 3)
        self.assertEqual(metadata["counts"]["test_rows"], 5)


if __name__ == "__main__":
    unittest.main()
