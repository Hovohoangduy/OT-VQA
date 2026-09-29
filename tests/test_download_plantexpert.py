"""Regression checks for the PlantExpertVQA downloader."""

import csv
import io
import json
import sys
import tempfile
import unittest
import urllib.error
import zipfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

from utils.download_plantexpert import (
    ImageEntry, archive_size, choose_rows, main, parse_args, read_image, request, retry_delay,
)


class ArchiveSizeTests(unittest.TestCase):
    def test_head_does_not_read_archive_body(self):
        response = MagicMock()
        response.__enter__.return_value = response
        response.headers = {"Content-Length": "4320650663"}
        response.read.side_effect = AssertionError("HEAD response body was read")

        with patch("utils.download_plantexpert.urllib.request.urlopen", return_value=response) as open_url:
            self.assertEqual(archive_size("https://example.com/archive.zip"), 4320650663)

        self.assertEqual(open_url.call_args.args[0].get_method(), "HEAD")
        response.read.assert_not_called()

    def test_image_member_uses_one_range_request(self):
        image_buffer = io.BytesIO()
        Image.new("RGB", (2, 2), "green").save(image_buffer, format="JPEG")
        image_bytes = image_buffer.getvalue()
        archive_buffer = io.BytesIO()
        with zipfile.ZipFile(archive_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("images/leaf.jpg", image_bytes)
        archive_bytes = archive_buffer.getvalue()
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            info = archive.getinfo("images/leaf.jpg")
        entry = ImageEntry(
            "https://example.com/archive.zip", len(archive_bytes), info.header_offset,
            len(info.filename.encode()), info.compress_size, info.file_size,
            info.compress_type, info.flag_bits, info.CRC,
        )

        def fake_request(url, *, start, end):
            return archive_bytes[start:end + 1], {}

        with patch("utils.download_plantexpert.request", side_effect=fake_request) as fetch:
            self.assertEqual(read_image(entry), image_bytes)
        fetch.assert_called_once()


class RateLimitTests(unittest.TestCase):
    def test_rate_limit_reset_header(self):
        self.assertEqual(retry_delay({"RateLimit": '"api";r=0;t=41'}, 0), 42.0)

    def test_429_waits_for_reset_and_retries(self):
        limited = urllib.error.HTTPError(
            "https://datasets-server.huggingface.co/rows", 429,
            "Too Many Requests", {"RateLimit": '"api";r=0;t=41'}, None,
        )
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = b'{"rows": []}'
        with patch("utils.download_plantexpert.urllib.request.urlopen",
                   side_effect=[limited, response]) as open_url, \
             patch("utils.download_plantexpert.time.sleep") as sleep:
            data, _ = request("https://datasets-server.huggingface.co/rows")
        self.assertEqual(data, b'{"rows": []}')
        self.assertEqual(open_url.call_count, 2)
        sleep.assert_called_once_with(42.0)


class SelectionCacheTests(unittest.TestCase):
    def test_csv_selection_is_reused_after_restart(self):
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=(
            "qa_id", "image_path", "question_text", "answer", "question_category",
            "crop", "disease",
        ))
        writer.writeheader()
        for index in range(400):
            writer.writerow({
                "qa_id": str(index), "image_path": f"images/{index // 5}.jpg",
                "question_text": f"Question {index}?", "answer": "Answer",
            })
        data = output.getvalue().encode()
        image_index = {f"{index}.jpg": None for index in range(80)}

        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            with patch("utils.download_plantexpert.urllib.request.urlopen",
                       side_effect=lambda *args, **kwargs: io.BytesIO(data)) as open_url:
                first = choose_rows("train", 5, 42, image_index, cache)
                second = choose_rows("train", 5, 42, image_index, cache)
            self.assertEqual(first, second)
            self.assertEqual(len(first), 5)
            self.assertEqual(open_url.call_count, 1)
            self.assertTrue((cache / "train_5_seed42_rows.json").is_file())

    def test_test_split_samples_rows_from_test_csv(self):
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=(
            "qa_id", "image_path", "question_text", "answer",
        ))
        writer.writeheader()
        for index in range(400):
            writer.writerow({
                "qa_id": str(index), "image_path": f"images/{index}.jpg",
                "question_text": f"Question {index}?", "answer": "Answer",
            })
        data = output.getvalue().encode()
        image_index = {f"{index}.jpg": None for index in range(400)}

        with tempfile.TemporaryDirectory() as directory:
            with patch("utils.download_plantexpert.urllib.request.urlopen",
                       side_effect=lambda *args, **kwargs: io.BytesIO(data)) as open_url:
                rows = choose_rows("test", 20, 42, image_index, Path(directory))
        self.assertEqual(len(rows), 20)
        self.assertEqual(len({row["anno_id"] for row in rows}), 20)
        self.assertTrue(any(int(row["anno_id"]) >= 200 for row in rows))
        self.assertTrue(open_url.call_args.args[0].full_url.endswith("/data/test.csv"))

    def test_default_pair_counts(self):
        with patch.object(sys, "argv", ["download_plantexpert"]):
            args = parse_args()
        self.assertEqual((args.train_pairs, args.val_pairs, args.test_pairs),
                         (10000, 1000, 2000))

    def test_main_writes_all_three_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            args = Namespace(output=output, train_pairs=1, val_pairs=1,
                             test_pairs=1, seed=42, workers=1)
            rows = [[{"anno_id": split, "image": f"{split}.jpg",
                      "question": "Question?", "answer": "Answer",
                      "question_category": "", "crop": "", "disease": ""}]
                    for split in ("train", "validation", "test")]
            with patch("utils.download_plantexpert.parse_args", return_value=args), \
                 patch("utils.download_plantexpert.index_images", return_value={
                     f"{split}.jpg": None for split in ("train", "validation", "test")
                 }), \
                 patch("utils.download_plantexpert.choose_rows", side_effect=rows) as choose, \
                 patch("utils.download_plantexpert.save_image"):
                main()
            self.assertEqual([call.args[0] for call in choose.call_args_list],
                             ["train", "validation", "test"])
            for split, filename in (("train", "train.csv"), ("validation", "val.csv"),
                                    ("test", "test.csv")):
                with (output / filename).open(newline="", encoding="utf-8") as handle:
                    self.assertEqual(list(csv.DictReader(handle))[0]["anno_id"], split)
            metadata = json.loads((output / "metadata.json").read_text())
            self.assertEqual(metadata["test_pairs"], 1)


if __name__ == "__main__":
    unittest.main()
