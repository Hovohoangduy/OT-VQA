"""Checks for SciVQA choice context, split integrity, and download/export."""

import csv
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

from utils.download_scivqa import (
    SOURCE_FILES, download_file, extract_images, format_rows, main, parse_args,
    select_rows, validate_splits,
)


def record(identifier="qa1", image="figure.png"):
    return {"instance_id": identifier, "image_file": image,
            "question": "Which curve?", "answer": "A,C",
            "answer_options": [{"A": "Red", "B": None}, {"B": "Blue"}, {"C": "Green"}],
            "caption": "A scientific figure.", "figure_type": "line chart"}


def png():
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), "red").save(buffer, format="PNG")
    return buffer.getvalue()


class SciVQATests(unittest.TestCase):
    def test_choices_and_caption_preserve_source_answer(self):
        row = format_rows([record()], include_caption=True)[0]
        self.assertEqual(row["answer"], "A,C")
        self.assertEqual(row["original_question"], "Which curve?")
        self.assertIn("Answer options: A: Red; B: Blue; C: Green", row["question"])
        self.assertIn("Figure caption: A scientific figure.", row["question"])
        self.assertEqual(json.loads(row["answer_options"]), record()["answer_options"])
        self.assertNotIn("Figure caption:", format_rows([record()])[0]["question"])

    def test_invalid_records_fail_before_export(self):
        for field in ("instance_id", "image_file", "question", "answer"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                format_rows([{**record(), field: None}])
        for name in ("../figure.png", "/figure.png", "..\\figure.png"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                format_rows([record(image=name)])
        with self.assertRaisesRegex(ValueError, "Duplicate instance_id"):
            format_rows([record(), record()])

    def test_sampling_and_official_split_overlap(self):
        rows = format_rows([record(str(i), f"figure{i}.png") for i in range(30)])
        self.assertEqual(select_rows(rows, 0, 42), rows)
        self.assertEqual(select_rows(rows, 5, 42), select_rows(rows, 5, 42))
        self.assertNotEqual(select_rows(rows, 5, 42), select_rows(rows, 5, 43))
        with self.assertRaises(ValueError):
            select_rows(rows, 31, 42)
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_splits({"train": [rows[0]], "test": [rows[0]]})

    def test_extraction_flattens_members_and_reuses_verified_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "images.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("../../figure.png", png())
                archive.writestr("unused.png", png())
            extract_images(archive_path, {"figure.png"}, root / "images")
            image_path = root / "images/figure.png"
            before = image_path.stat().st_mtime_ns
            extract_images(archive_path, {"figure.png"}, root / "images")
            self.assertEqual(image_path.stat().st_mtime_ns, before)
            self.assertFalse((root / "images/unused.png").exists())
            with self.assertRaisesRegex(ValueError, "missing"):
                extract_images(archive_path, {"missing.png"}, root / "images")

    def test_corrupt_image_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "images.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("figure.png", b"not an image")
            with self.assertRaisesRegex(ValueError, "Invalid image"):
                extract_images(archive_path, {"figure.png"}, root / "images")
            self.assertFalse((root / "images/figure.png").exists())

    def test_short_download_retries_and_cache_is_reused(self):
        def response(data):
            result = MagicMock()
            result.__enter__.return_value = result
            result.headers = {"Content-Length": "4"}
            result.read.side_effect = [data, b""]
            return result

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.json"
            with patch("utils.download_scivqa.urllib.request.urlopen",
                       side_effect=[response(b"ab"), response(b"abcd")]) as fetch, \
                 patch("utils.download_scivqa.time.sleep"):
                download_file("https://example.com/source.json", path)
                download_file("https://example.com/source.json", path)
            self.assertEqual(fetch.call_count, 2)
            self.assertEqual(path.read_bytes(), b"abcd")
            self.assertFalse(path.with_suffix(".json.part").exists())

    def test_main_exports_three_loader_compatible_splits(self):
        def fake_download(url, path):
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".json":
                split = next(key for key, name in SOURCE_FILES.items() if name == path.name)
                path.write_text(json.dumps([record(split, f"{split}.png")]))
            else:
                split = path.stem.removeprefix("images_")
                with zipfile.ZipFile(path, "w") as archive:
                    archive.writestr(f"images_{split}/{split}.png", png())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = parse_args(["--output", str(root)])
            with patch("utils.download_scivqa.parse_args", return_value=args), \
                 patch("utils.download_scivqa.download_file", side_effect=fake_download):
                main()
            for split, name in (("train", "train"), ("validation", "val"), ("test", "test")):
                with (root / f"{name}.csv").open(newline="", encoding="utf-8") as handle:
                    row = next(csv.DictReader(handle))
                self.assertEqual(row["anno_id"], split)
                self.assertTrue((root / "images" / row["image"]).is_file())
                self.assertIn("A: Red", row["question"])
            self.assertEqual(json.loads((root / "metadata.json").read_text())["unique_images"], 3)


if __name__ == "__main__":
    unittest.main()
