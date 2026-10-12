"""Download SciVQA and export CSVs compatible with this project's VQADataset.

Run ``python -m utils.download_scivqa`` for all official labeled splits.
Only the standard library and Pillow are required. Source JSONs and ZIPs are
cached; images are verified and reused when the command is rerun.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import http.client
import json
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from email.utils import parsedate_to_datetime
from pathlib import Path, PurePosixPath

from PIL import Image


DATASET_ID = "katebor/SciVQA"
SOURCE_FILES = {
    "train": "train_2025-07-03_09-06.json",
    "validation": "validation_2025-07-03_09-06.json",
    "test": "test_2025-07-03_09-01.json",
}
CSV_FIELDS = (
    "anno_id", "image", "question", "answer", "original_question",
    "answer_options", "figure_id", "caption", "figure_type", "compound",
    "figs_numb", "qa_pair_type", "venue", "categories", "source_dataset",
    "paper_id", "pdf_url",
)


def retry_delay(headers, attempt: int) -> float:
    value = (headers or {}).get("Retry-After", "")
    try:
        return max(0, float(value))
    except ValueError:
        try:
            return max(0, parsedate_to_datetime(value).timestamp() - time.time())
        except (TypeError, ValueError, OverflowError):
            return min(2 ** attempt, 60)


def download_file(url: str, path: Path) -> None:
    """Stream into an atomic cache file, retrying interrupted downloads."""
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    headers = {"User-Agent": "OT-VQA-SciVQA-downloader/1.0"}
    if os.environ.get("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    for attempt in range(6):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=180) as response:
                expected = response.headers.get("Content-Length")
                received = 0
                with temporary.open("wb") as handle:
                    while chunk := response.read(1024 * 1024):
                        handle.write(chunk)
                        received += len(chunk)
                if expected is not None and received != int(expected):
                    raise http.client.IncompleteRead(b"", int(expected) - received)
            os.replace(temporary, path)
            return
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 5:
                raise
            delay = retry_delay(exc.headers, attempt)
        except (urllib.error.URLError, TimeoutError, ConnectionError,
                http.client.IncompleteRead, http.client.RemoteDisconnected):
            if attempt == 5:
                raise
            delay = min(2 ** attempt, 60)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"Download interrupted; retrying in {delay:.0f}s: {path.name}", flush=True)
        # Short sleeps keep long server-requested cooldowns interruptible.
        while delay > 0:
            interval = min(delay, 60)
            time.sleep(interval)
            delay -= interval


def format_rows(records: list[dict], *, include_caption: bool = False) -> list[dict]:
    """Keep source answer labels and make their choices visible to the model."""
    rows = []
    seen = set()
    for record in records:
        for field in ("instance_id", "image_file", "question", "answer"):
            if not isinstance(record.get(field), str) or not record[field].strip():
                raise ValueError(f"Missing or empty {field} in SciVQA record")
        identifier = record["instance_id"]
        if identifier in seen:
            raise ValueError(f"Duplicate instance_id: {identifier}")
        seen.add(identifier)
        image_name = record["image_file"]
        if (image_name in (".", "..") or "/" in image_name or "\\" in image_name
                or Path(image_name).is_absolute()):
            raise ValueError(f"Unsafe image filename: {image_name}")
        options = record.get("answer_options") or []
        if not isinstance(options, list):
            raise ValueError(f"Expected a list of answer options for {identifier}")
        choices = {}
        for option in options:
            if not isinstance(option, dict):
                raise ValueError(f"Expected labeled answer options for {identifier}")
            for label, text in option.items():
                if text is not None:
                    if label in choices:
                        raise ValueError(f"Duplicate answer option {label} for {identifier}")
                    choices[label] = str(text)
        question = record["question"].strip()
        if choices:
            question += "\nAnswer options: " + "; ".join(
                f"{label}: {choices[label]}" for label in sorted(choices)
            )
        if include_caption and record.get("caption"):
            question += f"\nFigure caption: {record['caption']}"
        row = {field: record.get(field, "") for field in CSV_FIELDS}
        row.update(anno_id=identifier, image=image_name, question=question,
                   answer=record["answer"].strip(), original_question=record["question"],
                   answer_options=json.dumps(options, ensure_ascii=False))
        rows.append(row)
    return rows


def select_rows(rows: list[dict], count: int, seed: int) -> list[dict]:
    if count < 0 or count > len(rows):
        raise ValueError(f"Requested {count} pairs; available: {len(rows)} (0 means all)")
    return random.Random(seed).sample(rows, count) if count else rows


def validate_splits(splits: dict[str, list[dict]]) -> None:
    images, identifiers = set(), set()
    for split, rows in splits.items():
        split_images = {row["image"] for row in rows}
        split_ids = {row["anno_id"] for row in rows}
        if images & split_images or identifiers & split_ids:
            raise ValueError(f"Images or annotation IDs overlap with another split: {split}")
        images.update(split_images)
        identifiers.update(split_ids)


def valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, ValueError, SyntaxError):
        return False


def extract_images(archive_path: Path, names: set[str], folder: Path) -> None:
    """Copy only referenced PNGs, without trusting ZIP extraction paths."""
    folder.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        members = {}
        for info in archive.infolist():
            name = PurePosixPath(info.filename).name
            if info.is_dir() or "__MACOSX" in PurePosixPath(info.filename).parts:
                continue
            if name in names:
                if name in members:
                    raise ValueError(f"Ambiguous image in ZIP: {name}")
                members[name] = info
        missing = names - members.keys()
        if missing:
            raise ValueError(f"Images missing from {archive_path.name}: {sorted(missing)[:5]}")
        for name in sorted(names):
            path = folder / name
            if valid_image(path):
                continue
            temporary = path.with_suffix(path.suffix + ".part")
            try:
                temporary.write_bytes(archive.read(members[name]))  # ZIP CRC verified on read.
                if not valid_image(temporary):
                    raise ValueError(f"Invalid image in ZIP: {name}")
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)


def write_csv(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(".csv.part")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and format the official labeled SciVQA splits")
    parser.add_argument("--output", type=Path, default=Path("data/scivqa_dataset"))
    parser.add_argument("--revision", default="main", help="Hugging Face branch, tag, or commit")
    parser.add_argument("--train-pairs", type=int, default=0, help="0 downloads all training pairs")
    parser.add_argument("--val-pairs", type=int, default=0, help="0 downloads all validation pairs")
    parser.add_argument("--test-pairs", type=int, default=0, help="0 downloads all test pairs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-caption", action="store_true", help="Append figure captions to questions")
    args = parser.parse_args(argv)
    if min(args.train_pairs, args.val_pairs, args.test_pairs) < 0:
        parser.error("Pair counts must be nonnegative (0 means all)")
    return args


def main() -> None:
    args = parse_args()
    revision = urllib.parse.quote(args.revision, safe="")
    base_url = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{revision}/data"
    cache = args.output / ".download_cache" / hashlib.sha256(args.revision.encode()).hexdigest()[:16]
    splits = {}
    source_counts = {}
    counts = (args.train_pairs, args.val_pairs, args.test_pairs)
    for index, (split, filename) in enumerate(SOURCE_FILES.items()):
        path = cache / filename
        download_file(f"{base_url}/{filename}", path)
        with path.open(encoding="utf-8") as handle:
            records = json.load(handle)
        if not isinstance(records, list) or not records:
            raise ValueError(f"Expected a nonempty JSON array in {path}; remove invalid cache and retry")
        rows = format_rows(records, include_caption=args.include_caption)
        source_counts[split] = len(rows)
        splits[split] = select_rows(rows, counts[index], args.seed + index)
    validate_splits(splits)
    for split, rows in splits.items():
        names = {row["image"] for row in rows}
        archive = cache / f"images_{split}.zip"
        download_file(f"{base_url}/{archive.name}", archive)
        extract_images(archive, names, args.output / "images")
        print(f"{split}: {len(rows)} pairs, {len(names)} images", flush=True)
    for split, rows in splits.items():
        filename = "val.csv" if split == "validation" else f"{split}.csv"
        write_csv(args.output / filename, rows)
    metadata = {
        "source": f"https://huggingface.co/datasets/{DATASET_ID}",
        "revision": args.revision,
        "source_files": SOURCE_FILES,
        "source_pairs": source_counts,
        "seed": args.seed,
        "selection": "all source rows or seeded sample within each official split",
        "include_caption": args.include_caption,
        "answer_options_in_question": True,
        "pairs": {split: len(rows) for split, rows in splits.items()},
        "unique_images": len({row["image"] for rows in splits.values() for row in rows}),
    }
    temporary = args.output / "metadata.json.part"
    temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output / "metadata.json")
    print(f"Done: {args.output}", flush=True)


if __name__ == "__main__":
    main()
