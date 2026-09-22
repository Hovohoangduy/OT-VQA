"""Download a small, reproducible GQA subset from Hugging Face.

The output matches this repository's CSV/image-folder VQA format. Only Python's
standard library and Pillow (installed transitively with torchvision) are required.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image


DATASET_ID = "lmms-lab-encoder/GQA"
ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"
DATASET_API = "https://huggingface.co/api/datasets/lmms-lab-encoder/GQA"
PAGE_SIZE = 100


def fetch_bytes(url: str, attempts: int = 6) -> bytes:
    headers = {"User-Agent": "OTD-VQA-GQA-downloader/1.0"}
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=180) as response:
                return response.read()
        except Exception:
            if attempt + 1 == attempts:
                raise
            time.sleep(min(2**attempt, 20))
    raise RuntimeError("unreachable")


def fetch_json(url: str) -> dict:
    return json.loads(fetch_bytes(url).decode("utf-8"))


def rows_url(config: str, split: str, offset: int, length: int = PAGE_SIZE) -> str:
    query = urllib.parse.urlencode(
        {
            "dataset": DATASET_ID,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    return f"{ROWS_ENDPOINT}?{query}"


def fetch_image_records(source_split: str, count: int, workers: int) -> list[dict]:
    if count < 1 or workers < 1:
        raise ValueError("count and workers must be positive")
    config = f"{source_split}_balanced_images"
    offsets = list(range(0, count, PAGE_SIZE))
    pages = {}
    with ThreadPoolExecutor(max_workers=min(workers, len(offsets))) as pool:
        futures = {
            pool.submit(fetch_json, rows_url(config, source_split, offset)): offset
            for offset in offsets
        }
        for future in as_completed(futures):
            offset = futures[future]
            pages[offset] = future.result()["rows"]
    records = []
    for offset in sorted(pages):
        for item in pages[offset]:
            row = item["row"]
            records.append({"id": row["id"], "url": row["image"]["src"]})
    if len(records) < count:
        raise RuntimeError(f"Requested {count} images but only received {len(records)}")
    return records[:count]


def fetch_first_questions(
    source_split: str, image_ids: set[str], workers: int
) -> dict[str, dict[str, str]]:
    config = f"{source_split}_balanced_instructions"
    selected: dict[str, dict[str, str]] = {}
    offset = 0
    window_pages = max(4, workers * 2)

    while len(selected) < len(image_ids):
        offsets = [offset + PAGE_SIZE * index for index in range(window_pages)]
        pages = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    fetch_json,
                    rows_url(config, source_split, page_offset),
                ): page_offset
                for page_offset in offsets
            }
            for future in as_completed(futures):
                page_offset = futures[future]
                payload = future.result()
                pages[page_offset] = payload["rows"]
                total_rows = payload["num_rows_total"]

        for page_offset in sorted(pages):
            for item in pages[page_offset]:
                row = item["row"]
                image_id = row["imageId"]
                if image_id in image_ids and image_id not in selected:
                    question = str(row.get("question") or "").strip()
                    answer = str(row.get("answer") or "").strip()
                    if question and answer:
                        selected[image_id] = {"question": question, "answer": answer}

        offset += PAGE_SIZE * window_pages
        print(f"  annotations: {len(selected)}/{len(image_ids)}", flush=True)
        if offset >= total_rows:
            break

    missing = image_ids.difference(selected)
    if missing:
        preview = ", ".join(sorted(missing)[:10])
        raise RuntimeError(
            f"No question/answer found for {len(missing)} images: {preview}"
        )
    return selected


def valid_image(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def download_image(record: dict, destination: Path) -> str:
    path = destination / f"{record['id']}.jpg"
    if valid_image(path):
        return path.name
    temporary = path.with_suffix(".jpg.part")
    temporary.write_bytes(fetch_bytes(record["url"]))
    if not valid_image(temporary):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file is not a valid image: {record['id']}")
    os.replace(temporary, path)
    return path.name


def materialize_split(
    name: str,
    records: list[dict],
    annotations: dict[str, dict[str, str]],
    output_dir: Path,
    workers: int,
) -> list[dict[str, str]]:
    image_dir = output_dir / "images" / name
    image_dir.mkdir(parents=True, exist_ok=True)
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(download_image, record, image_dir): record
            for record in records
        }
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed % 50 == 0 or completed == len(records):
                print(f"  {name} images: {completed}/{len(records)}", flush=True)

    rows = []
    for record in records:
        annotation = annotations[record["id"]]
        rows.append(
            {
                "anno_id": str(record["id"]),
                "image": f"{name}/{record['id']}.jpg",
                "question": annotation["question"],
                "answer": annotation["answer"],
            }
        )
    csv_path = output_dir / f"{name}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("anno_id", "image", "question", "answer"))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a train/val/test GQA subset")
    parser.add_argument("--output", default="data/gqa_dataset")
    parser.add_argument("--train-images", type=int, default=5000)
    parser.add_argument("--val-images", type=int, default=1000)
    parser.add_argument("--test-images", type=int, default=500)
    parser.add_argument("--workers", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.train_images, args.val_images, args.test_images, args.workers) < 1:
        raise ValueError("Image counts and workers must be positive")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching train image metadata...", flush=True)
    train_records = fetch_image_records("train", args.train_images, args.workers)
    print("Fetching validation image metadata...", flush=True)
    held_out_count = args.val_images + args.test_images
    held_out = fetch_image_records("val", held_out_count, args.workers)
    val_records = held_out[: args.val_images]
    test_records = held_out[args.val_images :]

    train_ids = {record["id"] for record in train_records}
    held_out_ids = {record["id"] for record in held_out}
    if train_ids.intersection(held_out_ids):
        raise RuntimeError("Train and held-out image IDs overlap")

    print("Fetching train annotations...", flush=True)
    train_annotations = fetch_first_questions("train", train_ids, args.workers)
    print("Fetching validation/test annotations...", flush=True)
    held_out_annotations = fetch_first_questions("val", held_out_ids, args.workers)

    print("Downloading images and writing CSV files...", flush=True)
    train_rows = materialize_split(
        "train", train_records, train_annotations, output_dir, args.workers
    )
    val_rows = materialize_split(
        "val", val_records, held_out_annotations, output_dir, args.workers
    )
    test_rows = materialize_split(
        "test", test_records, held_out_annotations, output_dir, args.workers
    )

    corpus = "\n".join(
        value
        for row in train_rows
        for value in (row["question"], row["answer"])
    )
    (output_dir / "corpus.txt").write_text(corpus + "\n", encoding="utf-8")

    dataset_metadata = fetch_json(DATASET_API)
    metadata = {
        "source": f"https://huggingface.co/datasets/{DATASET_ID}",
        "dataset_id": DATASET_ID,
        "revision": dataset_metadata.get("sha"),
        "selection": "first balanced image records; first balanced QA per image",
        "source_splits": {"train": "train", "val": "val", "test": "val"},
        "counts": {
            "train_images": len(train_records),
            "val_images": len(val_records),
            "test_images": len(test_records),
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "test_rows": len(test_rows),
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Done: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
