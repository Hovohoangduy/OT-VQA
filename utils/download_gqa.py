"""Download a small, reproducible GQA subset from Hugging Face.

The output matches this repository's CSV/image-folder VQA format. Only Python's
standard library and Pillow (installed transitively with torchvision) are required.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import http.client
import json
import math
import os
import random
import re
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from email.utils import parsedate_to_datetime
from pathlib import Path

from PIL import Image


DATASET_ID = "lmms-lab-encoder/GQA"
ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"
DATASET_API = "https://huggingface.co/api/datasets/lmms-lab-encoder/GQA"
PAGE_SIZE = 100
_metadata_lock = threading.Lock()
_request_interval = 1.0
_next_metadata_request = 0.0


def _is_metadata_url(url: str) -> bool:
    parsed = urllib.parse.urlsplit(url)
    return (parsed.hostname == "datasets-server.huggingface.co" and parsed.path == "/rows") or (
        parsed.hostname == "huggingface.co" and parsed.path.startswith("/api/")
    )


def _wait_for_metadata_slot() -> None:
    """Share request spacing and cooldowns across all metadata workers."""
    global _next_metadata_request
    while True:
        with _metadata_lock:
            now = time.monotonic()
            delay = _next_metadata_request - now
            if delay <= 0:
                _next_metadata_request = now + _request_interval
                return
        # Recheck after sleeping: another worker may have extended the cooldown.
        time.sleep(min(delay, 60))


def _pause_metadata_requests(delay: float) -> None:
    global _next_metadata_request
    with _metadata_lock:
        _next_metadata_request = max(_next_metadata_request, time.monotonic() + delay)


def _retry_delay(error: urllib.error.HTTPError, attempt: int) -> float:
    """Honor both Retry-After and Hugging Face's RateLimit reset time."""
    delays = []
    headers = error.headers or {}
    retry_after = headers.get("Retry-After")
    if retry_after:
        try:
            seconds = float(retry_after)
        except ValueError:
            try:
                seconds = parsedate_to_datetime(retry_after).timestamp() - time.time()
            except (TypeError, ValueError, OverflowError):
                seconds = float("nan")
        if math.isfinite(seconds):
            delays.append(max(0, seconds) + 1)
    for reset in re.findall(r"(?:^|;)\s*t=(\d+)", headers.get("RateLimit", "")):
        delays.append(float(reset) + 1)
    return max(delays) if delays else min(5 * 2**attempt, 300)


def _sleep_retry(delay: float) -> None:
    while delay > 0:
        interval = min(delay, 60)
        time.sleep(interval)
        delay -= interval


def fetch_bytes(url: str, attempts: int = 12) -> bytes:
    headers = {"User-Agent": "OTD-VQA-GQA-downloader/1.0"}
    metadata_request = _is_metadata_url(url)
    if metadata_request and os.environ.get("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    for attempt in range(attempts):
        if metadata_request:
            _wait_for_metadata_slot()
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=180) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code != 429 and error.code not in (500, 502, 503, 504):
                raise
            if attempt + 1 == attempts:
                raise
            delay = _retry_delay(error, attempt)
            print(
                f"  HTTP {error.code}; waiting {delay:.0f}s before retry "
                f"{attempt + 2}/{attempts}", flush=True,
            )
            if metadata_request:
                _pause_metadata_requests(delay)
            else:
                _sleep_retry(delay)
        except (urllib.error.URLError, TimeoutError, ConnectionError,
                http.client.IncompleteRead, http.client.RemoteDisconnected):
            if attempt + 1 == attempts:
                raise
            _sleep_retry(min(2**attempt, 30))
    raise RuntimeError("unreachable")


def _image_url_expired(url: str) -> bool:
    expires = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query).get("Expires")
    if not expires:
        return False
    try:
        return float(expires[0]) <= time.time() + 60
    except ValueError:
        return True


def fetch_json(url: str, cache_dir: Path | None = None) -> dict:
    cache_path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{hashlib.sha256(url.encode()).hexdigest()}.json"
        if cache_path.is_file():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                # Viewer image URLs expire; annotation pages can be reused as-is.
                if not any(
                    _image_url_expired(item["row"]["image"]["src"])
                    for item in cached.get("rows", [])
                    if isinstance(item, dict) and "image" in item.get("row", {})
                ):
                    return cached
            except (OSError, ValueError):
                pass  # Refetch incomplete or damaged cache files.
    payload = json.loads(fetch_bytes(url).decode("utf-8"))
    if cache_path is not None:
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=cache_dir, suffix=".part", delete=False
            ) as handle:
                temporary = Path(handle.name)
                json.dump(payload, handle)
            os.replace(temporary, cache_path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return payload


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


def fetch_image_records(
    source_split: str, count: int, workers: int, seed: int = 42,
    cache_dir: Path | None = None,
) -> list[dict]:
    if count < 1 or workers < 1:
        raise ValueError("count and workers must be positive")
    config = f"{source_split}_balanced_images"
    first_page = fetch_json(rows_url(config, source_split, 0), cache_dir)
    total_rows = first_page["num_rows_total"]
    if count > total_rows:
        raise ValueError(
            f"Requested {count} {source_split} images but only {total_rows} are available"
        )

    # Sample distinct row positions across the full split, then fetch each
    # required page once. Preserve sample order when workers finish out of order.
    indices = random.Random(seed).sample(range(total_rows), count)
    offsets = sorted({index // PAGE_SIZE * PAGE_SIZE for index in indices})
    pages = {0: first_page["rows"]}
    pending_offsets = [offset for offset in offsets if offset != 0]
    if pending_offsets:
        with ThreadPoolExecutor(max_workers=min(workers, len(pending_offsets))) as pool:
            futures = {
                pool.submit(fetch_json, rows_url(config, source_split, offset), cache_dir): offset
                for offset in pending_offsets
            }
            for future in as_completed(futures):
                offset = futures[future]
                pages[offset] = future.result()["rows"]

    records = []
    for index in indices:
        offset = index // PAGE_SIZE * PAGE_SIZE
        position = index - offset
        if position >= len(pages[offset]):
            raise RuntimeError(f"Missing sampled {source_split} image row {index}")
        row = pages[offset][position]["row"]
        records.append({
            "id": row["id"], "url": row["image"]["src"],
            "row_url": rows_url(config, source_split, index, length=1),
        })
    return records


def fetch_first_questions(
    source_split: str, image_ids: set[str], workers: int,
    cache_dir: Path | None = None,
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
                    cache_dir,
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
    image_url = record["url"]
    if _image_url_expired(image_url) and "row_url" in record:
        image_url = fetch_json(record["row_url"])["rows"][0]["row"]["image"]["src"]
    try:
        content = fetch_bytes(image_url)
    except urllib.error.HTTPError as error:
        if error.code not in (403, 404) or "row_url" not in record:
            raise
        # A signed URL may expire or the viewer may evict its image cache.
        image_url = fetch_json(record["row_url"])["rows"][0]["row"]["image"]["src"]
        content = fetch_bytes(image_url)
    temporary.write_bytes(content)
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
    parser = argparse.ArgumentParser(description="Download a random train/val/test GQA subset")
    parser.add_argument("--output", default="data/gqa_dataset")
    parser.add_argument("--train-images", type=int, default=10000)
    parser.add_argument("--val-images", type=int, default=1000)
    parser.add_argument("--test-images", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42, help="Random sampling seed")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--metadata-workers", type=int, default=1,
                        help="Concurrent metadata requests (independent of image workers)")
    parser.add_argument("--request-interval", type=float, default=1.0,
                        help="Minimum seconds between metadata requests")
    return parser.parse_args()


def main() -> None:
    global _request_interval, _next_metadata_request
    args = parse_args()
    if min(args.train_images, args.val_images, args.test_images,
           args.workers, args.metadata_workers) < 1:
        raise ValueError("Image counts and workers must be positive")
    if not math.isfinite(args.request_interval) or args.request_interval <= 0:
        raise ValueError("Request interval must be finite and positive")
    with _metadata_lock:
        _request_interval = args.request_interval
        _next_metadata_request = 0.0
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / ".row_cache"

    print("Fetching train image metadata...", flush=True)
    train_records = fetch_image_records(
        "train", args.train_images, args.metadata_workers, args.seed, cache_dir
    )
    print("Fetching validation image metadata...", flush=True)
    held_out_count = args.val_images + args.test_images
    held_out = fetch_image_records(
        "val", held_out_count, args.metadata_workers, args.seed, cache_dir
    )
    val_records = held_out[: args.val_images]
    test_records = held_out[args.val_images :]

    train_ids = {record["id"] for record in train_records}
    held_out_ids = {record["id"] for record in held_out}
    if train_ids.intersection(held_out_ids):
        raise RuntimeError("Train and held-out image IDs overlap")

    print("Fetching train annotations...", flush=True)
    train_annotations = fetch_first_questions(
        "train", train_ids, args.metadata_workers, cache_dir
    )
    print("Fetching validation/test annotations...", flush=True)
    held_out_annotations = fetch_first_questions(
        "val", held_out_ids, args.metadata_workers, cache_dir
    )

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
        "selection": "random balanced image records without replacement; first balanced QA per image",
        "seed": args.seed,
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
