"""Download a small PlantExpertVQA subset without fetching the full image ZIPs.

Annotations stream from the source CSV files. Some individual image URLs in the
Hub repository return 404, so images are read by HTTP byte range from the four
ZIP archives instead. The output CSVs work with this repository's VQADataset.
"""

from __future__ import annotations

import argparse
import csv
import http.client
import io
import json
import math
import os
import random
import re
import struct
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path

from PIL import Image


DATASET_ID = "SyedNazmusSakib/PlantExpertVQA"
BASE_URL = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/main"
PAGE_SIZE = 100
CSV_FIELDS = ("anno_id", "image", "question", "answer", "question_category", "crop", "disease")


def retry_delay(headers: object, attempt: int) -> float:
    """Honor the server's reset time before falling back to exponential delay."""
    delays = [min(5 * 2 ** attempt, 120)]
    retry_after = headers.get("Retry-After")
    if retry_after:
        try:
            delays.append(float(retry_after))
        except ValueError:
            try:
                delays.append(parsedate_to_datetime(retry_after).timestamp() - time.time())
            except (TypeError, ValueError, OverflowError):
                pass
    rate_limit = headers.get("RateLimit", "")
    match = re.search(r"(?:^|;)\s*t=(\d+)", rate_limit)
    if match:
        delays.append(int(match.group(1)) + 1)
    return min(max(1.0, *delays), 300.0)


def request(url: str, *, start: int | None = None, end: int | None = None,
            method: str = "GET") -> tuple[bytes, object]:
    headers = {"User-Agent": "OT-VQA-PlantExpert-downloader/1.0"}
    if os.environ.get("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    if start is not None:
        headers["Range"] = f"bytes={start}-{end}"
    rate_limits = 0
    transient_failures = 0
    while True:
        try:
            req = urllib.request.Request(url, headers=headers, method=method)
            with urllib.request.urlopen(req, timeout=90) as response:
                if start is not None:
                    content_range = response.headers.get("Content-Range", "")
                    if response.status != 206 or not content_range.startswith(f"bytes {start}-{end}/"):
                        raise RuntimeError(f"Server ignored byte range for {url}")
                if method == "HEAD":
                    return b"", response.headers
                return response.read(), response.headers
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                rate_limits += 1
                if rate_limits > 12:
                    raise RuntimeError(f"Rate limit persisted after 12 retries: {url}") from exc
                delay = retry_delay(exc.headers, rate_limits - 1)
                print(f"Rate limited; waiting {delay:.0f}s before retrying", flush=True)
            elif exc.code in (500, 502, 503, 504):
                transient_failures += 1
                if transient_failures >= 6:
                    raise
                delay = min(2 ** (transient_failures - 1), 30)
            else:
                raise
        except (urllib.error.URLError, TimeoutError, http.client.IncompleteRead,
                http.client.RemoteDisconnected, ConnectionResetError):
            transient_failures += 1
            if transient_failures >= 6:
                raise
            delay = min(2 ** (transient_failures - 1), 30)
        time.sleep(delay)


def archive_size(url: str) -> int:
    _, headers = request(url, method="HEAD")
    return int(headers["Content-Length"])


class RemoteZipReader(io.RawIOBase):
    """Seekable reader used only while zipfile reads a ZIP's central directory."""

    def __init__(self, url: str, size: int):
        self.url = url
        self.size = size
        self.position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            self.position = offset
        elif whence == os.SEEK_CUR:
            self.position += offset
        elif whence == os.SEEK_END:
            self.position = self.size + offset
        else:
            raise ValueError("Invalid seek mode")
        return self.position

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self.size - self.position
        size = min(size, self.size - self.position)
        if size <= 0:
            return b""
        data, _ = request(self.url, start=self.position, end=self.position + size - 1)
        self.position += len(data)
        return data


@dataclass(frozen=True)
class ImageEntry:
    archive_url: str
    archive_size: int
    header_offset: int
    name_length: int
    compressed_size: int
    file_size: int
    compression: int
    flags: int
    crc: int


def index_images() -> dict[str, ImageEntry]:
    images: dict[str, ImageEntry] = {}
    for part in range(1, 5):
        url = f"{BASE_URL}/images_part{part}.zip"
        size = archive_size(url)
        with zipfile.ZipFile(RemoteZipReader(url, size)) as archive:
            added = 0
            for info in archive.infolist():
                name = info.filename.removeprefix("images/")
                if info.filename != f"images/{name}" or not name or "/" in name:
                    continue
                if name in images:
                    raise RuntimeError(f"Duplicate image in archives: {name}")
                images[name] = ImageEntry(
                    url, size, info.header_offset, len(info.filename.encode("utf-8")),
                    info.compress_size,
                    info.file_size, info.compress_type, info.flag_bits, info.CRC,
                )
                added += 1
        print(f"Indexed archive {part}/4: {added} images", flush=True)
    return images


def sample_csv_pages(split: str, count: int, seed: int) -> list[list[dict]]:
    """Reservoir-sample groups of 100 rows using one streaming CSV request."""
    source_name = "train" if split == "train" else "val"
    url = f"{BASE_URL}/data/{source_name}.csv"
    target_pages = max(math.ceil(count / PAGE_SIZE) + 2,
                       math.ceil(count / PAGE_SIZE * 1.5))
    headers = {"User-Agent": "OT-VQA-PlantExpert-downloader/1.0"}
    if os.environ.get("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    for attempt in range(6):
        rng = random.Random(seed)
        pages: list[list[dict]] = []
        current_page: list[dict] | None = None
        pages_seen = 0
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=180) as response:
                reader = csv.DictReader(io.TextIOWrapper(response, encoding="utf-8", newline=""))
                for row_number, row in enumerate(reader):
                    if row_number % PAGE_SIZE == 0:
                        pages_seen += 1
                        if len(pages) < target_pages:
                            current_page = []
                            pages.append(current_page)
                        else:
                            slot = rng.randrange(pages_seen)
                            if slot < target_pages:
                                current_page = []
                                pages[slot] = current_page
                            else:
                                current_page = None
                    if current_page is not None:
                        current_page.append(row)
            rng.shuffle(pages)
            print(f"Sampled {len(pages)} {split} pages from the source CSV", flush=True)
            return pages
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 5:
                raise
            delay = retry_delay(exc.headers, attempt)
        except (urllib.error.URLError, TimeoutError, http.client.IncompleteRead,
                http.client.RemoteDisconnected, ConnectionResetError):
            if attempt == 5:
                raise
            delay = min(5 * 2 ** attempt, 120)
        print(f"Source CSV interrupted; retrying in {delay:.0f}s", flush=True)
        time.sleep(delay)
    raise RuntimeError("unreachable")


def choose_rows(split: str, count: int, seed: int,
                image_index: dict[str, ImageEntry], cache_dir: Path) -> list[dict[str, str]]:
    cache_path = cache_dir / f"{split}_{count}_seed{seed}.json"
    if cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if (len(cached) == count and all(row["image"] in image_index for row in cached)):
                print(f"Reusing {count} cached {split} pairs", flush=True)
                return cached
        except (OSError, ValueError, TypeError, KeyError):
            pass

    pages = sample_csv_pages(split, count, seed)
    selected: list[dict[str, str]] = []
    seen: set[str] = set()
    for page in pages:
        for row in page:
            image_path = str(row.get("image_path") or "")
            image_name = image_path.removeprefix("images/")
            qa_id = str(row.get("qa_id") or "")
            question = str(row.get("question_text") or "").strip()
            answer = str(row.get("answer") or "").strip()
            if (image_path != f"images/{image_name}" or "/" in image_name
                    or image_name not in image_index or not qa_id or qa_id in seen
                    or not question or not answer):
                continue
            seen.add(qa_id)
            selected.append({
                "anno_id": qa_id,
                "image": image_name,
                "question": question,
                "answer": answer,
                "question_category": str(row.get("question_category") or ""),
                "crop": str(row.get("crop") or ""),
                "disease": str(row.get("disease") or ""),
            })
            if len(selected) == count:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache_path.with_suffix(".json.part")
                temporary.write_text(json.dumps(selected, ensure_ascii=False), encoding="utf-8")
                os.replace(temporary, cache_path)
                print(f"Selected {count} {split} pairs", flush=True)
                return selected
    raise RuntimeError(f"Only found {len(selected)} valid {split} pairs; requested {count}")


def read_image(entry: ImageEntry) -> bytes:
    # ZIP metadata gives the compressed size and filename length. Fetch the
    # local header plus member in one range; 1024 bytes covers its extra field
    # in these archives. Fetch a tail only if a member has a larger extra field.
    end = min(entry.archive_size - 1,
              entry.header_offset + 30 + entry.name_length + 1024
              + entry.compressed_size - 1)
    chunk, _ = request(entry.archive_url, start=entry.header_offset, end=end)
    header = chunk[:30]
    signature, _, flags, compression, _, _, _, _, _, name_length, extra_length = struct.unpack(
        "<IHHHHHIIIHH", header
    )
    if (signature != 0x04034B50 or compression != entry.compression
            or flags != entry.flags or flags & 1):
        raise RuntimeError("Unsupported or invalid ZIP member")
    member_start = 30 + name_length + extra_length
    compressed = chunk[member_start:member_start + entry.compressed_size]
    if len(compressed) < entry.compressed_size:
        tail_start = entry.header_offset + len(chunk)
        tail_end = entry.header_offset + member_start + entry.compressed_size - 1
        tail, _ = request(entry.archive_url, start=tail_start, end=tail_end)
        compressed += tail
    if entry.compression == zipfile.ZIP_STORED:
        data = compressed
    elif entry.compression == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(compressed, -15)
    else:
        raise RuntimeError(f"Unsupported ZIP compression: {entry.compression}")
    if len(data) != entry.file_size or zlib.crc32(data) != entry.crc:
        raise RuntimeError("Image failed ZIP size or CRC check")
    with Image.open(io.BytesIO(data)) as image:
        image.verify()
    return data


def valid_image(path: Path) -> bool:
    if not path.is_file() or not path.stat().st_size:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def save_image(name: str, entry: ImageEntry, folder: Path) -> None:
    path = folder / name
    if valid_image(path):
        return
    data = read_image(entry)
    temporary = folder / f"{name}.part"
    temporary.write_bytes(data)
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    temporary = path.with_suffix(".csv.part")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download PlantExpertVQA train/validation pairs")
    parser.add_argument("--output", type=Path, default=Path("data/plantexpert_dataset"))
    parser.add_argument("--train-pairs", type=int, default=5000)
    parser.add_argument("--val-pairs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.train_pairs, args.val_pairs, args.workers) < 1:
        raise ValueError("Pair counts and workers must be positive")
    image_index = index_images()
    cache_dir = args.output / ".selection_cache"
    train_rows = choose_rows("train", args.train_pairs, args.seed, image_index, cache_dir)
    val_rows = choose_rows("validation", args.val_pairs, args.seed + 1, image_index, cache_dir)
    if {r["image"] for r in train_rows} & {r["image"] for r in val_rows}:
        raise RuntimeError("Training and validation images overlap")

    image_folder = args.output / "images"
    image_folder.mkdir(parents=True, exist_ok=True)
    image_names = sorted({r["image"] for r in train_rows + val_rows})
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(save_image, name, image_index[name], image_folder): name
                   for name in image_names}
        for completed, future in enumerate(as_completed(futures), 1):
            future.result()
            if completed % 100 == 0 or completed == len(image_names):
                print(f"Images: {completed}/{len(image_names)}", flush=True)

    write_csv(args.output / "train.csv", train_rows)
    write_csv(args.output / "val.csv", val_rows)
    metadata = {
        "source": f"https://huggingface.co/datasets/{DATASET_ID}",
        "seed": args.seed,
        "selection": "seeded reservoir sample of source CSV pages",
        "train_pairs": len(train_rows),
        "validation_pairs": len(val_rows),
        "unique_images": len(image_names),
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Done: {args.output}", flush=True)


if __name__ == "__main__":
    main()
