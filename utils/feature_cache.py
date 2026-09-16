"""Validated on-disk cache for frozen image and question encoder outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


CACHE_FORMAT_VERSION = 1


def file_fingerprint(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FeatureCacheDataset(Dataset):
    """Load float16 encoder features after checking their provenance."""

    def __init__(self, cache_dir, csv_path=None, text_model=None, image_model=None):
        self.root = Path(cache_dir)
        manifest_path = self.root / "manifest.json"
        features_path = self.root / "features.pt"
        if not manifest_path.is_file() or not features_path.is_file():
            raise ValueError(f"Feature cache is incomplete: {self.root}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("format_version") != CACHE_FORMAT_VERSION:
            raise ValueError("Unsupported feature-cache format version")
        if csv_path is not None and self.manifest.get("dataset_fingerprint") != file_fingerprint(csv_path):
            raise ValueError("Feature cache does not match the CSV contents; rebuild it")
        if text_model is not None and self.manifest.get("text_model") != str(text_model):
            raise ValueError("Feature cache text encoder does not match the requested model")
        if image_model is not None and self.manifest.get("image_model") != str(image_model):
            raise ValueError("Feature cache image encoder does not match the requested model")
        self.samples = torch.load(features_path, map_location="cpu", weights_only=True)
        if len(self.samples) != self.manifest.get("sample_count"):
            raise ValueError("Feature cache sample count does not match its manifest")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def collate_feature_cache(samples):
    if not samples:
        raise ValueError("Cannot collate an empty feature batch")
    image_shapes = {tuple(sample["image_features"].shape) for sample in samples}
    if len(image_shapes) != 1:
        raise ValueError("Cached visual feature shapes differ within a batch")
    question_features = pad_sequence(
        [sample["question_features"] for sample in samples], batch_first=True
    )
    question_masks = pad_sequence(
        [sample["question_padding_mask"] for sample in samples],
        batch_first=True, padding_value=True,
    )
    return {
        "anno_ids": [sample["anno_id"] for sample in samples],
        "image_features": torch.stack([sample["image_features"] for sample in samples]),
        "question_features": question_features,
        "question_padding_mask": question_masks,
        "questions": [sample["question"] for sample in samples],
        "answers": [sample["answer"] for sample in samples],
    }


def write_feature_cache(cache_dir, samples, manifest):
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    payload = dict(manifest, format_version=CACHE_FORMAT_VERSION,
                   sample_count=len(samples), stored_dtype="float16")
    temporary = root / "features.pt.tmp"
    torch.save(samples, temporary)
    temporary.replace(root / "features.pt")
    temporary_manifest = root / "manifest.json.tmp"
    temporary_manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary_manifest.replace(root / "manifest.json")
