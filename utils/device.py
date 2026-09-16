"""Device selection and deterministic seeding for CPU, CUDA, and Apple MPS."""

from __future__ import annotations

import random

import torch


def resolve_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()
    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("device must be one of: auto, cpu, cuda, mps")
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    if requested == "mps" and not torch.backends.mps.is_available():
        detail = ("this PyTorch build has no MPS support" if not torch.backends.mps.is_built()
                  else "MPS is built but unavailable on this macOS runtime")
        raise RuntimeError(f"MPS was requested, but {detail}")
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
