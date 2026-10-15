"""Optimization boundaries and reproducible run state shared by model families."""
from contextlib import nullcontext
from itertools import islice
from pathlib import Path
import hashlib
import importlib.metadata
import random
import subprocess
import zipfile

import numpy as np
import torch
import torch.distributed as dist


def accumulation_groups(loader, steps):
    if steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    iterator = iter(loader)
    while group := list(islice(iterator, steps)):
        yield group


def autocast_context(device, amp):
    if amp == "off":
        return nullcontext()
    if device.type != "cuda":
        raise ValueError("Training AMP is supported on CUDA; use off for CPU/MPS")
    if amp == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 is unsupported on this CUDA device; T4 requires FP16")
    return torch.autocast("cuda", dtype=torch.float16 if amp == "fp16" else torch.bfloat16)


def optimizer_update(model, optimizer, scheduler, scaler, gradient_clip, device, gradient_stats=None):
    """All ranks agree on overflow, clip only after one unscale, count real steps."""
    enabled = scaler is not None and scaler.is_enabled()
    if enabled:
        scaler.unscale_(optimizer)
    params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    finite = (torch.stack([torch.isfinite(p.grad).all() for p in params]).all().to(dtype=torch.long)
              if params else torch.tensor(0, device=device))
    if dist.is_initialized():
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if not finite.item():
        if not enabled:
            raise FloatingPointError("Nonfinite gradients without a GradScaler")
        scaler.update(new_scale=scaler.get_scale() * scaler.get_backoff_factor())
        optimizer.zero_grad(set_to_none=True)
        return False
    if gradient_stats is not None:
        categories = {}
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            for category in ("visual_cost", "query_cost", "query_value", "gates", "outputs", "lora_"):
                if category in name:
                    value = parameter.grad.detach().float().square().sum()
                    categories[category] = categories.get(category, 0.) + value
                    break
        gradient_stats.update({name: value.sqrt().item() for name, value in categories.items()})
    if gradient_clip:
        torch.nn.utils.clip_grad_norm_(params, gradient_clip, error_if_nonfinite=True)
    if enabled:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    if scheduler is not None:
        scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return True


def capture_rng():
    np_state = np.random.get_state()
    return {"torch": torch.get_rng_state(), "python": random.getstate(),
            "numpy": (np_state[0], np_state[1].tolist(), *np_state[2:]),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "mps": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None}


def restore_rng(state):
    torch.set_rng_state(state["torch"].cpu())
    random.setstate(state["python"])
    if state.get("numpy") is not None:
        n = state["numpy"]
        np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), *n[2:]))
    if state.get("cuda") is not None and torch.cuda.is_available():
        if torch.is_tensor(state["cuda"]):
            torch.cuda.set_rng_state(state["cuda"].cpu())
        else:
            torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])
    if state.get("mps") is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"].cpu())


def gather_rng():
    state = capture_rng()
    if not dist.is_initialized():
        return [state]
    states = [None] * dist.get_world_size()
    dist.all_gather_object(states, state)
    return states


def source_manifest(destination):
    root = Path(__file__).resolve().parents[1]
    paths = sorted([*root.glob("*.py"), *root.glob("model/*.py"),
                    *root.glob("utils/*.py"), *root.glob("configs/*.py"),
                    *root.glob("scripts/*.py"), *root.glob("tests/*.py"),
                    *root.glob("requirements*.txt")])
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    with zipfile.ZipFile(destination / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, str(path.relative_to(root)))
    packages = {}
    for name in ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "rouge-score", "tokenizers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True)
    return {"source_sha256": hashes, "packages": packages,
            "git_commit": commit.stdout.strip() if commit.returncode == 0 else None}


def cuda_memory_report(device):
    if device.type != "cuda":
        return {"device": str(device), "cuda_available": False}
    torch.cuda.synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    reserved = torch.cuda.memory_reserved(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    usable_headroom = min(free, free + reserved - peak_reserved) / total
    return {"device": str(device), "name": torch.cuda.get_device_name(device),
            "allocated_peak_bytes": torch.cuda.max_memory_allocated(device),
            "reserved_peak_bytes": torch.cuda.max_memory_reserved(device),
            "free_bytes": free, "total_bytes": total,
            "reserved_headroom_fraction": 1 - peak_reserved/total,
            "free_fraction_at_sample": free/total,
            "usable_headroom_fraction": max(0., usable_headroom)}
