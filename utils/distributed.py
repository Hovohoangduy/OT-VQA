"""Small Distributed Data Parallel helpers used by training only."""

from __future__ import annotations

from dataclasses import dataclass
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from utils.device import resolve_device


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.enabled:
            dist.barrier()


def initialize_distributed(requested_device: str) -> DistributedContext:
    """Initialize NCCL when launched by torchrun; otherwise use one device."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        device = resolve_device(requested_device)
        return DistributedContext(False, 0, 1, 0, device)
    if requested_device not in {"auto", "cuda"}:
        raise ValueError("Multi-GPU training requires --device cuda or auto")
    if not torch.cuda.is_available():
        raise RuntimeError("torchrun requested multi-GPU training, but CUDA is unavailable")
    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} exceeds {torch.cuda.device_count()} visible CUDA devices"
        )
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return DistributedContext(
        True,
        int(os.environ.get("RANK", str(local_rank))),
        world_size,
        local_rank,
        torch.device("cuda", local_rank),
    )


def cleanup_distributed(context: DistributedContext) -> None:
    if context.enabled and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(module):
    return module.module if isinstance(module, DistributedDataParallel) else module


def reduce_totals(values, context: DistributedContext) -> list[float]:
    """Sum scalar counters across workers and return them on every rank."""
    tensor = torch.tensor(values, dtype=torch.float64, device=context.device)
    if context.enabled:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.cpu().tolist()


def broadcast_object(value, context: DistributedContext):
    if not context.enabled:
        return value
    payload = [value if context.is_main else None]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]
