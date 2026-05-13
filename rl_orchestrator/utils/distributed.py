"""Distributed training utilities for multi-node FSDP setup."""

from __future__ import annotations

import logging
import os
import socket
from typing import Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def init_distributed(
    backend: str = "nccl",
    timeout_minutes: int = 30,
) -> None:
    """
    Initialize the distributed process group.

    Reads RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT from
    environment variables (set by torchrun / SLURM launcher).
    """
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend=backend,
        init_method="env://",
        world_size=world_size,
        rank=rank,
        timeout=torch.distributed.default_pg_timeout,
    )

    # Verify connectivity
    if dist.is_initialized():
        barrier_tensor = torch.ones(1, device=f"cuda:{local_rank}")
        dist.all_reduce(barrier_tensor)
        logger.info(
            f"Distributed init complete | rank={rank} | local_rank={local_rank} "
            f"| world_size={world_size} | host={socket.gethostname()}"
        )


def get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized():
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
    return tensor


def all_gather_object(obj) -> list:
    if not dist.is_initialized():
        return [obj]
    output = [None] * get_world_size()
    dist.all_gather_object(output, obj)
    return output
