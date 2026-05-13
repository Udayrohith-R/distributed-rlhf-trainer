"""
FSDP model wrapping, checkpoint management, and gradient utilities.
"""

from __future__ import annotations

import glob
import logging
import os
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import (
    BackwardPrefetch,
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

logger = logging.getLogger(__name__)

_SHARDING_STRATEGIES = {
    "FULL_SHARD": ShardingStrategy.FULL_SHARD,
    "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
    "NO_SHARD": ShardingStrategy.NO_SHARD,
    "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
}


def build_fsdp_model(model: nn.Module, config) -> FSDP:
    """
    Wrap a model with FSDP using the provided config.

    Applies transformer-layer-level wrapping for optimal memory distribution.
    Mixed precision uses BF16 for params/grads, FP32 for reduction ops.
    """
    mp_policy = None
    if config.mixed_precision:
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16,
            cast_forward_inputs=True,
        )

    cpu_offload_policy = CPUOffload(offload_params=True) if config.cpu_offload else None

    bp = (
        BackwardPrefetch.BACKWARD_PRE
        if config.backward_prefetch
        else BackwardPrefetch.BACKWARD_POST
    )

    sharding = _SHARDING_STRATEGIES.get(config.sharding_strategy, ShardingStrategy.FULL_SHARD)

    # Attempt to auto-detect transformer block class for layer-wise wrapping
    transformer_cls = _detect_transformer_layer_cls(model)
    auto_wrap = None
    if transformer_cls:
        auto_wrap = lambda module, recurse, nonwrapped_numel: transformer_auto_wrap_policy(
            module, recurse, nonwrapped_numel, transformer_layer_cls={transformer_cls}
        )

    wrapped = FSDP(
        model,
        sharding_strategy=sharding,
        mixed_precision=mp_policy,
        cpu_offload=cpu_offload_policy,
        backward_prefetch=bp,
        auto_wrap_policy=auto_wrap,
        device_id=torch.cuda.current_device(),
        use_orig_params=True,  # Required for AdamW parameter groups
    )

    if dist.get_rank() == 0:
        logger.info(
            f"FSDP model wrapped | sharding={config.sharding_strategy} "
            f"| mixed_precision={config.mixed_precision} "
            f"| cpu_offload={config.cpu_offload}"
        )

    return wrapped


def get_fsdp_grad_norm(model: FSDP, max_norm: float) -> float:
    """Clip gradients and return the pre-clip gradient norm."""
    return model.clip_grad_norm_(max_norm).item()


def save_fsdp_checkpoint(
    policy: FSDP,
    value_model: FSDP,
    optimizer: torch.optim.Optimizer,
    step: int,
    checkpoint_dir: str,
    keep_last_n: int = 3,
) -> None:
    """
    Save a full state dict checkpoint from an FSDP model.

    Uses FULL_STATE_DICT mode to consolidate shards on rank 0.
    Only rank 0 writes to disk.
    """
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

    with FSDP.state_dict_type(policy, StateDictType.FULL_STATE_DICT, save_policy):
        policy_state = policy.state_dict()

    with FSDP.state_dict_type(value_model, StateDictType.FULL_STATE_DICT, save_policy):
        value_state = value_model.state_dict()

    if dist.get_rank() == 0:
        ckpt_path = os.path.join(checkpoint_dir, f"step_{step:07d}.pt")
        torch.save(
            {
                "step": step,
                "policy_state_dict": policy_state,
                "value_state_dict": value_state,
                "optimizer_state_dict": optimizer.state_dict(),
            },
            ckpt_path,
        )
        logger.info(f"Checkpoint saved: {ckpt_path}")
        _cleanup_old_checkpoints(checkpoint_dir, keep_last_n)


def load_fsdp_checkpoint(
    policy: FSDP,
    value_model: FSDP,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str,
) -> int:
    """Load a checkpoint and return the global step."""
    map_loc = {"cuda:0": f"cuda:{dist.get_rank()}"}
    ckpt = torch.load(checkpoint_path, map_location=map_loc)

    load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

    with FSDP.state_dict_type(policy, StateDictType.FULL_STATE_DICT, load_policy):
        policy.load_state_dict(ckpt["policy_state_dict"])

    with FSDP.state_dict_type(value_model, StateDictType.FULL_STATE_DICT, load_policy):
        value_model.load_state_dict(ckpt["value_state_dict"])

    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    step = ckpt["step"]

    logger.info(f"Checkpoint loaded from step {step}")
    return step


def _detect_transformer_layer_cls(model: nn.Module) -> Optional[type]:
    """Heuristically detect the transformer block class for auto-wrap policy."""
    for name, module in model.named_modules():
        cls_name = type(module).__name__
        if any(k in cls_name for k in ("DecoderLayer", "TransformerBlock", "Block", "Layer")):
            if sum(1 for _ in module.parameters()) > 10:
                return type(module)
    return None


def _cleanup_old_checkpoints(checkpoint_dir: str, keep_last_n: int) -> None:
    """Remove oldest checkpoints beyond keep_last_n."""
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "step_*.pt")))
    for ckpt in checkpoints[:-keep_last_n]:
        os.remove(ckpt)
        logger.debug(f"Removed old checkpoint: {ckpt}")
