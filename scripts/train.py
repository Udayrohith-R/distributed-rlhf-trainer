#!/usr/bin/env python3
"""
Launch script for distributed RLHF training.

Usage (single node, 8 GPUs):
    torchrun --nproc_per_node=8 scripts/train.py --config configs/ppo_7b.yaml

Usage (multi-node, 2 nodes x 8 GPUs):
    torchrun --nnodes=2 --nproc_per_node=8 --rdzv_id=rlhf_run \
        --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29500 \
        scripts/train.py --config configs/ppo_7b.yaml
"""

import argparse
import logging
import os

import torch
import wandb
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from rl_orchestrator.trainer.ppo_trainer import PPOConfig, PPOTrainer
from rl_orchestrator.rollout.rollout_worker import AsyncRolloutOrchestrator, RolloutConfig
from rl_orchestrator.utils.distributed import init_distributed, is_main_process, barrier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distributed RLHF Training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument("--wandb_project", type=str, default="rl-orchestrator")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    init_distributed()

    if is_main_process():
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=cfg,
        )

    ppo_cfg = PPOConfig(**cfg.get("ppo", {}))
    rollout_cfg = RolloutConfig(**cfg.get("rollout", {}))
    model_path = cfg["model"]["name_or_path"]

    logger.info(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    policy = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    )
    ref_policy = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    )
    value_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    )

    trainer = PPOTrainer(
        policy=policy,
        ref_policy=ref_policy,
        value_model=value_model,
        config=ppo_cfg,
        tokenizer=tokenizer,
    )

    orchestrator = AsyncRolloutOrchestrator(
        config=rollout_cfg,
        model_name_or_path=model_path,
    )

    if args.resume:
        from rl_orchestrator.trainer.fsdp_utils import load_fsdp_checkpoint
        step = load_fsdp_checkpoint(
            trainer.policy,
            trainer.value_model,
            trainer.policy_optimizer,
            args.resume,
        )
        trainer.global_step = step
        logger.info(f"Resumed from step {step}")

    barrier()
    logger.info("Training started")

    # Training loop placeholder — plug in your dataset and reward fn here
    for step in range(trainer.global_step, ppo_cfg.n_steps):
        # TODO: replace with real batch from rollout orchestrator
        batch = _dummy_batch(ppo_cfg, tokenizer)
        stats = trainer.train_step(batch)
        trainer.log_to_wandb(stats, step)

        if is_main_process() and step % 10 == 0:
            logger.info(
                f"step={step} | loss={stats.policy_loss:.4f} "
                f"| kl={stats.approx_kl:.4f} | tok/s={stats.tokens_per_second:.0f}"
            )

    orchestrator.shutdown()
    if is_main_process():
        wandb.finish()


def _dummy_batch(cfg: PPOConfig, tokenizer) -> dict:
    """Placeholder batch for smoke testing. Replace with real data loader."""
    B, T = cfg.mini_batch_size, 64
    return {
        "input_ids": torch.randint(0, tokenizer.vocab_size, (B, T)).cuda(),
        "attention_mask": torch.ones(B, T, dtype=torch.long).cuda(),
        "logprobs": torch.randn(B, T - 1).cuda(),
        "ref_logprobs": torch.randn(B, T - 1).cuda(),
        "values": torch.randn(B, T).cuda(),
        "advantages": torch.randn(B, T - 1).cuda(),
        "returns": torch.randn(B, T - 1).cuda(),
        "rewards": torch.randn(B).cuda(),
    }


if __name__ == "__main__":
    main()
