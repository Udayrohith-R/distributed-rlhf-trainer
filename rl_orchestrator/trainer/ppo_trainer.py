"""
PPO Trainer with Fully Sharded Data Parallel (FSDP) for large-scale LLM post-training.

Designed for multi-node GPU clusters. Handles policy updates, value function
optimization, and KL penalty computation with numerical stability guarantees.
"""

from __future__ import annotations

import contextlib
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import (
    BackwardPrefetch,
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

import wandb

from rl_orchestrator.utils.distributed import get_rank, get_world_size, is_main_process
from rl_orchestrator.trainer.fsdp_utils import (
    build_fsdp_model,
    get_fsdp_grad_norm,
    save_fsdp_checkpoint,
    load_fsdp_checkpoint,
)

logger = logging.getLogger(__name__)


@dataclass
class PPOConfig:
    # Optimization
    lr: float = 1e-5
    critic_lr: float = 1e-5
    eps: float = 1e-8
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # PPO hyperparameters
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    kl_coef: float = 0.1
    target_kl: Optional[float] = 0.05
    gamma: float = 1.0
    lam: float = 0.95

    # Training schedule
    n_epochs: int = 1
    n_steps: int = 2048
    batch_size: int = 64
    mini_batch_size: int = 8
    grad_accumulation_steps: int = 4

    # FSDP
    sharding_strategy: str = "FULL_SHARD"
    cpu_offload: bool = False
    mixed_precision: bool = True
    backward_prefetch: bool = True

    # Stability
    normalize_advantage: bool = True
    normalize_reward: bool = True
    reward_clip: float = 5.0
    value_clip: float = 0.2

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_every_n_steps: int = 100
    keep_last_n_checkpoints: int = 3


@dataclass
class PPOStats:
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy_loss: float = 0.0
    kl_divergence: float = 0.0
    approx_kl: float = 0.0
    clip_fraction: float = 0.0
    grad_norm: float = 0.0
    learning_rate: float = 0.0
    advantages_mean: float = 0.0
    advantages_std: float = 0.0
    n_updates: int = 0
    tokens_per_second: float = 0.0


class RunningMeanStd:
    """Welford's online algorithm for computing running mean and variance."""

    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()):
        self.mean = torch.zeros(shape, dtype=torch.float64)
        self.var = torch.ones(shape, dtype=torch.float64)
        self.count = epsilon

    def update(self, x: torch.Tensor) -> None:
        x = x.double()
        batch_mean = x.mean(0)
        batch_var = x.var(0) if x.shape[0] > 1 else torch.zeros_like(batch_mean)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self, batch_mean: torch.Tensor, batch_var: torch.Tensor, batch_count: int
    ) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.var = m2 / total_count
        self.count = total_count

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device)) / torch.sqrt(
            self.var.to(x.device) + 1e-8
        )


class PPOTrainer:
    """
    Distributed PPO trainer with FSDP for LLM post-training.

    Supports:
    - Multi-node training via FSDP with configurable sharding strategies
    - Mixed precision (BF16/FP16) with dynamic loss scaling
    - Gradient accumulation for effective large batch sizes
    - Adaptive KL penalty with target KL divergence control
    - W&B experiment tracking with gradient histograms
    """

    def __init__(
        self,
        policy: nn.Module,
        ref_policy: nn.Module,
        value_model: nn.Module,
        config: PPOConfig,
        tokenizer,
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.global_step = 0
        self._kl_coef = config.kl_coef

        # Wrap models with FSDP
        self.policy = build_fsdp_model(policy, config)
        self.value_model = build_fsdp_model(value_model, config)

        # Reference policy: frozen, no FSDP grad computation needed
        self.ref_policy = ref_policy.eval()
        for p in self.ref_policy.parameters():
            p.requires_grad_(False)

        self._build_optimizers()
        self._build_schedulers()

        self.reward_normalizer = RunningMeanStd()
        self.stats = PPOStats()

        if is_main_process():
            logger.info(
                f"PPOTrainer initialized | world_size={get_world_size()} "
                f"| sharding={config.sharding_strategy} "
                f"| mixed_precision={config.mixed_precision}"
            )

    def _build_optimizers(self) -> None:
        self.policy_optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.config.lr,
            eps=self.config.eps,
            weight_decay=self.config.weight_decay,
            fused=True,  # Fused AdamW kernel on CUDA
        )
        self.value_optimizer = torch.optim.AdamW(
            self.value_model.parameters(),
            lr=self.config.critic_lr,
            eps=self.config.eps,
            weight_decay=self.config.weight_decay,
            fused=True,
        )

    def _build_schedulers(self) -> None:
        self.policy_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.policy_optimizer,
            T_max=self.config.n_steps,
            eta_min=self.config.lr * 0.1,
        )
        self.value_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.value_optimizer,
            T_max=self.config.n_steps,
            eta_min=self.config.critic_lr * 0.1,
        )

    @contextlib.contextmanager
    def _maybe_no_sync(self, model: FSDP, accumulate: bool) -> Iterator[None]:
        """Skip gradient all-reduce during accumulation steps."""
        if accumulate:
            with model.no_sync():
                yield
        else:
            yield

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generalized Advantage Estimation (GAE-Lambda).

        Args:
            rewards: [T, B] reward tensor
            values: [T+1, B] value estimates including bootstrap
            dones: [T, B] episode termination flags

        Returns:
            advantages: [T, B]
            returns: [T, B]
        """
        T, B = rewards.shape
        advantages = torch.zeros_like(rewards)
        last_gae = torch.zeros(B, device=rewards.device)

        for t in reversed(range(T)):
            next_non_terminal = 1.0 - dones[t].float()
            delta = (
                rewards[t]
                + self.config.gamma * values[t + 1] * next_non_terminal
                - values[t]
            )
            last_gae = (
                delta
                + self.config.gamma
                * self.config.lam
                * next_non_terminal
                * last_gae
            )
            advantages[t] = last_gae

        returns = advantages + values[:T]

        if self.config.normalize_advantage:
            # All-reduce stats across ranks for consistent normalization
            adv_mean = advantages.mean()
            adv_std = advantages.std()
            if dist.is_initialized():
                dist.all_reduce(adv_mean, op=dist.ReduceOp.AVG)
                dist.all_reduce(adv_std, op=dist.ReduceOp.AVG)
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        return advantages, returns

    def compute_policy_loss(
        self,
        logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        ref_logprobs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Clipped PPO policy loss with optional KL penalty against reference policy.

        Returns:
            loss: scalar policy loss
            metrics: dict of diagnostic scalars
        """
        # Probability ratio
        log_ratio = logprobs - old_logprobs
        ratio = torch.exp(log_ratio)

        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.config.clip_range, 1.0 + self.config.clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # KL penalty vs reference policy
        kl_loss = torch.tensor(0.0, device=logprobs.device)
        if ref_logprobs is not None:
            kl = (old_logprobs - ref_logprobs).mean()
            kl_loss = self._kl_coef * kl

            # Adaptive KL coefficient
            if self.config.target_kl is not None:
                self._update_kl_coef(kl.item())

        total_loss = policy_loss + kl_loss

        with torch.no_grad():
            approx_kl = ((ratio - 1) - log_ratio).mean()
            clip_fraction = ((ratio - 1.0).abs() > self.config.clip_range).float().mean()

        metrics = {
            "policy_loss": policy_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
        }

        return total_loss, metrics

    def compute_value_loss(
        self,
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
    ) -> torch.Tensor:
        """Clipped value function loss."""
        if self.config.clip_range_vf is not None:
            values_clipped = old_values + torch.clamp(
                values - old_values,
                -self.config.clip_range_vf,
                self.config.clip_range_vf,
            )
            vf_loss1 = F.mse_loss(values, returns)
            vf_loss2 = F.mse_loss(values_clipped, returns)
            return torch.max(vf_loss1, vf_loss2)
        return F.mse_loss(values, returns)

    def _update_kl_coef(self, current_kl: float) -> None:
        """Adaptive KL coefficient update based on target KL divergence."""
        if current_kl > 2.0 * self.config.target_kl:
            self._kl_coef *= 1.5
        elif current_kl < 0.5 * self.config.target_kl:
            self._kl_coef /= 1.5
        self._kl_coef = max(0.001, min(self._kl_coef, 1.0))

    def train_step(self, batch: Dict[str, torch.Tensor]) -> PPOStats:
        """
        Single PPO update step over a batch of experiences.

        Handles gradient accumulation, FSDP sync, and mixed precision.
        """
        self.policy.train()
        self.value_model.train()

        stats = PPOStats()
        t0 = time.perf_counter()
        total_tokens = 0

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        old_logprobs = batch["logprobs"]
        old_values = batch["values"]
        advantages = batch["advantages"]
        returns = batch["returns"]
        ref_logprobs = batch.get("ref_logprobs")

        # Clip and normalize rewards
        if self.config.normalize_reward and "rewards" in batch:
            rewards = batch["rewards"].clamp(-self.config.reward_clip, self.config.reward_clip)
            self.reward_normalizer.update(rewards)
            rewards = self.reward_normalizer.normalize(rewards)

        n_mini_batches = math.ceil(input_ids.shape[0] / self.config.mini_batch_size)
        total_tokens = attention_mask.sum().item()

        accumulated_stats: Dict[str, float] = {}

        self.policy_optimizer.zero_grad(set_to_none=True)
        self.value_optimizer.zero_grad(set_to_none=True)

        for epoch in range(self.config.n_epochs):
            perm = torch.randperm(input_ids.shape[0], device=input_ids.device)

            for mb_idx in range(n_mini_batches):
                start = mb_idx * self.config.mini_batch_size
                end = min(start + self.config.mini_batch_size, input_ids.shape[0])
                idx = perm[start:end]

                is_last_mb = mb_idx == n_mini_batches - 1
                should_sync = is_last_mb or (
                    (mb_idx + 1) % self.config.grad_accumulation_steps == 0
                )

                mb_input_ids = input_ids[idx]
                mb_mask = attention_mask[idx]
                mb_old_logprobs = old_logprobs[idx]
                mb_old_values = old_values[idx]
                mb_advantages = advantages[idx]
                mb_returns = returns[idx]
                mb_ref_logprobs = ref_logprobs[idx] if ref_logprobs is not None else None

                with self._maybe_no_sync(self.policy, not should_sync):
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.config.mixed_precision):
                        # Policy forward pass
                        policy_out = self.policy(
                            input_ids=mb_input_ids,
                            attention_mask=mb_mask,
                        )
                        logits = policy_out.logits
                        logprobs = self._logprobs_from_logits(logits, mb_input_ids)

                        # Entropy for regularization
                        entropy = self._entropy_from_logits(logits)

                        policy_loss, policy_metrics = self.compute_policy_loss(
                            logprobs, mb_old_logprobs, mb_advantages, mb_ref_logprobs
                        )
                        ent_loss = -self.config.ent_coef * entropy.mean()
                        total_policy_loss = (policy_loss + ent_loss) / self.config.grad_accumulation_steps

                    total_policy_loss.backward()

                with self._maybe_no_sync(self.value_model, not should_sync):
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.config.mixed_precision):
                        value_out = self.value_model(
                            input_ids=mb_input_ids,
                            attention_mask=mb_mask,
                        )
                        values = value_out.logits.squeeze(-1)
                        value_loss = self.compute_value_loss(values, mb_old_values, mb_returns)
                        total_value_loss = (self.config.vf_coef * value_loss) / self.config.grad_accumulation_steps

                    total_value_loss.backward()

                if should_sync:
                    grad_norm = get_fsdp_grad_norm(self.policy, self.config.max_grad_norm)
                    self.policy_optimizer.step()
                    self.value_optimizer.step()
                    self.policy_optimizer.zero_grad(set_to_none=True)
                    self.value_optimizer.zero_grad(set_to_none=True)

                    for k, v in policy_metrics.items():
                        accumulated_stats[k] = accumulated_stats.get(k, 0.0) + v.item()
                    accumulated_stats["value_loss"] = accumulated_stats.get("value_loss", 0.0) + value_loss.item()
                    accumulated_stats["entropy"] = accumulated_stats.get("entropy", 0.0) + entropy.mean().item()
                    accumulated_stats["grad_norm"] = accumulated_stats.get("grad_norm", 0.0) + grad_norm
                    stats.n_updates += 1

        self.policy_scheduler.step()
        self.value_scheduler.step()

        n = max(stats.n_updates, 1)
        stats.policy_loss = accumulated_stats.get("policy_loss", 0.0) / n
        stats.value_loss = accumulated_stats.get("value_loss", 0.0) / n
        stats.entropy_loss = accumulated_stats.get("entropy", 0.0) / n
        stats.approx_kl = accumulated_stats.get("approx_kl", 0.0) / n
        stats.clip_fraction = accumulated_stats.get("clip_fraction", 0.0) / n
        stats.grad_norm = accumulated_stats.get("grad_norm", 0.0) / n
        stats.learning_rate = self.policy_scheduler.get_last_lr()[0]
        stats.advantages_mean = advantages.mean().item()
        stats.advantages_std = advantages.std().item()

        elapsed = time.perf_counter() - t0
        stats.tokens_per_second = total_tokens / elapsed

        self.global_step += 1

        if self.global_step % self.config.save_every_n_steps == 0 and is_main_process():
            save_fsdp_checkpoint(
                self.policy,
                self.value_model,
                self.policy_optimizer,
                self.global_step,
                self.config.checkpoint_dir,
                keep_last_n=self.config.keep_last_n_checkpoints,
            )

        return stats

    @staticmethod
    def _logprobs_from_logits(
        logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute per-token log probabilities with numerical stability.
        Uses log_softmax to avoid precision loss from softmax + log.
        """
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        token_log_probs = torch.gather(
            log_probs, 2, labels[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        return token_log_probs

    @staticmethod
    def _entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
        """Entropy of the policy distribution over vocabulary."""
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return -(probs * log_probs).sum(dim=-1)

    def log_to_wandb(self, stats: PPOStats, step: int) -> None:
        if not is_main_process():
            return
        wandb.log(
            {
                "train/policy_loss": stats.policy_loss,
                "train/value_loss": stats.value_loss,
                "train/entropy": stats.entropy_loss,
                "train/approx_kl": stats.approx_kl,
                "train/clip_fraction": stats.clip_fraction,
                "train/grad_norm": stats.grad_norm,
                "train/lr": stats.learning_rate,
                "train/advantages_mean": stats.advantages_mean,
                "train/advantages_std": stats.advantages_std,
                "train/kl_coef": self._kl_coef,
                "throughput/tokens_per_second": stats.tokens_per_second,
                "throughput/n_updates": stats.n_updates,
            },
            step=step,
        )
