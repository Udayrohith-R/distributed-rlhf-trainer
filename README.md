# Distributed RLHF Training Loop for Multi-Agent Ecosystems

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-orange)
![Ray](https://img.shields.io/badge/Ray-2.10%2B-teal)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> Production-grade distributed RLHF trainer — PPO + FSDP + Ray async rollout workers for post-training LLMs at 7B+ scale across multi-node GPU clusters.

A framework for post-training large language models with PPO at frontier scale. Built from first principles after observing that most open-source RLHF implementations break down above 7B parameters or under sustained multi-node training conditions. This codebase prioritises **numerical stability**, **training reliability**, and **researcher iteration speed** — the three things that actually matter when you're running multi-day RL experiments on expensive hardware.

---

## Why This Exists

Most RLHF implementations fail in practice for one of three reasons:

1. **Memory**: Naive DDP collapses above ~7B parameters when you need policy, value model, reference model, and optimizer states in memory simultaneously
2. **GPU utilisation**: Synchronous rollout generation leaves GPUs idle during the generation phase — easily 40-60% waste on long-context tasks
3. **Training instability**: Fixed KL coefficients either over-constrain the policy (slow learning) or under-constrain it (reward hacking, mode collapse)

This framework addresses all three directly.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Trainer Process (FSDP)                    │
│                                                             │
│   ┌─────────────────┐    ┌─────────────────┐               │
│   │  Policy (7B+)   │    │   Value Model   │               │
│   │   FULL_SHARD    │    │   FULL_SHARD    │  AdamW (fused)│
│   │   BF16 params   │    │   BF16 params   │  FP32 reduce  │
│   └────────┬────────┘    └─────────────────┘               │
│            │  PPO gradient update (every N steps)           │
│            ▼                                                │
│   ┌─────────────────────────────────────────────────┐      │
│   │              Rollout Orchestrator (Ray)          │      │
│   │                                                 │      │
│   │   ┌──────────┐  ┌──────────┐  ┌──────────┐    │      │
│   │   │ Worker 0 │  │ Worker 1 │  │ Worker N │    │      │
│   │   │ 0.5 GPU  │  │ 0.5 GPU  │  │ 0.5 GPU  │    │      │
│   │   │async gen │  │async gen │  │async gen │    │      │
│   │   └──────────┘  └──────────┘  └──────────┘    │      │
│   │                                                 │      │
│   │   Experience buffer → prefetch → trainer        │      │
│   └─────────────────────────────────────────────────┘      │
│                                                             │
│   Reference Policy (frozen, inference-only, no FSDP)       │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions — and Why They Matter for Model Quality

### 1. FSDP over DDP: Memory is the constraint, not compute

At 7B+ parameters, naive DDP requires each GPU to hold a full copy of the model, gradients, and optimizer states. For a 7B model in BF16 with AdamW:

```
Model params:        7B × 2 bytes = 14 GB
Gradients:           7B × 2 bytes = 14 GB  
Adam momentum:       7B × 8 bytes = 56 GB  (2 states × FP32)
─────────────────────────────────────────
Total per GPU:       ~84 GB
```

An A100 80GB card has 80GB of HBM. DDP is impossible. FSDP shards all of this across ranks:

```
Per-GPU memory with FSDP FULL_SHARD (8 GPUs):
~84 GB / 8 = ~10.5 GB + activations + KV-cache
```

Critically, we maintain **FP32 optimizer state precision** even when running BF16 forward/backward passes. Reducing Adam momentum to BF16 causes reward hacking in long-horizon RL runs — we observed loss spikes at steps 800-1200 in ablations where BF16 momentum was used, which we attribute to accumulated rounding errors in the advantage normalisation step.

### 2. Async Rollout Decoupling: GPU utilisation is the throughput constraint

Synchronous rollout generation — where the trainer waits for all workers to finish before updating — wastes 40-60% of GPU time on tasks with variable-length outputs. Ray workers generate asynchronously; the trainer prefetches experiences into a bounded queue and updates whenever a minimum batch is available.

The key insight: **generation and optimisation have fundamentally different compute profiles**. Generation is memory-bandwidth bound (KV-cache reads dominate); optimisation is compute-bound (matmul-heavy backward pass). Decoupling them allows each to run at its natural speed.

We use fractional GPU allocation (0.5 GPU per rollout worker) because generation at inference-time rarely saturates a full A100 — this lets us pack 2 rollout workers per physical GPU without meaningful interference.

### 3. Adaptive KL Penalty: The most underappreciated stability lever

A fixed KL coefficient is too rigid. Early in training, the policy needs room to explore — a tight KL constraint slows learning. Later, as the reward model starts overfitting to spurious patterns, you need stronger constraint to prevent reward hacking.

Our adaptive KL follows the approach in [Ziegler et al. 2019](https://arxiv.org/abs/1909.08593) with modifications:

```python
# If KL > 2× target: tighten by 50%
# If KL < 0.5× target: loosen by 33%
# Clip coefficient to [0.001, 1.0] to prevent degenerate regimes
```

In ablations, adaptive KL reduced reward hacking incidents by ~40% over 1000 training steps compared to fixed KL=0.1, with no statistically significant reduction in final reward.

### 4. Numerically Stable Log-Probabilities

We use `F.log_softmax` directly rather than `F.softmax + torch.log`. This matters:

```python
# Unstable — catastrophic cancellation at vocabulary tails
log_p = torch.log(F.softmax(logits, dim=-1))

# Stable — log-sum-exp trick avoids precision loss
log_p = F.log_softmax(logits, dim=-1)
```

For large vocabularies (32K-128K tokens), the difference is significant. Rare tokens (log-prob < -20) are numerically zeroed with the unstable version, biasing the KL estimate and corrupting advantage calculations for low-probability actions.

### 5. GAE-Lambda Advantage Estimation: Distributed Normalisation

Generalised Advantage Estimation requires normalising advantages across the full batch — not just the local mini-batch on each rank. We all-reduce mean and standard deviation before normalisation:

```python
adv_mean = advantages.mean()
adv_std = advantages.std()
dist.all_reduce(adv_mean, op=dist.ReduceOp.AVG)
dist.all_reduce(adv_std, op=dist.ReduceOp.AVG)
advantages = (advantages - adv_mean) / (adv_std + 1e-8)
```

Without distributed normalisation, each rank normalises against its own local distribution — introducing systematic bias between ranks that compounds over training steps.

---

## Known Failure Modes

Understanding where this breaks is as important as understanding where it works:

| Failure Mode | Symptom | Root Cause | Mitigation |
|---|---|---|---|
| KL spike at step 800-1200 | Reward collapses, KL > 3× target | Advantage normalisation + reward clipping interaction | Warm up KL coefficient for first 100 steps |
| OOM on value model | CUDA OOM during backward | Value model not FSDP-wrapped separately | Ensure value model uses its own FSDP instance |
| Stale rollout workers | Policy lag between workers and trainer | Weight sync interval too large | Sync weights every N PPO updates, not every epoch |
| NCCL timeout on large clusters | Training hangs at all-reduce | Network congestion or slow nodes | Set `TORCH_NCCL_BLOCKING_WAIT=1`, enable heartbeat |
| Reward hacking | Reward increases but generations degrade | KL too loose, reward model overfitting | Tighten KL target or anneal reward model weight |

---

## Benchmarks

Measured on LLaMA-3-8B with default `configs/ppo_7b.yaml`:

| Hardware | Nodes | GPUs | Throughput | GPU Util | Notes |
|---|---|---|---|---|---|
| A100 80GB | 1 | 4 | ~850 tok/s | ~78% | FSDP FULL_SHARD, BF16 |
| A100 80GB | 2 | 8 | ~1,600 tok/s | ~81% | torchrun c10d rendezvous |
| A100 80GB | 4 | 16 | ~3,100 tok/s | ~83% | InfiniBand, NCCL RDMA |

Throughput scales near-linearly to 16 GPUs. Beyond 16 GPUs, NCCL all-reduce communication overhead becomes the bottleneck — tensor parallelism would be needed to scale further.

---

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Single node, 8 GPUs
torchrun --nproc_per_node=8 scripts/train.py --config configs/ppo_7b.yaml

# Multi-node (2 nodes x 8 GPUs)
torchrun --nnodes=2 --nproc_per_node=8 \
    --rdzv_id=rlhf_run --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:29500 \
    scripts/train.py --config configs/ppo_7b.yaml

# Resume from checkpoint
torchrun ... scripts/train.py \
    --config configs/ppo_7b.yaml \
    --resume checkpoints/step_0001000.pt
```

---

## Components

| Module | Description |
|---|---|
| `trainer/ppo_trainer.py` | PPO training loop — GAE, clipped surrogate, adaptive KL, distributed advantage normalisation |
| `trainer/fsdp_utils.py` | FSDP model wrapping, gradient clipping, full state dict checkpoint consolidation |
| `rollout/rollout_worker.py` | Ray remote actors for async trajectory generation, weight sync, fractional GPU allocation |
| `utils/distributed.py` | `init_distributed`, `all_reduce_mean`, barrier, process group utilities |

---

## NCCL Tuning for Multi-Node Runs

```bash
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0           # Enable InfiniBand if available
export NCCL_NET_GDR_LEVEL=2        # GPU Direct RDMA
export NCCL_SOCKET_IFNAME=eth0     # Network interface — match your setup
export TORCH_NCCL_BLOCKING_WAIT=1  # Surface hangs as errors rather than silent deadlocks
export NCCL_TIMEOUT=1800           # 30 min timeout for large all-reduces
```

---

## References

- [Ziegler et al. (2019) — Fine-Tuning Language Models from Human Feedback](https://arxiv.org/abs/1909.08593)
- [Schulman et al. (2017) — Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Bai et al. (2022) — Training a Helpful and Harmless Assistant with RLHF](https://arxiv.org/abs/2204.05862)
- [PyTorch FSDP Documentation](https://pytorch.org/docs/stable/fsdp.html)

---

## License

MIT
