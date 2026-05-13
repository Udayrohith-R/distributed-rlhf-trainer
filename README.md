# Distributed RLHF Training Loop for Multi-Agent Ecosystems

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-orange)
![Ray](https://img.shields.io/badge/Ray-2.10%2B-teal)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A production-grade distributed RLHF training framework for post-training large language models with PPO across multi-node GPU clusters.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Trainer (FSDP)                     │
│  ┌──────────────┐  ┌──────────────┐                 │
│  │ Policy (7B+) │  │ Value Model  │   AdamW (fused) │
│  │   FULL_SHARD │  │  FULL_SHARD  │   BF16 / FP32   │
│  └──────┬───────┘  └──────────────┘                 │
│         │ PPO Update (every N steps)                 │
│         ▼                                            │
│  ┌──────────────────────────────────────────┐       │
│  │         Rollout Orchestrator              │       │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ │       │
│  │  │ Worker 0 │ │ Worker 1 │ │ Worker N │ │       │
│  │  │  Ray GPU │ │  Ray GPU │ │  Ray GPU │ │       │
│  │  └──────────┘ └──────────┘ └──────────┘ │       │
│  │     Async rollouts, KV-cache managed      │       │
│  └──────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────┘
```

## Key Design Decisions

**FSDP over DDP**: At 7B+ parameters, full model replication per GPU is infeasible. FSDP shards parameters, gradients, and optimizer states across ranks — enabling training on hardware that DDP would OOM on.

**Ray for rollout decoupling**: Separating experience generation from policy optimization is critical for GPU utilization. Rollout workers run on fractional GPUs asynchronously; the trainer never waits idle for generation to complete.

**Adaptive KL penalty**: Rather than a fixed KL coefficient, we track the running KL divergence vs reference policy and adjust the coefficient dynamically to stay near the target. This stabilizes training when the policy starts diverging too quickly.

**Numerically stable log-probs**: We use `F.log_softmax` directly rather than `F.softmax + torch.log` to avoid floating point precision loss at the tails of the vocabulary distribution.

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
```

## Components

| Module | Description |
|---|---|
| `trainer/ppo_trainer.py` | PPO training loop with GAE, clipped surrogate objective, adaptive KL |
| `trainer/fsdp_utils.py` | FSDP wrapping, gradient clipping, checkpoint consolidation |
| `rollout/rollout_worker.py` | Ray remote actors for async trajectory generation |
| `utils/distributed.py` | `init_distributed`, `all_reduce_mean`, process group utilities |

## NCCL Tuning

For multi-node runs, set these environment variables to optimize collective communication:

```bash
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0          # Enable InfiniBand if available
export NCCL_NET_GDR_LEVEL=2       # GPU Direct RDMA
export NCCL_SOCKET_IFNAME=eth0    # Network interface
export TORCH_NCCL_BLOCKING_WAIT=1 # Surface hangs as errors
```

## Checkpointing

Checkpoints are saved using `FULL_STATE_DICT` mode — shards are consolidated to rank 0 before writing, so checkpoints are portable and can be loaded on different world sizes.

```python
# Resume from checkpoint
torchrun ... scripts/train.py --config configs/ppo_7b.yaml --resume checkpoints/step_0001000.pt
```

## License

MIT
