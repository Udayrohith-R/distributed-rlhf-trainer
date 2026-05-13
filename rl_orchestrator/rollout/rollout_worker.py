"""
Async rollout workers using Ray for decoupled experience generation.

Separates experience collection from policy optimization, keeping GPUs
fully utilized during the generation phase. Supports batched inference
with dynamic KV-cache allocation and configurable generation strategies.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Dict, List, Optional, Tuple

import ray
import torch
import torch.nn as nn
from transformers import GenerationConfig

logger = logging.getLogger(__name__)


@dataclass
class RolloutConfig:
    # Generation
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 0
    repetition_penalty: float = 1.0
    do_sample: bool = True

    # Batching
    micro_batch_size: int = 8
    max_queue_size: int = 256
    prefetch_batches: int = 2

    # Workers
    n_rollout_workers: int = 4
    n_gpus_per_worker: float = 0.5  # Fractional GPU for inference workers

    # Timeouts
    worker_timeout_s: float = 300.0
    generation_timeout_s: float = 120.0


@dataclass
class Experience:
    """A single trajectory from a rollout worker."""
    input_ids: torch.Tensor
    response_ids: torch.Tensor
    attention_mask: torch.Tensor
    logprobs: torch.Tensor
    ref_logprobs: Optional[torch.Tensor]
    values: Optional[torch.Tensor]
    rewards: Optional[torch.Tensor]
    advantages: Optional[torch.Tensor]
    returns: Optional[torch.Tensor]
    metadata: Dict = field(default_factory=dict)


@ray.remote(num_gpus=0.5)
class RolloutWorker:
    """
    Ray remote actor for asynchronous trajectory generation.

    Each worker holds a copy of the policy (inference-only) and generates
    trajectories independently. Policy weights are periodically synced
    from the trainer via broadcast.
    """

    def __init__(
        self,
        worker_id: int,
        model_name_or_path: str,
        config: RolloutConfig,
        device: Optional[str] = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.worker_id = worker_id
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._generation_count = 0

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )
        self.model.eval()

        self._generation_config = GenerationConfig(
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k if config.top_k > 0 else None,
            repetition_penalty=config.repetition_penalty,
            do_sample=config.do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        logger.info(f"RolloutWorker {worker_id} initialized on {self.device}")

    def generate_batch(
        self,
        prompts: List[str],
    ) -> List[Dict]:
        """
        Generate responses for a batch of prompts.

        Returns list of dicts with input_ids, response_ids, logprobs.
        """
        encodings = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]
        prompt_len = input_ids.shape[1]

        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=self._generation_config,
                return_dict_in_generate=True,
                output_scores=True,
            )

        sequences = outputs.sequences
        response_ids = sequences[:, prompt_len:]

        # Compute per-token log probabilities from generation scores
        logprobs = self._compute_logprobs_from_scores(
            outputs.scores, response_ids
        )

        results = []
        for i in range(len(prompts)):
            resp_len = (response_ids[i] != self.tokenizer.eos_token_id).sum().item() + 1
            results.append({
                "input_ids": input_ids[i].cpu(),
                "response_ids": response_ids[i, :resp_len].cpu(),
                "attention_mask": attention_mask[i].cpu(),
                "logprobs": logprobs[i, :resp_len].cpu(),
            })

        self._generation_count += len(prompts)
        return results

    def sync_weights(self, state_dict: Dict) -> None:
        """
        Update local policy weights from trainer broadcast.
        Called periodically to keep rollout workers in sync with learner.
        """
        self.model.load_state_dict(
            {k: v.to(self.device) for k, v in state_dict.items()},
            strict=False,
        )
        logger.debug(f"Worker {self.worker_id}: weights synced")

    def health_check(self) -> Dict:
        return {
            "worker_id": self.worker_id,
            "device": self.device,
            "generation_count": self._generation_count,
            "status": "healthy",
        }

    @staticmethod
    def _compute_logprobs_from_scores(
        scores: Tuple[torch.Tensor, ...],
        response_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Stack generation scores into per-token log probabilities."""
        import torch.nn.functional as F
        stacked = torch.stack(scores, dim=1)  # [B, T, V]
        log_probs = F.log_softmax(stacked, dim=-1)
        token_logprobs = torch.gather(
            log_probs, 2, response_ids[:, :log_probs.shape[1]].unsqueeze(-1)
        ).squeeze(-1)
        return token_logprobs


class AsyncRolloutOrchestrator:
    """
    Manages a pool of RolloutWorkers, distributes prompts, and streams
    experiences back to the trainer.

    Decouples generation from optimization: workers generate continuously
    while the trainer updates. Uses an async queue to buffer experiences
    and prevent GPU idle time.
    """

    def __init__(self, config: RolloutConfig, model_name_or_path: str) -> None:
        self.config = config
        self._experience_queue: asyncio.Queue = asyncio.Queue(
            maxsize=config.max_queue_size
        )
        self._workers: List[RolloutWorker] = []
        self._model_name_or_path = model_name_or_path
        self._active = False

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        self._init_workers()

    def _init_workers(self) -> None:
        for i in range(self.config.n_rollout_workers):
            worker = RolloutWorker.options(
                num_gpus=self.config.n_gpus_per_worker
            ).remote(
                worker_id=i,
                model_name_or_path=self._model_name_or_path,
                config=self.config,
            )
            self._workers.append(worker)
        logger.info(f"Initialized {len(self._workers)} rollout workers")

    async def generate_experiences(
        self,
        prompt_iterator: AsyncIterator[List[str]],
        reward_fn,
        ref_model: Optional[nn.Module] = None,
    ) -> AsyncIterator[List[Experience]]:
        """
        Asynchronously generate experiences from prompts.

        Yields batches of Experience objects as workers complete generation.
        Distributes prompts round-robin across workers for load balancing.
        """
        worker_idx = 0
        pending_futures = []

        async for prompt_batch in prompt_iterator:
            # Distribute to workers round-robin
            worker = self._workers[worker_idx % len(self._workers)]
            worker_idx += 1

            future = worker.generate_batch.remote(prompt_batch)
            pending_futures.append(future)

            # Prefetch: yield when we have enough futures pending
            if len(pending_futures) >= self.config.prefetch_batches:
                ready, pending_futures = await asyncio.wait(
                    [asyncio.wrap_future(f.future()) for f in pending_futures],
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=self.config.generation_timeout_s,
                )

                for task in ready:
                    raw_experiences = await task
                    processed = await self._process_raw_experiences(
                        raw_experiences, reward_fn, ref_model
                    )
                    yield processed

        # Drain remaining futures
        for future in pending_futures:
            raw_experiences = await asyncio.wrap_future(future.future())
            processed = await self._process_raw_experiences(
                raw_experiences, reward_fn, ref_model
            )
            yield processed

    async def _process_raw_experiences(
        self,
        raw: List[Dict],
        reward_fn,
        ref_model: Optional[nn.Module],
    ) -> List[Experience]:
        """Score experiences and compute reference log probs."""
        experiences = []
        for item in raw:
            reward = await asyncio.get_event_loop().run_in_executor(
                None, reward_fn, item
            )
            ref_logprobs = None
            if ref_model is not None:
                with torch.inference_mode():
                    ref_logprobs = self._compute_ref_logprobs(ref_model, item)

            experiences.append(Experience(
                input_ids=item["input_ids"],
                response_ids=item["response_ids"],
                attention_mask=item["attention_mask"],
                logprobs=item["logprobs"],
                ref_logprobs=ref_logprobs,
                values=None,
                rewards=torch.tensor([reward], dtype=torch.float32),
                advantages=None,
                returns=None,
            ))
        return experiences

    def broadcast_weights(self, state_dict: Dict) -> None:
        """Broadcast updated policy weights to all rollout workers."""
        futures = [w.sync_weights.remote(state_dict) for w in self._workers]
        ray.get(futures)
        logger.debug("Weights broadcast complete")

    def health_check(self) -> List[Dict]:
        return ray.get([w.health_check.remote() for w in self._workers])

    def shutdown(self) -> None:
        for worker in self._workers:
            ray.kill(worker)
        logger.info("All rollout workers shut down")

    @staticmethod
    def _compute_ref_logprobs(
        ref_model: nn.Module, item: Dict
    ) -> torch.Tensor:
        import torch.nn.functional as F
        input_ids = item["input_ids"].unsqueeze(0)
        with torch.inference_mode():
            out = ref_model(input_ids=input_ids)
        log_probs = F.log_softmax(out.logits[:, :-1, :], dim=-1)
        token_logprobs = torch.gather(
            log_probs, 2, input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1).squeeze(0)
        return token_logprobs
