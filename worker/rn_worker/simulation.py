"""Local network simulation: the whole protocol on one machine, no transport.

Runs the complete cycle — genesis, schedule, N workers training in parallel,
quantisation, aggregation, outer step, checkpoint, challenge, replay, bit-exact
comparison — with every consensus rule enforced exactly as it would be over a
network. What is absent is only the pipe.

This is where two open questions get answered, and neither of them is a
networking question:

  1. Does DiLoCo converge on this model at M replicas? Published results stop
     well short of the sizes and the heterogeneity this protocol assumes.
  2. Does recomputation actually catch a dishonest update on a real model, with
     int8 quantisation and momentum in the loop? The earlier poison experiment
     used a toy model and a loss-based gate — this is the real mechanism.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .consensus import canon, diloco, genesis
from . import inner_loop
from .inner_loop import PoisonPolicy


class _OptimizerSession:
    """Owns the per-parameter optimizers and their hooks for exactly one run."""

    def __init__(self, optimizers, handles):
        self.optimizers = optimizers
        self._handles = handles

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.optimizers.clear()


@dataclass
class WorkerSpec:
    worker_id: int
    poison: PoisonPolicy | None = None
    determinism_class: int = 1


@dataclass
class RoundReport:
    outer_step: int
    contributors: int
    base_checkpoint: str
    new_checkpoint: str
    mean_loss: float
    challenges: int
    matches: int
    mismatches: int
    caught: list[int] = field(default_factory=list)
    seconds: float = 0.0


class LocalNetwork:
    """One process standing in for a whole network.

    Every worker keeps its own model instance, exactly as separate machines would;
    they share only what the protocol shares — the base checkpoint and the
    schedule.
    """

    def __init__(self, network: str, genesis_dir: str, tokens: np.ndarray, workers: list[WorkerSpec],
                 device: str = "cuda", lr: float = 1e-3, verbose: bool = True,
                 dtype: torch.dtype = torch.float32, grad_checkpointing: bool = False):
        self.round = genesis.load_network(network, genesis_dir)
        self.policy = genesis.load_policy_for_network(network, genesis_dir)
        self.tokens = tokens
        self.workers = workers
        self.device = device if torch.cuda.is_available() else "cpu"
        self.lr = lr
        self.verbose = verbose
        # A real worker runs bf16 with activation checkpointing — that is what makes
        # a billion-parameter model fit a consumer card in deterministic mode.
        self.dtype = dtype
        self.grad_checkpointing = grad_checkpointing

        inner_loop.enable_determinism(seed=0)
        self.model = self._build_model()
        self.n_params = sum(p.numel() for _, p in inner_loop.flat_parameters(self.model, self.round.model))

        # The dataset root binds the schedule to a specific corpus. In the real
        # network it comes from the published manifest.
        self.dataset_root = canon.sha3_256(b"local-simulation-corpus")

        self.chain: list[canon.CheckpointHeader | None] = []
        self.checkpoint_ids: list[bytes] = []
        self.outer_step = 0
        self.optimizer = diloco.OuterOptimizer(
            self.n_params, scale_exp=16,
            momentum_q16=self.policy.outer_momentum_q16,
            outer_lr_q16=self.policy.outer_lr_q16,
            nesterov=self.policy.nesterov,
        )
        # Momentum lives as an int64 array: a billion-element Python list would be
        # tens of gigabytes. Equivalent to OuterOptimizer's scalar buffer.
        self.momentum = np.zeros(self.n_params, dtype=np.int64)
        self.checkpoint_ids.append(inner_loop.weights_hash(self.model, self.round.model))

    def _build_model(self):
        from types import SimpleNamespace
        from .model_llama import LlamaModel

        m = self.round.model
        cfg = SimpleNamespace(
            d_model=m.d_model, n_layers=m.n_layers, n_heads=m.n_heads, n_kv_heads=m.n_kv_heads,
            d_ff=m.d_ff, vocab_size=m.vocab_size, qk_norm=m.qk_norm,
            rope_theta=float(m.rope_theta), ctx_train=m.seq_len, tie_embeddings=m.tie_embeddings,
        )
        return LlamaModel(cfg, grad_checkpointing=self.grad_checkpointing).to(self.device,
                                                                               dtype=self.dtype)

    # ------------------------------------------------------------------
    def _optimizer_factory(self, model, lr):
        """Layerwise Adafactor: one optimizer per parameter, stepped inside backward
        so gradients are freed one at a time. That is what keeps full-parameter
        training inside a single consumer GPU."""
        from transformers.optimization import Adafactor

        optimizers = {}
        for p in model.parameters():
            if not p.requires_grad:
                continue
            optimizers[p] = Adafactor([p], lr=lr, relative_step=False, scale_parameter=False,
                                      warmup_init=False)

        def step_and_free(param):
            opt = optimizers[param]
            opt.step()
            opt.zero_grad(set_to_none=True)
            # Must return None: the hook contract accepts only None or a Tensor.

        # Handles are returned so the caller can detach them. Reusing one model
        # across runs without removing hooks would leave the previous run's hooks
        # attached, stepping the optimizer once per accumulated hook — the update
        # would silently be wrong, and only a bit-exact comparison would reveal it.
        handles = [p.register_post_accumulate_grad_hook(step_and_free) for p in optimizers]
        return _OptimizerSession(optimizers, handles)

    def _base_weights(self) -> list[torch.Tensor]:
        """The round's starting weights, held on the CPU. Keeping a second model on
        the GPU would double the memory for no reason — at a billion parameters
        that is the difference between fitting a consumer card and not."""
        return [p.detach().to("cpu").clone() for _, p in inner_loop.flat_parameters(self.model, self.round.model)]

    def _restore(self, base: list[torch.Tensor]) -> None:
        with torch.no_grad():
            for (_, p), saved in zip(inner_loop.flat_parameters(self.model, self.round.model), base):
                p.copy_(saved.to(p.device))

    def _run_from_base(self, spec: WorkerSpec, base: list[torch.Tensor],
                       poison: PoisonPolicy | None) -> dict:
        inner_loop.enable_determinism(seed=0)      # identical starting conditions
        self._restore(base)
        result = inner_loop.run_inner_loop(
            self.model,
            self.round.model, self._optimizer_factory, self.tokens, self.dataset_root,
            self.round.round_id, spec.worker_id, self.outer_step,
            self.policy.inner_steps, self.policy.micro_batch, self.round.model.seq_len,
            self.lr, self.device, poison=poison,
        )
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return result

    def _train_one_worker(self, spec: WorkerSpec, base: list[torch.Tensor]) -> dict:
        """A worker's local run, from the shared base weights."""
        return self._run_from_base(spec, base, spec.poison)

    def _replay(self, spec: WorkerSpec, base: list[torch.Tensor]) -> dict:
        """A verifier reproducing a worker's run. Identical inputs, no poison —
        that is exactly the point: the verifier does what the worker SHOULD have."""
        return self._run_from_base(spec, base, None)

    # ------------------------------------------------------------------
    def run_round(self, challenge_all: bool = True) -> RoundReport:
        started = time.time()
        base_checkpoint = self.checkpoint_ids[-1]
        base = self._base_weights()

        # --- workers train locally ---
        submissions = []
        for spec in self.workers:
            result = self._train_one_worker(spec, base)
            header = canon_contribution_header(
                round_id=self.round.round_id, worker_id=spec.worker_id,
                outer_step=self.outer_step + 1, base_checkpoint=base_checkpoint,
                payload_hash=result["payload_hash"], n_params=result["n_params"],
                inner_steps=self.policy.inner_steps, scale_exp=result["scale_exp"],
            )
            submissions.append((spec, header, result))
            if self.verbose:
                print(f"  worker {spec.worker_id}: loss {result['final_loss']:.4f} "
                      f"scale 2^-{result['scale_exp']} clamped {result['clamped']}"
                      f"{' [DISHONEST]' if spec.poison else ''}")

        if len(submissions) < self.policy.min_contributors:
            raise RuntimeError(
                f"round abandoned: {len(submissions)} contributors < quorum "
                f"{self.policy.min_contributors}")

        # --- verification: recompute a sample and compare bit for bit ---
        challenges = matches = mismatches = 0
        caught: list[int] = []
        for spec, header, result in submissions:
            if not challenge_all:
                continue
            challenges += 1
            replayed = self._replay(spec, base)
            if replayed["payload_hash"] == result["payload_hash"]:
                matches += 1
            else:
                mismatches += 1
                caught.append(spec.worker_id)
                if self.verbose:
                    print(f"  MISMATCH worker {spec.worker_id}: "
                          f"claimed {result['payload_hash'].hex()[:16]} vs "
                          f"recomputed {replayed['payload_hash'].hex()[:16]}")

        # --- aggregate every contribution, then take one outer step ---
        contributions = [(spec.worker_id, result["scale_exp"], result["values"])
                         for spec, _, result in submissions]
        update, scale_exp, count = diloco.aggregate_and_step_chunked(
            contributions, self.n_params, self.momentum, self.optimizer.scale_exp,
            self.policy.outer_momentum_q16, self.policy.outer_lr_q16, self.policy.nesterov)
        self.optimizer.steps_taken += 1
        # Apply to the ROUND'S base weights: the model instance currently holds
        # whatever the last replay left behind, which is not the shared state.
        self._restore(base)
        inner_loop.apply_update(self.model, self.round.model, update, self.optimizer.scale_exp)

        self.outer_step += 1
        new_id = inner_loop.weights_hash(self.model, self.round.model)
        self.checkpoint_ids.append(new_id)

        return RoundReport(
            outer_step=self.outer_step,
            contributors=count,
            base_checkpoint=base_checkpoint.hex()[:16],
            new_checkpoint=new_id.hex()[:16],
            mean_loss=float(np.mean([r["final_loss"] for _, _, r in submissions])),
            challenges=challenges, matches=matches, mismatches=mismatches, caught=caught,
            seconds=time.time() - started,
        )


def canon_contribution_header(round_id: int, worker_id: int, outer_step: int,
                              base_checkpoint: bytes, payload_hash: bytes, n_params: int,
                              inner_steps: int, scale_exp: int) -> dict:
    """The header a worker would publish. Kept as a dict here because the Python
    side only ever consumes canon objects; the C++ node authors them."""
    return {
        "round_id": round_id, "worker_id": worker_id, "outer_step": outer_step,
        "base_checkpoint": base_checkpoint, "payload_hash": payload_hash,
        "n_params": n_params, "inner_steps": inner_steps, "scale_exp": scale_exp,
    }


def synthetic_corpus(n_tokens: int, vocab_size: int, seed: int = 1234) -> np.ndarray:
    """A deterministic corpus for regtest runs. Real runs memory-map a tokenized
    corpus whose Merkle root is pinned in the round descriptor."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, vocab_size, size=n_tokens, dtype=np.uint32)
