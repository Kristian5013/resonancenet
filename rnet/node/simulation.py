"""A whole round, start to finish, in one process.

Genesis to a checkpoint the chain accepts: several workers train from the same
weights on data none of them chose, their differences are aggregated in
integers, one elected producer applies the outer step, and the result is a
checkpoint every other node can check by recomputing.

WHAT THIS IS FOR. It is the honest end-to-end test — no network, no daemon, no
sockets, but every consensus decision the real thing makes, made the same way by
the same code. When this passes, what is left between here and a live network is
transport, not protocol.

WHAT IT IS NOT. It is not a claim that the model learns. On regtest there is no
corpus and the windows are synthetic, so the loss wanders around chance and is
supposed to. What it demonstrates is that the arithmetic closes: the same
contributions produce the same checkpoint on every node that aggregates them,
whatever order they arrived in.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..consensus import genesis as genesis_module
from ..consensus.init import hash_of_values
from ..consensus.objects import CheckpointHeader, ContributionHeader
from ..consensus.params import NETWORKS
from ..diloco import inner
from ..diloco.aggregate import Contribution, average_contributions, contribution_root
from ..diloco.chain import Chain, Outcome
from ..diloco.outer import OuterOptimizer, apply_update
from ..model import weights as W
from ..model.layout import layout


@dataclass
class WorkerResult:
    worker_id: int
    contribution: Contribution
    header: ContributionHeader
    final_loss: float
    seconds: float


@dataclass
class RoundReport:
    outer_step: int
    checkpoint: CheckpointHeader
    outcome: Outcome
    workers: list[WorkerResult]
    weights_hash: bytes
    optimizer_state_hash: bytes
    seconds: float
    losses: list[float] = field(default_factory=list)


class Simulation:
    """One network, one set of weights, as many rounds as asked for."""

    def __init__(self, network: str, *, device: str = "cpu", lr: float = 1e-3):
        genesis_module.verify_build(network)
        self.network = network
        self.round_desc, self.policy = NETWORKS[network]
        self.genesis_hash = bytes.fromhex(genesis_module.GENESIS_HASH[network])
        self.device = device
        self.lr = lr

        self.spec = self.round_desc.model
        self.numerics = self.round_desc.numerics

        # The weights every worker starts from, held as canonical words so the
        # producer's arithmetic is over the same representation the network
        # agreed on rather than over one node's tensors.
        self.weights = {
            name: words for name, words in
            W.save_weights(W.build(self.spec, self.numerics, self.genesis_hash,
                                   device=device, grad_checkpointing=False)).items()
        }
        self.order = [t.name for t in layout(self.spec)]
        self.optimizer = OuterOptimizer(self.policy.outer_momentum_q16,
                                        self.policy.outer_lr_q16,
                                        self.policy.nesterov)

        first = CheckpointHeader(
            round_id=self.round_desc.round_id, outer_step=0, parent=bytes(32),
            weights_hash=hash_of_values(self.spec, self.weights),
            optimizer_state_hash=self.optimizer.state_hash(),
            contribution_root=bytes(32), producer_id=0, timestamp_ms=0)
        self.chain = Chain(first, retained=self.policy.retained_checkpoints)

    # -- the parts ---------------------------------------------------------

    def train_one(self, worker_id: int, outer_step: int) -> WorkerResult:
        """One worker's round. Starts from the agreed weights, every time."""
        model = W.build(self.spec, self.numerics, self.genesis_hash,
                        device=self.device, grad_checkpointing=False)
        W.load_weights(model, self.weights)

        started = time.monotonic()
        result = inner.run(
            model, self.spec, self.numerics,
            dataset_root=self.round_desc.dataset_root,
            round_id=self.round_desc.round_id, worker_id=worker_id,
            outer_step=outer_step, inner_steps=self.policy.inner_steps,
            micro_batch=self.policy.micro_batch, lr=self.lr, device=self.device)
        seconds = time.monotonic() - started

        base = self.chain.head
        header = ContributionHeader(
            round_id=self.round_desc.round_id, outer_step=outer_step,
            worker_id=worker_id, assignment_id=outer_step * 1000 + worker_id,
            base_checkpoint=base.id,
            base_weights_hash=base.header.weights_hash,
            payload_hash=_payload_hash(result.payload, result.scale_exp),
            payload_bytes=int(result.payload.size),
            scale_exp=result.scale_exp,
            contribution_format=self.numerics.contribution_format)
        return WorkerResult(
            worker_id=worker_id,
            contribution=Contribution(result.payload, result.scale_exp,
                                      self.numerics.contribution_format, worker_id),
            header=header, final_loss=result.final_loss, seconds=seconds)

    def produce(self, results: list[WorkerResult], outer_step: int,
                producer_id: int) -> CheckpointHeader:
        """Aggregate, step, apply, and commit to what came out.

        Everything here is integer arithmetic over the contributions, so any
        node holding the same set reaches the same checkpoint — which is what
        makes producing a job anyone can check rather than a job anyone has to
        be trusted with.
        """
        mean, exp = average_contributions([r.contribution for r in results])
        update, exp = self.optimizer.step(mean, exp)

        flat = np.concatenate([_to_float(self.weights[n]) for n in self.order])
        moved = apply_update(flat, update, exp)

        at = 0
        from ..consensus.init import float32_to_bf16
        for tensor in layout(self.spec):
            self.weights[tensor.name] = float32_to_bf16(
                moved[at:at + tensor.numel].astype(np.float32))
            at += tensor.numel

        parent = self.chain.head
        return CheckpointHeader(
            round_id=self.round_desc.round_id, outer_step=outer_step,
            parent=parent.id,
            weights_hash=hash_of_values(self.spec, self.weights),
            optimizer_state_hash=self.optimizer.state_hash(),
            contribution_root=contribution_root([r.header.id for r in results]),
            producer_id=producer_id, timestamp_ms=0)

    # -- the whole thing ---------------------------------------------------

    def run_round(self, n_workers: int, *, on_worker=None) -> RoundReport:
        outer_step = self.chain.height + 1
        started = time.monotonic()

        results = []
        for worker_id in range(1, n_workers + 1):
            result = self.train_one(worker_id, outer_step)
            results.append(result)
            if on_worker is not None:
                on_worker(result)

        if len(results) < self.policy.min_contributors:
            raise RuntimeError(
                f"round: {len(results)} contributors, policy needs "
                f"{self.policy.min_contributors} — the round cannot close")

        checkpoint = self.produce(results, outer_step, producer_id=results[0].worker_id)
        outcome = self.chain.add(checkpoint)
        return RoundReport(
            outer_step=outer_step, checkpoint=checkpoint, outcome=outcome,
            workers=results, weights_hash=checkpoint.weights_hash,
            optimizer_state_hash=checkpoint.optimizer_state_hash,
            seconds=time.monotonic() - started,
            losses=[r.final_loss for r in results])


def _payload_hash(payload: np.ndarray, scale_exp: int) -> bytes:
    import hashlib
    return hashlib.sha3_256(
        scale_exp.to_bytes(8, "big", signed=True)
        + payload.astype(">i8").tobytes()).digest()


def _to_float(words: np.ndarray) -> np.ndarray:
    from ..consensus.init import bf16_to_float32
    return bf16_to_float32(words).astype(np.float64)
