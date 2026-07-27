"""The trainer loop: ask, train, submit, repeat.

This is the program a contributor runs. It needs a GPU and it needs a daemon;
it decides nothing. Which weights to start from, which text to train on, how
many steps to take — all of it arrives from the daemon, which got it from
consensus, and a worker that could choose any of it would produce work nobody
could check.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np

from ..consensus import genesis as genesis_module
from ..diloco import inner
from ..diloco.quantize import pack
from ..model import weights as W
from . import ipc
from .client import AnchorMismatch, DaemonClient, WorkerError
from .produce import Producer


@dataclass
class TrainerConfig:
    network: str = "regtest"
    datadir: str = ""
    device: str = "cuda"
    lr: float = 1e-3
    rounds: int = 0             # 0 means until stopped
    poll_seconds: float = 5.0

    def resolved_datadir(self) -> str:
        return self.datadir or os.path.join(
            os.path.expanduser("~"), ".rnet", self.network)


def run(config: TrainerConfig, *, log=print) -> int:
    network = config.network
    genesis_module.verify_build(network)
    round_desc = genesis_module.round_descriptor(network)
    policy = genesis_module.policy_descriptor(network)
    genesis_hash = bytes.fromhex(genesis_module.GENESIS_HASH[network])

    import torch
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise WorkerError("worker: no CUDA device; pass --device cpu to train slowly")

    inner.enable_determinism(seed=0, device=config.device)
    if config.device == "cpu":
        inner.tune_cpu_threads()

    socket_path = os.path.join(config.resolved_datadir(), "worker.sock")
    spec, numerics = round_desc.model, round_desc.numerics

    log(f"network   {network}, round {round_desc.round_id}")
    log(f"model     {spec.parameter_count():,} parameters, seq_len {spec.seq_len}")
    log(f"arithmetic {numerics.describe()}")

    with DaemonClient(socket_path=socket_path, genesis_hash=genesis_hash,
                      policy_hash=bytes.fromhex(
                          genesis_module.POLICY_HASH[network])) as client:
        try:
            client.hello()
        except AnchorMismatch as exc:
            raise WorkerError(
                f"the daemon is running different consensus rules: {exc}") from exc
        log(f"worker id {client.worker_id} (assigned by the daemon, not chosen here)")

        # Built once and reloaded from the assignment's base each round: a model
        # rebuilt from scratch every time would spend minutes deriving weights
        # it is about to overwrite.
        model = W.build(spec, numerics, genesis_hash, device=config.device,
                        grad_checkpointing=True)
        producer = Producer(spec)
        base_weights = W.save_weights(model)
        done = 0

        while config.rounds == 0 or done < config.rounds:
            reply = client.next_work()
            if isinstance(reply, ipc.NoWork):
                log(f"waiting: {reply.reason}")
                time.sleep(max(0.1, reply.retry_ms / 1000.0))
                continue

            if isinstance(reply, ipc.Apply):
                started = time.time()
                result = producer.apply(model, reply,
                                        numerics.contribution_format,
                                        base_weights)
                # The applied weights become the base for the next round, for
                # every worker that applied — which is all of them, because
                # DiLoCo's outer step is not the producer's privilege.
                base_weights = W.save_weights(model)
                client.applied(reply.outer_step, result.weights_hash,
                               result.optimizer_state_hash)
                log(f"step {reply.outer_step}: applied, weights "
                    f"{result.weights_hash.hex()[:16]}…, "
                    f"{time.time() - started:.1f}s")
                continue

            started = time.time()
            log(f"\nstep {reply.outer_step}: {reply.inner_steps} inner steps "
                f"from {reply.base_weights_hash.hex()[:16]}…")

            def progress(i, loss, _total=reply.inner_steps, _t=started):
                if i % 5 and i + 1 != _total:
                    return
                at = i + 1
                rate = (time.time() - _t) / at
                bar = int(20 * at / _total)
                print(f"\r  [{'#' * bar}{'.' * (20 - bar)}] {at:>4}/{_total}  "
                      f"loss {loss:6.3f}  {rate:4.1f}s/step  "
                      f"{(_total - at) * rate / 60:5.1f} min left",
                      end="", flush=True)

            result = inner.run(
                model, spec, numerics,
                dataset_root=round_desc.dataset_root,
                round_id=reply.round_id, worker_id=client.worker_id,
                outer_step=reply.outer_step, inner_steps=reply.inner_steps,
                micro_batch=reply.micro_batch, lr=config.lr,
                device=config.device, on_step=progress)
            print()

            base_weights = result.base_weights
            payload = pack(result.payload, numerics.contribution_format)
            accepted = client.submit(
                assignment_id=reply.assignment_id, payload=payload,
                scale_exp=result.scale_exp, value_count=int(result.payload.size),
                final_loss=result.final_loss)
            log(f"step {reply.outer_step}: submitted "
                f"{accepted.contribution_id.hex()[:16]}…, "
                f"loss {result.final_loss:.4f}, {time.time() - started:.0f}s")
            done += 1

    return 0
