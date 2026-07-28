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
from .verify import replay


@dataclass
class TrainerConfig:
    network: str = "regtest"
    datadir: str = ""
    corpus: str = ""            # a local corpus file; otherwise fetched via the daemon
    device: str = "cuda"
    lr: float = 1e-3
    rounds: int = 0             # 0 means until stopped
    poll_seconds: float = 5.0

    def resolved_datadir(self) -> str:
        return self.datadir or os.path.join(
            os.path.expanduser("~"), ".rnet", self.network)


def _open_local(path: str, round_desc, tokenizer, log):
    """A corpus on this machine, using a cached index where one is valid.

    The index is a cache of the boundary scan, and scanning seven terabytes is
    hours. It is checked against the file's size and the round's target before
    it is believed — an index that could disagree with the corpus and win would
    be a way to make two nodes train on different text while agreeing on a root.
    """
    from ..dataset.corpus import LocalCorpus
    from ..dataset.index import CorpusIndex, build_index

    target = 1 << 20
    index_path = path + ".rnidx"
    index = CorpusIndex.load(index_path, path, target)
    if index is None:
        log("corpus    no usable index; scanning (this takes a while)")
        index = build_index(path, target)
        index.save(index_path)
    if index.root != round_desc.dataset_root:
        raise WorkerError(
            f"corpus: {path} indexes to {index.root.hex()[:16]}… and this round "
            f"pins {round_desc.dataset_root.hex()[:16]}…. That is a different "
            f"corpus, not a different copy of the same one.")
    log(f"corpus    {index.n_chunks:,} chunks, root verified against the round")
    return LocalCorpus.open(path, target, tokenizer, offsets=index.offsets)


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

    # The corpus is opened BEFORE the handshake and before a model is built. A
    # round that pins one and cannot read it must fail, and failing after
    # twenty minutes of GPU is not failing, it is wasting somebody's evening.
    corpus = None
    if round_desc.dataset_root != bytes(32):
        from ..dataset.corpus import LocalCorpus
        from ..dataset.tokenizer import load as load_tokenizer
        tokenizer = load_tokenizer(round_desc.tokenizer_hash)
        if config.corpus:
            log(f"corpus    {config.corpus} — scanning for chunk boundaries")
            corpus = _open_local(config.corpus, round_desc, tokenizer, log)
        else:
            log(f"corpus    fetched through the daemon "
                f"({round_desc.dataset_chunks:,} chunks)")

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

        if corpus is None and round_desc.dataset_root != bytes(32):
            from ..dataset.corpus import RemoteCorpus
            from ..dataset.tokenizer import load as load_tokenizer
            corpus = RemoteCorpus(client=client,
                                  n_chunks=round_desc.dataset_chunks,
                                  tokenizer=load_tokenizer(round_desc.tokenizer_hash))

        # Built once and reloaded from the assignment's base each round: a model
        # rebuilt from scratch every time would spend minutes deriving weights
        # it is about to overwrite.
        model = W.build(spec, numerics, genesis_hash, device=config.device,
                        grad_checkpointing=True)
        producer = Producer(spec)
        base_weights = W.save_weights(model)
        # The base of the round most recently completed, which is what a
        # challenge about that round has to be replayed against. One extra copy
        # of the weights, and the alternative is fetching them back over a
        # socket at the moment a challenge lands.
        verify_base = None
        done = 0

        while config.rounds == 0 or done < config.rounds:
            reply = client.next_work()
            if isinstance(reply, ipc.NoWork):
                log(f"waiting: {reply.reason}")
                time.sleep(max(0.1, reply.retry_ms / 1000.0))
                continue

            if isinstance(reply, ipc.Verify):
                started = time.time()
                answer = replay(model, spec, numerics, reply,
                                dataset_root=round_desc.dataset_root,
                                base_weights=verify_base, device=config.device,
                                lr=config.lr, corpus=corpus)
                # Replaying leaves the model wherever the target's round ended.
                # Put it back, or the next round would train from somebody
                # else's weights.
                W.load_weights(model, base_weights)
                client.verdict(answer)
                from ..consensus.objects import Verdict as Kind
                log(f"verdict {Kind(answer.verdict).name} on "
                    f"{reply.challenge_id.hex()[:16]}…"
                    + (f" — {answer.note}" if answer.note else "")
                    + f", {time.time() - started:.1f}s")
                continue

            if isinstance(reply, ipc.Apply):
                started = time.time()
                result = producer.apply(model, reply,
                                        numerics.contribution_format,
                                        base_weights)
                # The applied weights become the base for the next round, for
                # every worker that applied — which is all of them, because
                # DiLoCo's outer step is not the producer's privilege. The
                # weights the round STARTED from are what a challenge about it
                # will need.
                verify_base = base_weights
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
                device=config.device, corpus=corpus, on_step=progress)
            print()

            base_weights = result.base_weights
            payload = pack(result.payload, numerics.contribution_format)
            try:
                accepted = client.submit(
                    assignment_id=reply.assignment_id, payload=payload,
                    scale_exp=result.scale_exp,
                    value_count=int(result.payload.size),
                    final_loss=result.final_loss)
            except WorkerError as exc:
                # Being overtaken is the ordinary fate of the slowest card in a
                # round, not a fault. Exiting here tore down a long-running
                # contributor over a normal race, discarding the built model and
                # the weights a challenge would need, and needed an operator to
                # restart it.
                if "STALE_ASSIGNMENT" not in str(exc):
                    raise
                log(f"step {reply.outer_step}: overtaken while training — "
                    f"asking for current work")
                continue
            log(f"step {reply.outer_step}: submitted "
                f"{accepted.contribution_id.hex()[:16]}…, "
                f"loss {result.final_loss:.4f}, {time.time() - started:.0f}s")
            done += 1

    return 0
