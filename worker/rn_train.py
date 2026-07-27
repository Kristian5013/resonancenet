#!/usr/bin/env python3
"""The trainer. Run this to contribute to the network.

    python3 worker/rn_train.py

It connects to the node running on this machine, asks what to train, trains it,
and submits the result. Then it does that again, for as long as it is left
running.

Nothing here decides anything. Which weights to start from, which text to train
on, how many steps to take, when to stop — all of it comes from the node, which
gets it from consensus. That is the point: a worker that could choose its own
data or its own schedule would be a worker whose contribution nobody could check.

WHAT YOU NEED
    A node:     ./build/rnetd --network main --worker-id N --datadir ~/.rnet/main
    A GPU:      24 GB for round 0. Less will not fit at seq_len 16384.
    PyTorch:    with CUDA. See docs/training.md.

WHAT IT DOES, EACH ROUND
    1. Asks the node for an assignment: which step, which weights.
    2. Derives its batches from protocol state — a chunk of the corpus per inner
       step, fetched through the node and verified against dataset_root before
       a single token of it is trained on.
    3. Runs the round's inner steps.
    4. Sends back the difference between the weights it started with and the
       weights it ended with, quantised to one byte per parameter.
    5. If the node elects it producer, applies the aggregated update and reports
       the resulting weights hash.
"""

from __future__ import annotations

# Must be set before torch initialises CUDA, which happens on import.
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import hashlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "worker"))

import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402
from tokenizers import Tokenizer                      # noqa: E402

from rn_worker import inner_loop                      # noqa: E402
from rn_worker.consensus import canon, genesis        # noqa: E402
from rn_worker.corpus import RemoteCorpus             # noqa: E402
from rn_worker.ipc.client import DaemonClient, AnchorMismatch, IpcError  # noqa: E402
from rn_worker.model_llama import LlamaModel          # noqa: E402


def load_anchors(datadir: Path, network: str) -> tuple[bytes, bytes]:
    """The two hashes this worker will hold the daemon to.

    The genesis hash is compiled into this package; the policy hash comes from the
    artifact the node emitted. Both are checked against what the daemon sends
    during the handshake, and a mismatch ends the connection — a worker training
    under different rules produces contributions nobody can use and cannot be
    told apart from one that is cheating.
    """
    g = bytes.fromhex(genesis.GENESIS_HASH[network])
    policy_path = datadir / f"{network}.rnpol"
    if not policy_path.exists():
        raise SystemExit(
            f"{policy_path} not found. The node writes it; run:\n"
            f"    ./build/rnet-tool genesis-emit --network {network} --out {datadir}"
        )
    p = hashlib.sha3_256(policy_path.read_bytes()).digest()
    if p != bytes.fromhex(genesis.POLICY_HASH[network]):
        raise SystemExit(
            f"{policy_path} hashes to {p.hex()}, this build expects "
            f"{genesis.POLICY_HASH[network]}. The node and the worker are on different "
            f"commits."
        )
    return g, p


def load_tokenizer(round_desc, genesis_dir: Path) -> Tokenizer:
    path = genesis_dir / "tokenizer_round0.json"
    if not path.exists():
        raise SystemExit(f"{path} not found; emit the genesis artifacts first")
    actual = hashlib.sha3_256(path.read_bytes()).digest()
    if actual != round_desc.tokenizer_hash:
        raise SystemExit(
            f"{path} hashes to {actual.hex()}, the round pins "
            f"{round_desc.tokenizer_hash.hex()}. Training with it would produce token ids "
            f"that mean different strings than the network agreed on."
        )
    return Tokenizer.from_file(str(path))


def optimizer_factory(model, lr):
    """One Adafactor per parameter, stepped inside backward.

    Gradients are freed one at a time rather than all at once, which is what keeps
    full-parameter training of a 400M model inside a consumer card alongside a
    16384-token context.
    """
    from transformers.optimization import Adafactor

    optimizers = {}
    handles = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        opt = Adafactor([param], lr=lr, relative_step=False, scale_parameter=False,
                        warmup_init=False)
        optimizers[param] = opt

        def step(p, *, _opt=opt):
            _opt.step()
            _opt.zero_grad(set_to_none=True)

        handles.append(param.register_post_accumulate_grad_hook(step))

    class Session:
        def close(self):
            for h in handles:
                h.remove()

    return Session()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--network", default="main")
    ap.add_argument("--datadir", type=Path, default=Path.home() / ".rnet" / "main")
    ap.add_argument("--rounds", type=int, default=0, help="0 means run until stopped")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("no CUDA device. Training round 0 needs one; see docs/training.md")

    genesis_dir = ROOT / "share" / "genesis"
    round_desc = genesis.load_network(args.network, str(genesis_dir))
    policy = genesis.load_policy_for_network(args.network, str(genesis_dir))
    g, p = load_anchors(args.datadir, args.network)

    if round_desc.dataset_root == bytes(32):
        raise SystemExit(
            "this round has no corpus pinned (dataset_root is all-zero), so there is "
            "nothing to train on yet"
        )

    socket_path = args.datadir / "rnet.sock"
    if not socket_path.exists():
        raise SystemExit(
            f"no node at {socket_path}. Start one:\n"
            f"    ./build/rnetd --network {args.network} --worker-id N --datadir {args.datadir}"
        )

    tokenizer = load_tokenizer(round_desc, genesis_dir)

    print(f"network   {args.network}, round {round_desc.round_id}")
    print(f"model     {round_desc.model.parameter_count():,} parameters, "
          f"seq_len {round_desc.model.seq_len}")
    print(f"corpus    {round_desc.dataset_root.hex()[:16]}…")
    print(f"schedule  {policy.inner_steps} inner steps per round")

    with DaemonClient(str(socket_path), g, p) as client:
        try:
            client.hello()
        except AnchorMismatch as exc:
            raise SystemExit(
                f"the node is running different consensus rules than this worker: {exc}"
            ) from exc
        print(f"worker id {client.worker_id} (assigned by the node, not chosen here)")

        inner_loop.enable_determinism(seed=0)

        from types import SimpleNamespace
        m = round_desc.model
        cfg = SimpleNamespace(
            d_model=m.d_model, n_layers=m.n_layers, n_heads=m.n_heads,
            n_kv_heads=m.n_kv_heads, d_ff=m.d_ff, vocab_size=m.vocab_size,
            qk_norm=m.qk_norm, rope_theta=float(m.rope_theta), ctx_train=m.seq_len,
            tie_embeddings=m.tie_embeddings,
        )
        # bf16 with activation checkpointing: what puts this model inside 24 GB at
        # seq_len 16384, measured rather than assumed.
        model = LlamaModel(cfg, grad_checkpointing=True).to(args.device, dtype=torch.bfloat16)

        corpus = RemoteCorpus(client, round_desc.dataset_chunks, tokenizer)

        done = 0
        while args.rounds == 0 or done < args.rounds:
            assignment = client.get_assignment()
            if assignment.get("kind") == "none":
                time.sleep(5)
                continue

            step = assignment["outer_step"]
            started = time.time()
            print(f"\nstep {step}: training {policy.inner_steps} inner steps")

            # A round is a quarter of an hour. Without this the terminal is silent
            # for all of it, and silence is indistinguishable from a hang — which
            # is exactly what a first-time joiner will conclude.
            def progress(i, loss, _started=started, _total=policy.inner_steps):
                if i % 5 and i + 1 != _total:
                    return
                done = i + 1
                elapsed = time.time() - _started
                rate = elapsed / done
                left = (_total - done) * rate
                bar = int(20 * done / _total)
                print(f"\r  [{'#' * bar}{'.' * (20 - bar)}] {done:>3}/{_total}  "
                      f"loss {loss:6.3f}  {rate:4.1f}s/step  {left / 60:4.1f} min left",
                      end="", flush=True)

            result = inner_loop.run_inner_loop(
                model, round_desc.model, optimizer_factory, corpus,
                round_desc.dataset_root, round_desc.round_id, client.worker_id, step,
                policy.inner_steps, policy.micro_batch, round_desc.model.seq_len,
                args.lr, args.device, on_step=progress,
            )
            print()

            client.submit_contribution(assignment["assignment_id"], result["payload"],
                                       result["scale_exp"])
            print(f"step {step}: submitted, loss {result['final_loss']:.4f}, "
                  f"{time.time() - started:.0f}s")
            done += 1

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nstopped")
