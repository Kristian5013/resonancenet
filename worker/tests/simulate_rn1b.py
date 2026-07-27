#!/usr/bin/env python3
"""The local simulation on the real RN-1B model.

Answers what regtest cannot: whether the protocol holds at a billion parameters,
with the memory profile a real worker has (bf16, activation checkpointing,
deterministic attention) and with int8 quantisation and momentum in the loop.

Uses the `test` network: same RN-1B model as main, lighter schedule (50 inner
steps rather than 250), which is what a testnet is for.

    python worker/tests/simulate_rn1b.py --workers 4 --rounds 1 --challenge
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "worker"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rn_worker.consensus import genesis  # noqa: E402
from rn_worker.inner_loop import PoisonPolicy  # noqa: E402
from rn_worker.simulation import LocalNetwork, WorkerSpec  # noqa: E402

GENESIS_DIR = os.path.join(ROOT, "share", "genesis")
CORPUS = os.path.join(ROOT, "worker", "data", "rn1b_tokens.bin")


def load_corpus(vocab_size: int, n_tokens_cap: int) -> np.ndarray:
    """Memory-maps the tokenized corpus if it fits the round's vocabulary.

    A corpus tokenized with a different vocabulary is refused rather than used or
    quietly replaced. Using it indexes past the end of the embedding table and
    dies as a CUDA device-side assert, which names neither the corpus nor the
    vocabulary — that is how a 128k-vocabulary file survived the move to 32k and
    turned a simulation into an unexplained crash. Replacing it silently would be
    worse: the run would train on random numbers and report a loss.
    """
    if os.path.exists(CORPUS):
        tokens = np.memmap(CORPUS, dtype=np.uint32, mode="r")
        sample = tokens[: min(len(tokens), 10_000_000)]
        highest = int(sample.max())
        if highest >= vocab_size:
            raise SystemExit(
                f"{CORPUS} holds token id {highest}, and this round's vocabulary is "
                f"{vocab_size}. That file was tokenized for a different round. Delete it, or "
                f"rebuild it with worker/tools/build_corpus.py against the current tokenizer."
            )
        print(f"corpus: {CORPUS} ({len(tokens)/1e6:.0f}M tokens, max id {highest})")
        return tokens[: min(len(tokens), n_tokens_cap)]
    print(f"corpus: synthetic, {vocab_size}-token vocabulary (no {os.path.basename(CORPUS)})")
    rng = np.random.default_rng(1234)
    return rng.integers(0, vocab_size, size=n_tokens_cap, dtype=np.uint32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=None,
                    help="a text corpus (UTF-8, documents separated by a blank line). "
                         "Without it the run uses synthetic tokens and says so.")
    ap.add_argument("--network", default="test")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--challenge", action="store_true",
                    help="recompute every contribution and compare bit for bit")
    ap.add_argument("--poison", type=int, default=0,
                    help="worker id to make dishonest (0 = all honest)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--inner-steps", type=int, default=0,
                    help="RESEARCH ONLY: override the policy's inner_steps to shorten a run. "
                         "Bit-exactness either holds or it does not, so a short round answers "
                         "the same question as a long one; a real node never does this.")
    ap.add_argument("--out", default=os.path.join(ROOT, "worker", "runs", "rn1b_simulation.json"))
    args = ap.parse_args()

    round_descriptor = genesis.load_network(args.network, GENESIS_DIR)
    policy = genesis.load_policy_for_network(args.network, GENESIS_DIR)
    model = round_descriptor.model

    print(f"network {args.network}: d_model={model.d_model} layers={model.n_layers} "
          f"vocab={model.vocab_size} seq={model.seq_len}")
    print(f"policy: inner_steps={policy.inner_steps} micro_batch={policy.micro_batch} "
          f"challenge={policy.challenge_percent}%")
    print(f"workers: {args.workers}{f' (worker {args.poison} DISHONEST)' if args.poison else ''}")

    tokens = load_corpus(model.vocab_size, 20_000_000)
    if args.inner_steps:
        print(f"NOTE: inner_steps overridden {policy.inner_steps} -> {args.inner_steps} "
              "(research run, not protocol-conformant)")

    specs = []
    for i in range(1, args.workers + 1):
        poison = PoisonPolicy("delta") if i == args.poison else None
        specs.append(WorkerSpec(worker_id=i, poison=poison))

    started = time.time()
    corpus = None
    if args.corpus:
        from tokenizers import Tokenizer
        from rn_worker.corpus import LocalCorpus
        from rn_worker.consensus import genesis as gen
        round_desc = gen.load_network(args.network, GENESIS_DIR)
        tok_path = os.path.join(GENESIS_DIR, "tokenizer_round0.json")
        with open(tok_path, "rb") as fh:
            import hashlib
            actual = hashlib.sha3_256(fh.read()).digest()
        if actual != round_desc.tokenizer_hash:
            raise SystemExit(
                f"{tok_path} hashes to {actual.hex()}, the round pins "
                f"{round_desc.tokenizer_hash.hex()}. Training with it would produce token "
                f"ids that mean different strings than the network agreed on."
            )
        corpus = LocalCorpus(args.corpus, 1 << 20, Tokenizer.from_file(tok_path))
        print(f"corpus: {args.corpus} ({corpus.n_bytes/1e9:.2f} GB of text, "
              f"{corpus.n_chunks} chunks)")

    net = LocalNetwork(
        args.network, GENESIS_DIR, tokens, specs, corpus=corpus,
        lr=args.lr, verbose=True,
        # The real worker profile: bf16 plus activation checkpointing is what puts
        # a billion parameters inside 24 GB in deterministic mode.
        dtype=torch.bfloat16, grad_checkpointing=True,
    )
    if args.inner_steps:
        net.policy.inner_steps = args.inner_steps
    print(f"model built: {net.n_params/1e9:.2f}B parameters, "
          f"{torch.cuda.max_memory_allocated()/1e9:.1f} GB allocated")

    reports = []
    for _ in range(args.rounds):
        report = net.run_round(challenge_all=args.challenge)
        reports.append(report)
        print(f"round {report.outer_step}: {report.contributors} contributors, "
              f"loss {report.mean_loss:.4f}, "
              f"{report.matches} match / {report.mismatches} mismatch, "
              f"{report.seconds:.0f}s, peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
        if report.caught:
            print(f"  caught dishonest workers: {report.caught}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "network": args.network,
            "model": {"d_model": model.d_model, "n_layers": model.n_layers,
                      "vocab_size": model.vocab_size, "seq_len": model.seq_len},
            "n_params": net.n_params,
            "inner_steps": policy.inner_steps,
            "workers": args.workers,
            "poisoned_worker": args.poison,
            "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
            "total_seconds": time.time() - started,
            "rounds": [{"outer_step": r.outer_step, "contributors": r.contributors,
                        "mean_loss": r.mean_loss, "matches": r.matches,
                        "mismatches": r.mismatches, "caught": r.caught,
                        "seconds": r.seconds} for r in reports],
        }, f, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
