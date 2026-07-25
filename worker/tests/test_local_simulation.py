#!/usr/bin/env python3
"""End-to-end protocol test on one machine: genesis to verified checkpoint.

Proves, without a single line of transport code, that:

  * workers build the model from a hash-verified genesis artifact,
  * they train only on batches the protocol assigned them,
  * every contribution is aggregated (a sum, not a race),
  * an honest contribution is reproduced bit for bit by a verifier,
  * a dishonest one is caught by that same recomputation.

Run from the repository root:
    python worker/tests/test_local_simulation.py [--workers 4] [--rounds 2]
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "worker"))

from rn_worker.inner_loop import PoisonPolicy  # noqa: E402
from rn_worker.simulation import LocalNetwork, WorkerSpec, synthetic_corpus  # noqa: E402

GENESIS_DIR = os.path.join(ROOT, "share", "genesis")

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name}\n        {detail}")
        failures.append(name)


def build_network(workers: list[WorkerSpec], network: str = "regtest") -> LocalNetwork:
    from rn_worker.consensus import genesis as gen

    round_descriptor = gen.load_network(network, GENESIS_DIR)
    corpus = synthetic_corpus(200_000, round_descriptor.model.vocab_size)
    return LocalNetwork(network, GENESIS_DIR, corpus, workers, verbose=False)


def test_honest_round_is_reproducible() -> None:
    """The core claim: an honest worker's update is bit-exactly reproducible."""
    print("[honest round]")
    net = build_network([WorkerSpec(1), WorkerSpec(2), WorkerSpec(3)])
    report = net.run_round(challenge_all=True)

    check("every contribution was aggregated", report.contributors == 3,
          f"{report.contributors} of 3")
    check("all challenges were answered", report.challenges == 3, f"{report.challenges}")
    check("every honest update reproduced bit for bit",
          report.matches == 3 and report.mismatches == 0,
          f"{report.matches} match / {report.mismatches} mismatch")
    check("the checkpoint advanced", report.new_checkpoint != report.base_checkpoint)
    print(f"        round took {report.seconds:.1f}s, loss {report.mean_loss:.4f}")


def test_dishonest_worker_is_caught() -> None:
    """A worker that trains on data it was not assigned, or reports an update it
    did not compute, must be caught by recomputation."""
    print("[dishonest workers]")
    for kind in ("data", "delta"):
        net = build_network([
            WorkerSpec(1),
            WorkerSpec(2, poison=PoisonPolicy(kind)),
            WorkerSpec(3),
        ])
        report = net.run_round(challenge_all=True)
        check(f"{kind}-poisoned worker caught",
              report.mismatches == 1 and report.caught == [2],
              f"mismatches={report.mismatches} caught={report.caught}")
        check(f"{kind}: honest workers not accused",
              report.matches == 2, f"matches={report.matches}")


def test_multiple_rounds_advance_the_chain() -> None:
    """Momentum carries across rounds, so the second outer step must differ from
    the first even on identical contributions."""
    print("[multi-round]")
    net = build_network([WorkerSpec(1), WorkerSpec(2)])
    ids = [net.checkpoint_ids[0]]
    losses = []
    for _ in range(2):
        report = net.run_round(challenge_all=True)
        ids.append(net.checkpoint_ids[-1])
        losses.append(report.mean_loss)
        check(f"round {report.outer_step}: no false accusations", report.mismatches == 0)

    check("each round produced a distinct checkpoint", len(set(ids)) == len(ids))
    check("outer optimizer advanced", net.optimizer.steps_taken == 2,
          f"{net.optimizer.steps_taken}")
    print(f"        losses {[f'{v:.3f}' for v in losses]}")


def test_scaling_across_replicas(worker_counts: list[int]) -> None:
    """Throughput scales with replicas because contributions are summed, not
    raced. This records the loss at each M so the convergence-versus-replicas
    question can be answered with data rather than extrapolation."""
    print("[replica scaling]")
    for m in worker_counts:
        net = build_network([WorkerSpec(i + 1) for i in range(m)])
        report = net.run_round(challenge_all=False)
        check(f"M={m}: all {m} contributions aggregated", report.contributors == m,
              f"{report.contributors}")
        print(f"        M={m:2d}  loss {report.mean_loss:.4f}  {report.seconds:.1f}s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--scaling", type=int, nargs="*", default=[2, 4])
    args = ap.parse_args()

    if not os.path.exists(os.path.join(GENESIS_DIR, "regtest.rnet")):
        print("error: genesis artifacts missing. Run: ./build/rnet-tool genesis-emit "
              "--network regtest --out share/genesis")
        return 2

    test_honest_round_is_reproducible()
    test_dishonest_worker_is_caught()
    test_multiple_rounds_advance_the_chain()
    test_scaling_across_replicas(args.scaling)

    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED: {', '.join(failures)}")
        return 1
    print("local simulation: the full protocol works on one machine")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
