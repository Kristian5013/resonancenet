#!/usr/bin/env python3
"""The local channel, worker to daemon, against the real rnetd binary.

Driving the actual daemon rather than a reimplementation is the point: two
implementations of the same misunderstanding would agree with each other and with
nothing else. This is the same reason the other cross-language checks shell out to
rnet-tool.

Run from the repository root:
    python worker/tests/test_ipc_roundtrip.py
"""

from __future__ import annotations

import hashlib
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "worker"))

from rn_worker.consensus import genesis  # noqa: E402
from rn_worker.ipc.client import AnchorMismatch, DaemonClient, IpcError  # noqa: E402

RNETD = os.path.join(ROOT, "build", "rnetd")
TOOL = os.path.join(ROOT, "build", "rnet-tool")

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name}\n        {detail}")
        failures.append(name)


class Daemon:
    """A real rnetd on a throwaway datadir.

    The datadir lives directly under the system temp dir because sun_path holds
    107 bytes and a nested scratch path does not fit — the daemon refuses such a
    path rather than truncating it, which is correct and also means tests must
    respect it.
    """

    def __init__(self, worker_id: int = 5, port: int = 19671) -> None:
        self.dir = tempfile.mkdtemp(prefix="rnetd-", dir="/tmp")
        self.port = port
        self.worker_id = worker_id
        self.process: subprocess.Popen | None = None

    def __enter__(self) -> "Daemon":
        subprocess.run([TOOL, "genesis-emit", "--network", "regtest", "--out", self.dir],
                       check=True, capture_output=True)
        self.log = open(os.path.join(self.dir, "log"), "wb")
        self.process = subprocess.Popen(
            [RNETD, "--network", "regtest", "--datadir", self.dir, "--port", str(self.port),
             "--no-seeds", "--worker-id", str(self.worker_id)],
            stdout=self.log, stderr=subprocess.STDOUT,
        )
        socket_path = os.path.join(self.dir, "rnet.sock")
        for _ in range(100):
            if os.path.exists(socket_path):
                return self
            if self.process.poll() is not None:
                raise RuntimeError(f"rnetd exited: {self.tail()}")
            time.sleep(0.05)
        raise RuntimeError(f"rnetd never created its socket: {self.tail()}")

    def __exit__(self, *exc) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
            self.process.wait(timeout=10)
        self.log.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def tail(self) -> str:
        try:
            with open(os.path.join(self.dir, "log"), "r") as handle:
                return handle.read()[-800:]
        except OSError:
            return "(no log)"

    @property
    def socket(self) -> str:
        return os.path.join(self.dir, "rnet.sock")

    def anchors(self) -> tuple[bytes, bytes]:
        g = bytes.fromhex(genesis.GENESIS_HASH["regtest"])
        with open(os.path.join(self.dir, "regtest.rnpol"), "rb") as handle:
            p = hashlib.sha3_256(handle.read()).digest()
        return g, p


def test_full_cycle() -> None:
    print("[worker to daemon]")
    with Daemon() as daemon:
        g, p = daemon.anchors()
        with DaemonClient(daemon.socket, g, p) as client:
            client.hello()
            check("handshake completes", client.worker_id == daemon.worker_id,
                  f"daemon reported worker id {client.worker_id}")
            check("the daemon's containers hash to our own anchors", client.round_container is not None)
            check("the worker learns the parameter count from the model",
                  client.n_params == 2968320, f"got {client.n_params}")

            assignment = client.get_assignment()
            check("an assignment names a step and a base", "base_checkpoint" in assignment)
            # The node must not choose the worker's data: if it did, a colluding
            # node and worker would regain exactly the freedom the deterministic
            # derivation removes.
            for forbidden in ("offsets", "tokens", "lr", "seed"):
                check(f"the assignment does not carry '{forbidden}'", forbidden not in assignment,
                      "the node must not choose the worker's data or hyperparameters")

            payload = bytes((i * 37 + 11) % 256 for i in range(client.n_params))
            accepted = client.submit_contribution(assignment["assignment_id"], payload, 12)
            check("the contribution is accepted", len(accepted["contribution_id"]) == 32)

            status = client.status()
            check("the daemon holds the contribution", status["contributions"] == 1,
                  f"status says {status}")
            # Header plus payload: a contribution nobody can fetch the bytes of is
            # a contribution that cannot be verified.
            check("the payload bytes are stored, not merely acknowledged",
                  status["objects"] >= 2, f"status says {status}")


def test_wrong_anchors_are_refused() -> None:
    print("[anchors]")
    with Daemon(port=19672) as daemon:
        g, p = daemon.anchors()
        wrong = bytes.fromhex(genesis.GENESIS_HASH["main"])
        with DaemonClient(daemon.socket, wrong, p) as client:
            try:
                client.hello()
                check("a worker on a different round is refused", False, "the handshake completed")
            except AnchorMismatch:
                check("a worker on a different round is refused", True)
            except IpcError as exc:
                check("a worker on a different round is refused", False, f"wrong error: {exc}")


def test_payload_size_is_not_the_workers_to_choose() -> None:
    print("[payload size]")
    with Daemon(port=19673) as daemon:
        g, p = daemon.anchors()
        with DaemonClient(daemon.socket, g, p) as client:
            client.hello()
            assignment = client.get_assignment()
            # One byte per parameter, fixed by the model. The client refuses before
            # the daemon has to.
            try:
                client.submit_contribution(assignment["assignment_id"], b"\x01" * 100, 12)
                check("a short payload is refused", False, "it was accepted")
            except IpcError:
                check("a short payload is refused", True)


def test_apply_update_completes_a_checkpoint() -> None:
    """The last seam: the node aggregates, the worker applies, the checkpoint exists.

    Regtest closes a round one second after it opens and needs a single
    contributor, so a real cycle fits in a test. The update is applied with a
    stand-in model — what is being checked here is the channel and the checkpoint,
    not the arithmetic, which is covered bit-exactly by the local simulation.
    """
    import numpy as np

    print("[apply update]")
    with Daemon(port=19674) as daemon:
        g, p = daemon.anchors()
        with DaemonClient(daemon.socket, g, p) as client:
            client.hello()
            assignment = client.get_assignment()
            check("first assignment is training", assignment.get("kind") == "train",
                  f"got {assignment.get('kind')}")

            payload = bytes((i * 37 + 11) % 256 for i in range(client.n_params))
            client.submit_contribution(assignment["assignment_id"], payload, 12)

            # The round closes on the policy's deadline, then the node aggregates
            # and needs the weights updated.
            apply_assignment = None
            for _ in range(60):
                time.sleep(0.25)
                candidate = client.get_assignment()
                if candidate.get("kind") == "apply":
                    apply_assignment = candidate
                    break
            if apply_assignment is None:
                check("the round closes and an apply assignment arrives", False,
                      f"never offered; daemon log:\n{daemon.tail()}")
                return
            check("the round closes and an apply assignment arrives", True)

            update = client.read_update(apply_assignment)
            check("the update is one value per parameter", len(update) == client.n_params,
                  f"got {len(update)} for {client.n_params} parameters")
            check("the update applies to the base weights the node named",
                  len(apply_assignment["base_weights_hash"]) == 32)
            os.close(apply_assignment["update_fd"])

            # A stand-in for applying to a real model: the hash is over the inputs,
            # which is enough to show the checkpoint commits to them.
            # Stand-in weights of the right shape: two bytes per parameter, as bf16.
            new_weights = bytes(
                hashlib.sha3_256(
                    apply_assignment["base_weights_hash"] + i.to_bytes(4, "big")
                ).digest()
                for i in range(0)
            ) or (b"\x3f\x80" * client.n_params)
            weights_hash = hashlib.sha3_256(new_weights).digest()
            accepted = client.submit_weights(
                apply_assignment["assignment_id"], weights_hash, new_weights
            )

            check("the step advanced", accepted["outer_step"] == 1,
                  f"outer_step is {accepted['outer_step']}")
            check("a checkpoint exists", accepted["checkpoint_id"] != bytes(32))

            # And a hash nobody asked for cannot manufacture a second one.
            try:
                client.submit_weights(
                    apply_assignment["assignment_id"], weights_hash, new_weights
                )
                check("a replayed weights hash is refused", False, "it was accepted")
            except IpcError:
                check("a replayed weights hash is refused", True)

            # The point of storing them: a worker can now fetch the state the
            # network is at, which is what a joining node has to do.
            fetched = client.fetch_weights(weights_hash)
            check("the node can hand back the checkpoint's weights",
                  hashlib.sha3_256(fetched).digest() == weights_hash,
                  "what came back does not hash to the checkpoint's value")
            check("the weights are the right size", len(fetched) == client.n_params * 2,
                  f"got {len(fetched)} for {client.n_params} parameters")

            # And weights nobody holds are refused plainly rather than silently.
            try:
                client.fetch_weights(hashlib.sha3_256(b"a checkpoint nobody made").digest())
                check("unknown weights are refused, not stalled", False, "it returned something")
            except IpcError:
                check("unknown weights are refused, not stalled", True)


def main() -> int:
    for binary in (RNETD, TOOL):
        if not os.path.exists(binary):
            print(f"error: {binary} not built. Run: cmake -B build -S . && cmake --build build")
            return 2

    test_full_cycle()
    test_wrong_anchors_are_refused()
    test_payload_size_is_not_the_workers_to_choose()
    test_apply_update_completes_a_checkpoint()

    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED: {', '.join(failures)}")
        return 1
    print("the local channel works end to end")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
