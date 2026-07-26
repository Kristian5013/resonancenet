# Running a node and a worker

There is no `rnet-train` command. The worker is a Python library —
`worker/rn_worker/` — and the things that run it today are test harnesses that
happen to be complete end-to-end runs. This page is what actually works, in the
order worth trying it.

Everything here uses **regtest**: a 2,968,320-parameter model with a one-second
round deadline. A full cycle takes seconds on a CPU. Main is RN-1B and needs a
24 GB GPU and ten minutes per round.

---

## 0. Build and check

```bash
cmake -B build -S .
cmake --build build -j"$(nproc)"
./build/rnet_tests
```

The last two lines must read:

```
313 passed, 0 failed
suite: consensus + transport (complete)
```

If the second line instead says `consensus ONLY`, your build did not compile the
net, protocol and IPC suites — they are UNIX-only — and the green number above it
is describing a third less code than you think.

For the Python side:

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install numpy tokenizers
```

Torch is only needed for the simulations that actually train. `rnet-tool`, the
node, and the cross-language check need nothing but the standard library.

---

## 1. The whole protocol, one process, no transport

```bash
python3 worker/tests/test_local_simulation.py --workers 4 --rounds 2
```

Genesis to a verified checkpoint, with no networking code involved at all. It
proves, and prints a line for each:

- workers build the model from a hash-verified genesis artifact;
- they train only on the batches the protocol assigned them;
- every contribution is aggregated — a sum, not a race;
- an honest contribution is reproduced bit for bit by a verifier;
- a dishonest one is caught by that same recomputation.

This is the fastest way to see the whole idea work. It is also what `ci/run_tests.sh`
runs, so if it fails on your machine, something is wrong with the machine.

A larger variant, sized like the launch model:

```bash
python3 worker/tests/simulate_rn1b.py
```

---

## 2. A real daemon, a real worker, over the real socket

```bash
python3 worker/tests/test_ipc_roundtrip.py
```

This starts an actual `rnetd` on a temporary datadir and talks to it through
`DaemonClient`. Driving the real binary is deliberate: two implementations of the
same misunderstanding would agree with each other and with nothing else.

It exercises the whole local channel — handshake with anchor verification, getting
an assignment, submitting a contribution through a sealed `memfd`, receiving an
aggregated update, applying it, and reporting the resulting weights hash.

---

## 3. Two nodes talking to each other

Terminal one:

```bash
./build/rnetd --network regtest --worker-id 1 \
              --datadir /tmp/rnet-a --port 19555 --log info
```

Terminal two:

```bash
./build/rnetd --network regtest --worker-id 2 \
              --datadir /tmp/rnet-b --port 19556 \
              --connect 127.0.0.1:19555 --log info
```

Within a second the second node logs a completed handshake, and both start
printing the sixty-second summary line. Nothing else happens: neither node has a
worker attached, so no contribution is ever made and no round can close for want
of the two contributors regtest requires.

To make something happen, attach a worker to one of them. The socket is at
`/tmp/rnet-a/rnet.sock`, and `DaemonClient` is the client:

```python
import hashlib, sys
sys.path.insert(0, "worker")
from rn_worker.consensus import genesis
from rn_worker.ipc.client import DaemonClient

# The two anchors the client will hold the daemon to. The genesis hash is
# compiled into the Python side; the policy hash is taken from the artifact the
# node emitted into its datadir.
g = bytes.fromhex(genesis.GENESIS_HASH["regtest"])
with open("/tmp/rnet-a/regtest.rnpol", "rb") as handle:
    p = hashlib.sha3_256(handle.read()).digest()

with DaemonClient("/tmp/rnet-a/rnet.sock", g, p) as client:
    client.hello()
    print("worker id", client.worker_id, "params", client.n_params)
    print(client.status())
```

`hello()` makes the daemon hand over the round and policy containers as raw
bytes, and the client hashes them against `g` and `p` before believing a word of
it — a mismatch raises `AnchorMismatch` and the connection is dropped. That is
why the worker learns its id and parameter count *from* the handshake rather than
being configured with them.

---

## 4. Running against main

Two things stand between you and a useful main-net node, both listed in the
README's status section and neither of them a build flag:

- **`dataset_root` is all-zero.** No corpus is pinned, so a worker cannot be told
  what to train on. Pinning one changes the genesis anchor, which is a
  coordinated event, not a configuration change.
- **`seed.resonancenet.org` does not resolve.** There is nobody to connect to.
  `--connect` against a peer you started yourself is the only way to have peers.

So `--network main` today gets you a node that derives 983 million parameters of
initial weights, verifies its anchors, opens a port, and finds nothing. That is
the honest state of it.

---

## The datadir

`--datadir` (default `~/.rnet`) holds:

| Path | What it is |
| --- | --- |
| `rnet.sock` | The worker socket. Only present with `--worker-id`. Owner-only. |
| `peers.dat` | The address database, saved every five minutes. |
| `key` | The node's address key. Generated on first run, persisted after. |

Deleting the datadir is safe: everything in it is either regenerable or a cache
of who the node has met.

---

## Reading the logs

`--log info` is the default and prints one summary line a minute plus a line at
every consensus event. `--log debug` adds per-object detail — what was announced,
requested, accepted, refused, and why. `--log trace` adds the wire.

What each line means, and what a monitoring setup would have to scrape in the
absence of a status command: **[observability.md](observability.md)**.
