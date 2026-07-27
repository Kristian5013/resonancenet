<div align="center">
  <h1>ResonanceNet</h1>
  <p><strong>A protocol for decentralized, verifiable language-model training.</strong></p>
</div>

---

ResonanceNet is a peer-to-peer protocol in which a language model is the product
of a network, the way a blockchain is the product of Bitcoin. Anyone with a
suitable GPU joins, trains the shared model, and the network agrees on which
checkpoint becomes canonical.

Three properties define the design:

- **Consensus over the model.** Every architectural detail — dimensions,
  tokenizer, optimizer, corpus root — is a consensus parameter, pinned in a
  genesis artifact and verified by hash. Nodes that disagree cannot produce
  compatible gradients, so the parameters are treated with the same rigour a
  currency applies to its chain parameters.
- **Verification by recomputation, not by trust.** A worker does not choose its
  training data: batches are derived deterministically from protocol state, so a
  verifier can reproduce any claimed update exactly and compare it bit for bit.
- **Continuous training.** The model is not frozen at release. Checkpoints are
  published on an ongoing cadence, each one recomputable from the contributions
  it names, and the chain reorganises when a competing checkpoint wins the fork
  rule.

---

## Status

**The corpus is pinned and there is a seed.** What you cannot yet do is join
late: a node that starts after the first checkpoint has no way to catch up, so
round 0 begins with everyone starting together. That is the one remaining hole
and it is being closed.

<details open>
<summary><strong>Round 0</strong></summary>

| | |
| --- | --- |
| Parameters | 397,728,768 |
| Shape | `d_model` 1024, 24 layers, 8 heads (head_dim 128), GQA 4:1, `d_ff` 4096 |
| Vocabulary | 32,000 — byte-level BPE, pinned by hash |
| Context | 16,384 tokens |
| Corpus | FineWeb-Edu, every dump, 7.36 TB of text in 6,956,933 chunks |
| Schedule | 200 inner steps per round, 20-minute rounds |
| Hardware | 24 GB GPU. Measured: 6.0 GB peak, 4.09 s per inner step on an RTX 5090 |

Every number was chosen by measurement on a consumer card rather than by
analogy. The reasoning is in the commits; the short version is in
[docs/training.md](docs/training.md).
</details>

<details open>
<summary><strong>What is implemented and tested</strong></summary>

Canonical serialization; the genesis trust anchor; initial weights derived from
that anchor rather than distributed; corpus integrity over document-aligned
chunks of text; deterministic batch derivation; quantised pseudo-gradients and
integer aggregation; the producer election and its fork rule; the peer-to-peer
transport; the local channel between a node and a worker; and fetching corpus
chunks from a seed with an inclusion proof against `dataset_root`.

A full cycle runs end to end: contribute, close the round, elect a producer,
aggregate, publish a checkpoint — including catching a worker that trained on
data it chose itself. 320 tests, plus a cross-language suite that drives the
real `rnet-tool` binary from Python so the two implementations cannot drift
apart silently.
</details>

<details open>
<summary><strong>What is missing</strong></summary>

- **A node cannot catch up.** There is no chain sync: a node that joins after
  the first checkpoint receives ones whose parents it does not hold and stays
  where it started. Round 0 therefore starts with everyone together.
- **A node cannot re-derive optimizer state.** Momentum is committed to as a
  hash, not distributed. A node that missed a step follows the chain but will
  not produce.
- **Verification is not wired up.** Challenge selection and bit-exact replay
  work as libraries and are exercised by the simulations. In a running node
  nothing answers a challenge: `AssignVerify` and `SubmitVerdict` are message
  types with no handler.
- **No incentive layer.** Contributions are not rewarded.
</details>

Linux only for a full node. `rnet-tool` is portable and is how you verify the
anchors on any platform; a Windows GPU can run the Linux build under WSL2.

---

## Joining

Three steps. The whole thing is one node and one trainer.

### 1. Build

```bash
git clone https://github.com/Kristian5013/resonancenet.git
cd resonancenet
cmake -B build -S .
cmake --build build -j"$(nproc)"
./build/rnet_tests
```

The last two lines must read `320 passed, 0 failed` and `suite: consensus +
transport (complete)`. If the second says `consensus ONLY`, the transport
suites were not compiled and the green number above describes a third less code
than you think.

Requires CMake ≥ 3.24 and GCC ≥ 14 or Clang ≥ 19. Nothing to fetch.

### 2. Check what your build believes

```bash
./build/rnet-tool genesis-show --network main
```

`genesis_hash` must equal `pinned_anchor`, and `policy_hash` must equal
`pinned_policy_anchor`. If they differ, your build would train a different model
from everyone else, and nothing it produced would be accepted.

To go further and re-derive the initial weights — real arithmetic over 397
million parameters, a couple of minutes:

```bash
./build/rnet-tool genesis-weights --network main
```

### 3. Run a node, then a trainer

```bash
./build/rnet-tool genesis-emit --network main --out ~/.rnet/main
./build/rnetd --network main --worker-id YOUR_ID --datadir ~/.rnet/main
```

`--worker-id` is any number nobody else on the network is using. The node finds
the seed through DNS; no `--connect` is needed.

Then, in another terminal:

```bash
python3 -m venv worker/.venv
worker/.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
worker/.venv/bin/pip install numpy tokenizers

export CUBLAS_WORKSPACE_CONFIG=:4096:8
worker/.venv/bin/python worker/rn_train.py
```

That is the whole thing. The trainer asks the node what to train, trains it, and
submits. It chooses nothing: weights, corpus, schedule and stopping point all
come from consensus, which is what makes a contribution checkable rather than
merely received.

`CUBLAS_WORKSPACE_CONFIG` must be set in the shell, before Python starts. By the
time a script could set it, the decision it affects has been made.

### What you will see

```
network   main, round 0
model     397,728,768 parameters, seq_len 16384
corpus    7195da139188f4a1…
schedule  200 inner steps per round
worker id 7 (assigned by the node, not chosen here)

step 0: training 200 inner steps
step 0: submitted, loss 10.3421, 818s
```

If instead you see the node and the worker disagreeing about a genesis hash,
they are on different commits — rebuild both.

Longer form, including what each assertion means and what to do when it breaks:
**[docs/training.md](docs/training.md)**.

---

## The command line

Two binaries. `rnet-tool` is offline and portable; `rnetd` is the node.

### `rnet-tool` — anchors, corpora, schedules

| Command | What it does |
| --- | --- |
| `genesis-show` | Prints the whole consensus state as JSON: model dimensions, policy, anchors. |
| `genesis-emit` | Writes the genesis artifacts to disk and prints their hashes. |
| `genesis-weights` | Re-derives the initial weights from the anchor and prints their hash. |
| `genesis-check` | Verifies an artifact file against an expected hash. |
| `dataset-build` | Builds a `.rnds` manifest — a Merkle root over a tokenized corpus. |
| `dataset-check` | Verifies a corpus file against its manifest. |
| `schedule-show` | Prints the exact training windows a worker is assigned at a step. |

Full options, exact output, and worked examples: **[docs/cli.md](docs/cli.md)**.
Running a node and a worker without joining the network — the whole protocol on
one machine — is in **[docs/running-a-node.md](docs/running-a-node.md)**.

```bash
# What is this build training?
./build/rnet-tool genesis-show --network main

# Prove the initial weights are derived, not handed out
./build/rnet-tool genesis-weights --network main

# Which windows would worker 7 train on at step 100?
./build/rnet-tool schedule-show --manifest data/corpus.rnds --worker 7 --step 100
```

### `rnetd` — the node

```
--network <main|test|regtest>   which consensus rules to run (default: main)
--datadir <path>                state directory (default: ~/.rnet)
--worker-id <n>                 this node's worker identity; required to participate
--connect <host:port>           connect only to this peer; repeatable
--bind <addr>                   listen address (default: 0.0.0.0)
--port <n>                      listen port (default: the network's own)
--no-listen                     do not accept inbound connections
--no-seeds                      do not query DNS seeds
--max-inbound <n>               inbound slots (default: 64)
--max-outbound <n>              outbound slots (default: 8)
--corpus <path>                 tokenized corpus to serve (needs --manifest)
--manifest <path>               the .rnds manifest describing that corpus
--verify                        answer verification challenges assigned to this node
--log <error|warn|info|debug|trace>
```

Without `--worker-id` the daemon runs as a relay: it gossips objects but takes no
part in consensus and opens no worker socket.

---

## Watching what a node is doing

Honest answer first: **there is no status query command.** You cannot ask a
running node for its state from another terminal — no `rnetd status`, no RPC
port, no metrics endpoint. That is a real gap, listed here rather than papered
over.

What exists today is the log. A participating node prints a summary line every
sixty seconds:

```
peers 5 (5 ready, 2 in, 3 out) — addresses 143 (61 tried) — objects 892
consensus: step 47, 3 contributions this round, 12 objects
```

and a line at every consensus event: a round closing, a producer being elected, a
checkpoint arriving, a reorganisation, a peer being scored. `--log debug` adds
per-object detail; `--log trace` adds the wire.

The one machine-readable interface is the worker socket. A process holding the
IPC connection can call `status()` and gets the node's step, tip and round state
as CBOR — that is how the worker knows what to train. `DaemonClient` in
`worker/rn_worker/ipc/client.py` is the client.

Which fields exist, how to read the log lines, and what a stats CLI would need to
expose: **[docs/observability.md](docs/observability.md)**.

---

## The dataset question

*When is the model "done" with a dataset, and who decides to move to the next
one?*

Nothing ever finishes a dataset, because nothing walks through one. A worker's
training windows are drawn independently at random:

```
offset_i = SHA3-256(canon(BatchSeed{dataset_root, round, worker, step, i})) mod W
```

That is sampling **with replacement** over all `W` valid window positions. There
is no cursor, no epoch counter, no "consumed" flag, and therefore no exhaustion
event to detect. Training draws from the corpus indefinitely, and coverage is
statistical: after `k·W` windows have been drawn, the expected fraction of the
corpus seen at least once is `1 − e^(−k)` — about 63% at one pass' worth of
draws, 95% at three, 99.3% at five.

So "the model has been trained on the dataset" is a judgement, not an event. The
signal to move on is the loss curve flattening, not the corpus running out.

**Changing the corpus means changing `dataset_root`, which is a consensus value**
— it lives in the round descriptor, whose hash is the genesis anchor every node
checks at startup. Changing it produces a different anchor, so it is not a
setting an operator can flip. It is a new *round*: `round_id` exists in every
consensus object precisely to separate one set of round parameters from another.

**And the round transition is not built.** `round_id` is 0 on all three networks,
and nothing implements starting round 1. Worse, if it were done today by emitting
a new genesis artifact, all prior training would be discarded: a round's initial
weights come from `genesis_weights_hash`, which is derived from the artifact, so
round 1 would start from freshly derived random weights rather than from round 0's
final checkpoint.

What a real transition has to do, and why each part is required:
**[docs/dataset-lifecycle.md](docs/dataset-lifecycle.md)**.

---

## Running the model with llama.cpp

Separate from the protocol: this is how you run a GGUF model locally, whether to
sanity-check an architecture or just to have a strong assistant on the same
machine you are developing on.

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build --config Release -j"$(nproc)"
```

`CMAKE_CUDA_ARCHITECTURES=120` is Blackwell — an RTX 5090, including the Laptop
part. Use `89` for Ada (4090), `86` for Ampere (3090), or drop the flag to build
for everything and wait considerably longer. Without a CUDA toolkit, omit
`-DGGML_CUDA=ON` and it builds for CPU.

Then serve a model with its web UI:

```bash
./build/bin/llama-server \
  -m ~/models/your-model.gguf \
  --host 127.0.0.1 --port 8080 \
  -ngl 99 -c 32768 --flash-attn on
```

Open <http://127.0.0.1:8080>. `-ngl 99` offloads every layer to the GPU; lower it
until the model fits. `-c` is the context window in tokens — raise it only as far
as VRAM allows, since the KV cache grows linearly with it.

Long contexts, quantised KV cache, choosing a quantisation, and one-shot CLI use:
**[docs/llama-cpp.md](docs/llama-cpp.md)**.

---

## How it fits together

| Layer | What it owns |
| --- | --- |
| `canon` | Canonical serialization. Big-endian, fixed-width, length-prefixed; a trailing byte is malleability and is refused. |
| `crypto` | SHA3-256 and RFC-6962 Merkle trees. Checked against 621 vectors generated from CPython's `hashlib`, not from itself. |
| `consensus` | Network parameters and the genesis anchor. The one thing a node trusts a priori is a hash. |
| `diloco` | The chain, the round, integer aggregation, the outer optimizer, the producer election and fork rule. |
| `dataset` | Corpus integrity and deterministic batch derivation — the mechanism that removes a worker's ability to choose its own data. |
| `verification` | Challenge selection and bit-exact replay. Implemented; not yet wired into the running node. |
| `net` | The peer-to-peer transport: handshake, gossip, bulk transfer. |
| `ipc` | The local channel to a training worker: Unix socket, canonical CBOR, sealed memfd for bulk. |
| `protocol` | The node itself — participant, service, worker service. |

### Why deterministic batches

An earlier experiment in this project measured whether a held-out validation gate
can catch a poisoned contribution. It cannot: a targeted backdoor reached a 92%
attack success rate while *improving* held-out loss, and Byzantine-robust
aggregation did not stop colluding submitters either. The protocol therefore does
not try to detect bad data after the fact — it removes the worker's ability to
choose data at all, and verifies the computation instead.

---

## Contributing

Read **[CONTRIBUTING.md](CONTRIBUTING.md)** first. Every rule in it exists because
its absence produced a specific silent bug in this repository — tests green, node
reporting itself healthy, damage visible only from outside.

Before opening a pull request:

```bash
ci/run_tests.sh              # build, registered-test diff, tests, artifact reproducibility, cross-language
ci/run_tests.sh sanitize     # address and undefined-behaviour sanitizers
```

Both must pass. If you changed anything hashed, `ci/run_tests.sh` will tell you
the anchors no longer reproduce — that is the check working.

---

## Licence

MIT. See [LICENSE](LICENSE).
