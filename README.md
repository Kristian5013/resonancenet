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
  published on an ongoing cadence, each one gated on not regressing against a
  held-out set, with rollback if a later measurement disagrees.

> Status: **pre-launch — there is no network to join.**
>
> Implemented and tested: canonical serialization, the genesis trust anchor,
> derived initial weights, corpus integrity, deterministic batch derivation,
> quantised pseudo-gradients and integer aggregation, challenge selection and
> bit-exact replay, the peer-to-peer transport, and the local channel between a
> node and a training worker. A full cycle runs end to end on one machine:
> contribute, close the round, elect a producer, aggregate, publish a checkpoint,
> and let a second worker catch up from it.
>
> What is missing, and why you cannot join yet:
>
> - **The corpus root is unpinned.** `dataset_root` is all-zero on every network,
>   so no worker can be told which tokens to train on.
> - **No seed infrastructure.** `seed.resonancenet.org` does not resolve. Nodes
>   find each other only through `--connect`.
> - **Slashing runs in shadow mode.** Evidence is built and recorded; nothing is
>   settled. The economic consequence of a false accusation is permanent, so it
>   waits until the evidence has been produced by real hardware for long enough
>   to be believed.
> - **No incentive layer.** Contributions are verified but not rewarded.
>
> Linux only for a full node. `rnet-tool` is portable and is how you verify the
> anchors on any platform; a Windows GPU can run the Linux build under WSL2.

## Design in one page

| Layer | What it guarantees |
| --- | --- |
| `canon` | One byte layout for every hashed object. Big-endian, length-prefixed, CRC- and hash-checked. Cross-language by construction. |
| `crypto` | SHA3-256 (FIPS 202) everywhere — the same primitive as Lattica, so artifacts anchor natively. Merkle trees with RFC-6962 domain separation. |
| `consensus` | Network parameters and the genesis trust anchor. A node trusts exactly one hash a priori and derives everything else from it. |
| `dataset` | Corpus integrity (Merkle root over token chunks) and deterministic batch derivation — the mechanism that removes a worker's ability to choose its own data. |

### Why deterministic batches

An earlier experiment in this project measured whether a held-out validation gate
can catch a poisoned contribution. It cannot: a targeted backdoor reached a 92%
attack success rate while *improving* held-out loss, and Byzantine-robust
aggregation did not stop colluding submitters either. The protocol therefore does
not try to detect bad data after the fact — it removes the worker's ability to
choose data at all, and verifies the computation instead.

## Building

Requires CMake ≥ 3.24 and a C++26 compiler (GCC ≥ 14 or Clang ≥ 19).

```bash
cmake -B build -S .
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure
```

See [`doc/build-unix.md`](doc/build-unix.md) for the full instructions.

## Using the tool

```bash
# Show a network's consensus parameters and its genesis hash
./build/rnet-tool genesis-show --network main

# Emit the artifacts that ship with a release
./build/rnet-tool genesis-emit --network main --out share/genesis

# Verify an artifact against a known anchor
./build/rnet-tool genesis-check --file share/genesis/main.rnet --hash <sha3>

# Index a tokenized corpus and publish its manifest
./build/rnet-tool dataset-build --file corpus.bin --out data/corpus

# Confirm a corpus matches its manifest (detects any alteration)
./build/rnet-tool dataset-check --file corpus.bin --manifest data/corpus.rnds

# Show the exact training windows the protocol assigns to a worker
./build/rnet-tool schedule-show --manifest data/corpus.rnds --worker 1 --step 0
```

## Repository layout

```
src/
  util/        result types, hex, logging, atomic file I/O, JSON, argument parsing
  crypto/      SHA3-256, Merkle trees
  canon/       canonical serialization and the typed consensus objects
  consensus/   network parameters, genesis emission and verification
  dataset/     corpus indexing, manifests, deterministic batch scheduling
  test/        unit tests (ctest target `unit`)
  tools/       rnet-tool
worker/        the Python training client (PyTorch)
doc/design/    architecture and protocol notes
share/genesis/ shipped genesis artifacts and the tokenizer
```

## The worker

Training runs in Python on PyTorch, under `worker/`. It hardcodes no model
parameters: it loads a genesis artifact, verifies it against the pinned anchor,
verifies the tokenizer against the hash inside it, and builds the model from
that. See [`worker/README.md`](worker/README.md).

## Contributing

Developer notes and coding standards: [`doc/developer-notes.md`](doc/developer-notes.md).
Report vulnerabilities privately — see [`SECURITY.md`](SECURITY.md).

## License

MIT. See [`COPYING`](COPYING).
