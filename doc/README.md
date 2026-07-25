# Documentation

## Getting started

- [`build-unix.md`](build-unix.md) — dependencies, build, tests
- [`developer-notes.md`](developer-notes.md) — coding standards and the rules
  that consensus code must follow

## Design

- [`design/protocol.md`](design/protocol.md) — objects, trust bootstrap, corpus
  integrity, deterministic batch derivation, the verification model
- [`design/architecture.md`](design/architecture.md) — the full architecture
  review this implementation follows, including the components still in progress

## Reading order for a new contributor

1. `design/protocol.md` sections 1-4 — what the protocol guarantees and why
2. `src/canon/canon.h` — the byte layout everything else is built on
3. `src/consensus/params.cpp` — every consensus value, in one file
4. `src/dataset/scheduler.h` — the mechanism that closes data poisoning
5. `src/test/` — each guarantee has a test that proves the rejection path

## Conventions in this repository

- Consensus values live in `src/consensus`. Nowhere else.
- Hashing is SHA3-256 (FIPS 202), everywhere, once.
- Changing a serialized layout is a hard fork: bump the protocol version,
  re-emit genesis, update the anchors, and mirror it in the Python worker.
