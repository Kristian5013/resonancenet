# Developer notes

## The rule that shapes everything

**If two honest nodes must agree on a value, it lives in `src/consensus` and is
verified by hash.** No model dimension, tokenizer, corpus root or schedule
constant may be written into a worker, a tool or a script. They read a genesis
artifact, check it against the pinned anchor, and use what it says.

A hardcoded consensus value is not a shortcut, it is a latent network split.

## Layering

```
util  <-  crypto  <-  canon  <-  consensus  <-  dataset
```

A layer may only include from layers below it. `util` has no project
dependencies; `canon` knows nothing about networks; `consensus` knows nothing
about corpora.

## Error handling

No exceptions on consensus paths, no bare `bool` returns that a caller can ignore.
Every fallible operation returns `Result<T>` or `Status` and every one is
`[[nodiscard]]`.

```cpp
auto raw = util::ReadFile(path);
if (!raw) return Err(raw.error());     // explicit, with a reason
```

This exists because of concrete defects found in audits of comparable projects:
a checkpoint loader that ignored `fread` results and loaded a truncated file as
zeroes "successfully", and a gradient clip that failed open on NaN. Both were
silent. Nothing in this codebase is allowed to fail silently.

## Serialization

All hashed bytes go through `canon`. Fixed-width, big-endian, length-prefixed,
no padding, no implementation-defined types. When adding a field:

1. Add it to the struct and to `Serialize()`/`Deserialize()` in the same order.
2. Add a case to the "every field affects identity" test — a field that does not
   change the object hash is a field an attacker can vary freely.
3. Bump `kProtocolVersion`. Changing a layout is a hard fork.
4. Re-emit genesis artifacts and update the pinned anchors.
5. Mirror the change in `worker/rn_worker/consensus/` and confirm
   `worker/tests/test_cross_language.py` still passes.

## Tests

Every consensus behaviour needs a test that proves the *rejection* path, not just
the happy path. Malformed, truncated, appended, tampered and cross-network inputs
must all be refused. `RNET_CHECK_ERR` exists specifically for that.

Run `ci/run_tests.sh` before submitting; it is the same script CI runs, including
the genesis reproducibility check.

## Style

`.clang-format` is authoritative (Google base, 100 columns, 4-space indent).
Comments explain *why* — the code already says what. A comment that restates the
next line is noise; a comment that records the measurement or the attack behind a
decision is the reason the decision survives a refactor.

## Determinism

Verification works by recomputing a worker's update and comparing bit for bit, so
anything that affects arithmetic is consensus-relevant. Deterministic mode costs
roughly 3x the memory of the fast path (the attention matrix is materialised),
which is why the launch model and context length are what they are — they were
measured on the target hardware, not chosen for elegance.

Cross-architecture reproducibility is an **open measurement**, not an assumption.
Until it is settled, verification is sharded by determinism class and
`DeterminismClass::Pending` (0) means "unassigned".
