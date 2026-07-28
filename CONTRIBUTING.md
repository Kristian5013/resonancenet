# Contributing

## Every asserting comment names the test that proves it

A comment that says "this closes X" or "measured at Y" must name a test, and the
test must test the claim rather than something adjacent.

This is not style. An audit of this tree found comments naming tests that passed
vacuously — a determinism-class width justified by a test that passes at half
the width, a "measured: 224" that reproduces at 341. **A comment naming a test
that does not prove it is worse than no comment**, because it buys confidence
nobody checked.

Where a number is quoted, it should be reproducible by running something.

## Bugs are found by running, not by reading

Nearly every real defect in this tree was found by starting two processes, not
by review:

- a field named `payload` shadowing `Message.payload()` — invisible in the code,
  a TypeError on the first real send
- an unsolicited `Apply` desynchronising a strict request/response channel
- a self-connection detected on the peer that does not know which address caused
  it
- a trainer silently substituting hash noise for a pinned corpus, which 429
  passing tests did not catch because every test injected a fake corpus that
  production never constructs

If you are adding a code path, add a test that exercises **the path production
takes**, not a parallel implementation of it.

## Consensus changes fork the network

Anything in `rnet/consensus/params.py` is a value the network is defined by.
Changing one means regenerating the anchors, and a node with different anchors
refuses every peer it meets.

```bash
python -m rnet genesis-anchors     # then paste into genesis.py, deliberately
python -m rnet genesis-weights     # the weights anchors depend on the genesis ones
```

`verify_build()` runs on every load and will refuse a build whose tables and
anchors disagree.

## Bounds are not optional

Every length read from a socket is an attacker-chosen allocation. Bound it
before the read, with a constant that lives beside the message rather than a
judgement at the call site. Every collection keyed by something a peer supplies
needs a ceiling and something that empties it — a reaper that is never called is
not a defence, which is a mistake this tree has already made.

## Running the tests

```bash
./ci/run_tests.sh
```

The script exists to export `CUBLAS_WORKSPACE_CONFIG` before python starts,
which a test cannot do for itself: torch reads it when CUDA initialises at
import.

## Style

Match the surrounding code. Comments explain *why*, not *what* — the code
already says what it does. Prefer stating the failure a decision prevents over
describing the decision.
