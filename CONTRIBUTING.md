# Contributing to ResonanceNet

This is a consensus protocol. The rules below are not style preferences — each one
exists because its absence produced a specific bug in this repository, and every
one of those bugs was silent: the tests stayed green, the node reported itself
healthy, and the damage was only visible from outside.

---

## 1. A comment that asserts a property must name the test that proves it

The comments in this codebase are written as specifications. That is deliberate
and worth keeping. It is also the most dangerous thing about it, because a reader
— including the person who wrote it, six months later — trusts prose that reads
like a spec.

An audit of this repository found roughly a dozen comments asserting properties
the code did not have. A sample, all real:

| The comment said | The code did |
| --- | --- |
| "One transfer, one source" | The check was deletable with 295/295 still green |
| "The beacon is never a value any participant can grind" | It was |
| "The Python mirror is asserted against it" | Both sides were computed live; nothing was asserted |
| "Integer overflow must trap in debug" | It did not; UBSan found the overflow later |

So:

```cpp
// A peer that announces and never delivers stalls the node silently, so requests
// expire and are retried elsewhere.
//   proven by: net_tests.cpp UndeliveredRequestsExpireAndCanGoElsewhere
```

If you cannot name a test, the comment must be rewritten to describe what the code
does rather than what it guarantees. "Requests are given a deadline" is a
description and needs no test. "A peer cannot stall the node" is a claim and does.

This is the highest-leverage rule in this document. A wrong comment is worse than
no comment, because it stops the next person from checking.

---

## 2. Every state must have an exit

Three separate bugs in this repository were the same shape: a component entered a
state and could never leave it. The node kept ticking, logged nothing, answered
handshakes normally, and never did useful work again. From the network it looked
like a peer that was simply quiet.

- A round expired without quorum, was marked `Abandoned`, and stayed there.
  Every later contribution was refused because the round was not `Open`.
- A round `Closed` on a payload that never arrived. Same outcome.
- A round `Closed` on a producer that crashed. Same outcome.

When you add a state, write down how it is left — including the case where the
thing it is waiting for never comes. If the answer is "it waits", bound the wait.

---

## 3. Consensus values are computed by every node, not by one

The outer optimizer used to step only on the node producing a checkpoint. Producer
election rotates, so a follower arrived at its own turn with momentum nobody else
had and computed a different update from identical contributions: 56 against 33
from the same aggregate on the first step, compounding after that because Nesterov
feeds momentum through the lookahead point.

Nothing detected it. The checkpoint committed to which contributions were used and
to the resulting weights hash, but not to the state that produced them.

Two rules follow. **Any value that affects a consensus outcome must be derived by
every node that will act on it**, not by whichever node happens to publish. And
**any state carried between steps must be committed to and verified**, or it is
accepted on someone's word.

---

## 4. Bound the product, not the operand

`kMaxAlignShift` bounded a shift to 24 and `kMaxRescaleShift` bounded another to
32. Both checks passed. Their product did not fit an int64, and the addition after
it was undefined behaviour — meaning the result was whatever the compiler and
target chose, so two nodes could disagree about an update with neither wrong about
its own arithmetic.

Bound the value that reaches the end of the computation. Where the arithmetic can
leave its domain, do it in a wider type and check the narrowing, so the failure is
a named error rather than a wrap.

Run the sanitizers. `ci/run_tests.sh sanitize` found what a careful reading did
not, and the inputs it failed on passed every validation the code had.

---

## 5. Cross-language means tested against the other implementation

The C++ node and the Python worker must agree byte for byte. A test that generates
its expectations from the code under test proves only that the code agrees with
itself.

- The SHA3 primitive is checked against vectors generated from CPython's `hashlib`,
  which wraps the Keccak reference implementation — a second implementation
  disagreeing or not.
- `worker/tests/test_cross_language.py` drives the real `rnet-tool` binary rather
  than a reimplementation of it.

When you change anything hashed, add the vector before the change and watch it
fail.

---

## 6. Tests must be shown to fail

A test that cannot fail reports success forever. Before trusting a new test, break
the thing it covers and watch it go red. The vector-file tests in
`crypto_tests.cpp` include an explicit count check for exactly this reason: a
truncated vector file would otherwise verify nothing and pass.

The same applies to a test that reads data from disk. If the file is missing, the
test must fail, not skip.

---

## 7. Agents read; humans write

If you use an AI agent on this repository, give it read access to `src/` and
`src/test/` and let it write only to a scratch directory. Merge by hand.

This is not a hypothetical. During an audit of this codebase, agents were allowed
to add temporary probe tests to demonstrate their findings. When they reverted
those probes they restored the whole file, deleting three tests a human had
written in the meantime. The fix those tests covered stayed in the code; the proof
of it vanished, and it was noticed only because the test count dropped by three.

An agent with write access to the tests can accidentally prove anything, including
something false. The test count is a useful canary, but a canary is detection, not
prevention.

---

## 8. Format changes are gathered, not sprinkled

Anything that changes a hashed layout — a field in a canonical object, the
protocol version, a consensus parameter — re-pins every network's anchors and
requires re-deriving the genesis weights. Before launch that costs a few minutes;
after launch it is a hard fork.

So: collect format-breaking changes, land them together, bump
`canon::kProtocolVersion` once, and re-emit all three networks in the same commit.
`PROTOCOL_VERSION` in `worker/rn_worker/consensus/canon.py` must move with it.

---

## Before you open a pull request

```
ci/run_tests.sh              # build, tests, artifact reproducibility, cross-language
ci/run_tests.sh sanitize     # address and undefined-behaviour sanitizers
```

Both must pass. If you changed anything hashed, `ci/run_tests.sh` will tell you the
anchors no longer reproduce — that is the check working, not a problem with it.
