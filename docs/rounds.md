# A round, from assignment to checkpoint

Two nested loops. The **inner loop** is ordinary training: a worker takes the
agreed weights, runs `inner_steps` optimizer steps on batches it did not choose,
and reports the difference. The **outer loop** is consensus: contributions are
aggregated in integers, one participant applies the result, and the new weights
become a checkpoint every other node can check by recomputing.

```
worker 2 contributed 012d5040a3d83872 at step 1 (loss 5.7036)
worker 1 contributed cb91752c74b9e42d at step 1 (loss 5.6843)
step 1: applying 2 contribution(s)
step 1: checkpoint c0a52a89cbb050f4 (extended), weights 740d844658cc9f31
```

## The contribution

`before - after`, flattened in canonical tensor order, quantised to the round's
width with one shared power-of-two exponent. Not the weights: a 30-billion
model is 59 GB and its update is 29, but more importantly **differences
aggregate and weights do not**.

## Aggregation must not depend on arrival order

Floating-point addition is not associative, so a sum of dequantised floats gives
one answer on a node that received Bob first and another on a node that received
Alice first — both correct about their own arithmetic, both disagreeing about
the checkpoint. That is a chain split arriving through a rounding rule.

So it is integers end to end, and order-independence is checked by shuffling
rather than argued.

**Alignment goes toward the FINEST exponent.** Shifting a coarse contribution up
is exact; shifting a fine one down discards its low bits, and past eight bits
discards all of them — `127 >> 8` is `0`. This aligned toward the coarsest until
an audit caught it, which meant one participant choosing a coarse exponent
silently zeroed everybody else's work on every node that aggregated it.

A contribution more than 24 bits from its peers is refused rather than aligned:
past that it is not a different scale, it is a different quantity, and a worker
could produce it deliberately.

## The outer step happens in the worker

The daemon holds no weights and no momentum. A relay node runs on a machine with
no GPU, and making it carry 3.2 GB of optimizer state for the dense model — 235
GB for the mixture — would mean only machines that could train could also relay.

The daemon supplies what consensus decided (the aggregate, the Q16 rates); the
worker supplies the arithmetic. Every input is public and every operation is
integer, so any node holding the same inputs reaches the same two hashes.

**Every worker applies, not just the producer.** One that skipped the outer step
would start the next round from weights the network has moved past. Only the
first report becomes the checkpoint; the rest are a free cross-check, and two
workers reaching different hashes from identical inputs is worth shouting about.

## Momentum, and knowing whether you have caught up

The outer optimizer's momentum cannot be re-derived from the chain. Under a
constant aggregate the recurrence has a **plateau** of fixed points rather than
one — `m = trunc(0.9m) + 127` holds at 1269 and at 1270 alike — so two differing
histories stay different forever, at an offset of one unit as readily as two
thousand. No rounding rule removes it; it is a property of integer contraction.
Measured in `tests/test_momentum_memory.py`.

Under live training it does converge, at 224 steps for the worst of sixty
trials. But a node must not assume it has, and it does not have to: every
checkpoint header commits to `optimizer_state_hash`, so a node compares its own
against the settled checkpoint and knows exactly.

Out of sync, it contributes and does not produce. In sync, it may produce.

## Fork choice without proof of work

Bitcoin breaks ties with accumulated work, which exists because blocks arrive
unpredictably and anyone may make one. Neither holds here: a round has a
deadline, every node knows when it closes, and heights advance in lockstep. Two
checkpoints at one height are not a race — they are two producers who disagreed,
or one who lied.

**So the tie-break IS the rule: the lower checkpoint id wins.** Not first seen,
which depends on the network. Not most contributions, which a producer chooses.
Not earliest timestamp, which a producer writes. The id is the hash of the whole
header, so every node reaches the same answer from bytes it already holds.

Height still beats hash: a node that fell behind and receives two steps at once
follows them.

## Orphans

A checkpoint whose parent is unknown is the first thing a joining node sees.
Dropping it means a late node never catches up, because the message it needs in
order to ask for the parent is the one it threw away. They are held, capped at
4096, and connect when the parent lands — cascading, so attaching one parent
releases a whole run.

A full pool is **not** the sender's fault, and saying so matters: conflating it
with "malformed" made a node that had been filled once ban every honest peer
offering a checkpoint it could not yet connect. It eclipsed itself.
