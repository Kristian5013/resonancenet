# The four anchors

A node is compiled knowing a small number of 32-byte values and believes an
artifact only if it hashes to one of them. Everything else it will ever
accept — a model shape, a corpus root, a challenge rate, a starting weight —
arrives inside an artifact that had to clear that check first.

That is the whole trust bootstrap. There is no signing authority, no registry,
and nobody to ask.

```bash
python -m rnet genesis-show main
```

| anchor | what it pins | cost to check |
| --- | --- | --- |
| **genesis** | the model shape, the arithmetic, the corpus root, the tokenizer | one SHA3 over 176 bytes |
| **policy** | inner steps, quorum, deadlines, challenge rate, shadow mode | one SHA3 over 116 bytes |
| **corpus** | 7,359,506,899,436 bytes of text, as a Merkle root | a proof per chunk |
| **weights** | where the model started, as a Merkle root over tensors | 1.5 s for 400M, 85 s for 29.4B |

## Why the weights are an anchor and not a file

A network that shipped its initial weights would be asking every participant to
trust whoever produced the file. Here they are a pure function of the genesis
hash:

```
value = uniform(±1/sqrt(fan_in)) drawn from SHAKE-256(domain ‖ genesis ‖ tensor name)
```

Anyone can recompute them and check that the network started where it says it
did.

```bash
python -m rnet genesis-weights main
main     26358eaeb57666cf9e5d5fa59106ab407e5dbbd0f67da925f040abd064bdb37d  397,728,768 params  1.5s
```

**The stream is keyed by the tensor's NAME, not by a running counter.** Two
consequences, both wanted. An architecture change touches only the tensors it
changes, instead of reshuffling every value after the edit and turning a small
diff into an unreviewable one. And **any tensor can be derived alone** — a
holder of one expert shard reproduces its own experts without walking 29 billion
parameters it does not have, which is what makes a sharded mixture checkable by
the people holding it.

## Why the weights hash is a Merkle root

A flat digest over the tensors concatenated must be computed in order, on one
core, and offers nothing but the answer. A root over per-tensor leaves computes
each leaf independently — 32 bytes come back from a worker instead of a
gigabyte — and buys **an inclusion proof for one tensor**. A shard holder can
prove its experts are the ones the network settled on without producing a model
that exists nowhere in one place.

## Changing an anchor is not an upgrade

It names a different network. A node with different anchors refuses every peer
it meets, which is the intended and only safe behaviour.

`verify_build()` checks that the tables in `rnet/consensus/params.py` still
produce the anchors in `rnet/consensus/genesis.py`, so a consensus value edited
without regenerating its anchor is caught at import rather than at the first
handshake with a stranger.

```bash
python -m rnet genesis-anchors     # regenerate, deliberately: this forks the network
```

## The networks

| network | model | corpus | runs today |
| --- | --- | --- | --- |
| `main` | dense 397,728,768 @ seq 16384 | FineWeb-Edu | ✗ no reader |
| `moe` | 29,408,635,904 total / 3.08B active / 8.34B per shard | FineWeb-Edu | ✗ sharding unsolved |
| `test` | same as main, lighter schedule | FineWeb-Edu | ✗ no reader |
| `regtest` | 4,000,512 @ seq 256, portable arithmetic | none — synthetic | ✓ |

Every one starts in **shadow mode**: verdicts recorded, nothing punished, until
the verification path has been watched against real contributions long enough to
trust it. Turning that off is a policy change, which is a fork, not a flag.
