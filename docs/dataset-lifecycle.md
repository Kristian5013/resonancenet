# The dataset lifecycle

> *When is the model done with a dataset? Do you stop it manually and swap in
> another one? How is that supposed to work?*

Short answer: **nothing ever finishes a dataset, because nothing walks through
one.** And swapping one in is not an operator action — it is a new round, and the
round transition is not built yet.

The rest of this page is why, and what a real transition has to do.

---

## 1. There is no cursor, so there is no end

A worker's training windows are drawn like this:

```
offset_i = SHA3-256(canon(BatchSeed{dataset_root, round_id, worker_id, step, i})) mod W
```

where `W` is the number of valid window start positions in the corpus —
`n_tokens − seq_len`, roughly.

That is **sampling with replacement**. Each window is drawn independently and
uniformly. There is no read position, no epoch counter, no "consumed" bitmap, and
no state that could tell you the corpus had been exhausted. Two consecutive steps
can draw the same window; a window can go undrawn for a very long time.

This is not an oversight — it is what makes the scheme work:

- **A worker cannot choose its data.** The offsets are a pure function of protocol
  state, and `worker_id` is assigned by the node, not claimed by the worker.
- **A verifier can reproduce any batch exactly**, from the same five inputs, with
  no shared state to synchronise. A cursor would be shared mutable state that
  every node would have to agree on, and disagreement about it would be a fork.

You can see any worker's assignment for yourself:

```bash
./build/rnet-tool schedule-show --manifest data/corpus.rnds --worker 7 --step 100
```

---

## 2. So what does "trained on the dataset" mean?

Coverage, and it is statistical rather than an event.

After `n` windows have been drawn from `W` positions, the expected fraction of the
corpus seen at least once is `1 − e^(−n/W)`:

| Draws | Expected coverage |
| --- | --- |
| `1 × W` | 63.2% |
| `2 × W` | 86.5% |
| `3 × W` | 95.0% |
| `5 × W` | 99.3% |
| `7 × W` | 99.9% |

So "one epoch's worth of draws" leaves over a third of the corpus untouched, and
you need about five times the corpus size in draws to have seen essentially all
of it. Random sampling with replacement is the standard trade for not having to
coordinate a cursor across a network, and the cost is this constant factor.

The number of draws per outer step is fixed by the policy:
`inner_steps × micro_batch` per worker — 250 on main — multiplied by the number of
contributing workers.

**The signal to move on is the loss curve flattening, not the corpus running out.**
Nothing in the protocol can detect the former, and the latter never happens.

---

## 3. Changing the corpus is a consensus change

`dataset_root` lives in the round descriptor:

```
RoundDescriptor {
    protocol_version, network_magic, round_id,
    model, determinism_class, optimizer,
    tokenizer_hash,
    dataset_root        <- here
}
```

The SHA3-256 of that structure **is** the genesis anchor — the one value a node
trusts a priori, compiled into both the C++ node and the Python worker, and
checked at every handshake. Change `dataset_root` and the anchor changes, so:

- every node and worker must be rebuilt or reconfigured with the new anchor;
- a worker on the old anchor is refused at the handshake with `AnchorMismatch`,
  which is exactly right — it would otherwise train a different model and call the
  result a contribution.

This is why `round_id` exists in every consensus object. It is the field that says
which set of round parameters an object belongs to. Contributions, challenges and
evidence are all checked against the node's own `round_id` and refused if they
disagree.

---

## 4. The transition is not built

Concretely, today:

- `round_id` is **0** on main, test and regtest. Nothing anywhere sets it to 1.
- There is no message, no command, and no procedure for announcing that round 0
  has ended and round 1 has begun.
- The chain's own error text — `"chain: parameter count changed without a new
  round"` — refers to a mechanism that does not exist.

And the part that actually matters:

**A new round would start from scratch.** A round's initial weights come from
`genesis_weights_hash` in the network parameters, and that value is checked
against weights *derived* from the genesis anchor:

```bash
./build/rnet-tool genesis-weights --network main
# deriving 983635968 parameters for 'main' — this is real work, not a lookup
# derived  57ecba31…
# anchor   57ecba31…
# match: this node starts from the state the network agreed on.
```

So emitting a round-1 artifact today produces a round whose step 0 is freshly
derived random weights. Every step of round 0's training would be discarded at
the moment the corpus changed — which is the opposite of the point.

---

## 5. What a real transition has to do

Four requirements. Each one is load-bearing; dropping any of them either breaks
verifiability or throws away the training.

### 5.1 A round must be able to start from a checkpoint

`genesis_weights_hash` currently means "the hash of the weights derived from this
anchor". Round 1 needs it to mean "the hash of the weights of checkpoint X of
round 0", with X named in the descriptor. That makes the initial state of a round
either *derived* (round 0) or *inherited* (every round after), and a node must be
able to tell which and check accordingly.

Inheriting has a consequence the derived case does not have: the weights can no
longer be computed, so they must be transferred. The bulk transport already moves
gigabyte objects with Merkle-verified chunks, so the mechanism exists; what does
not exist is anything that asks for a checkpoint's weights by hash on joining.

### 5.2 The optimizer state has to cross the boundary, or be reset deliberately

Momentum accumulates across steps and is committed to as a hash in every
checkpoint. At a round boundary there are exactly two honest options:

- **Reset it to zero**, and say so in the descriptor, so every node starts round 1
  from the same empty state — one warmup's worth of quality cost, and no
  coordination problem.
- **Carry it over**, which requires distributing the momentum tensor itself, since
  a hash cannot be inverted.

There is no third option where nodes "just continue", because a node that joins at
round 1 has no way to reach a nonzero momentum state.

Note this is the same missing mechanism as the one that stops a node ever joining
mid-round today. Building it once solves both.

### 5.3 Checkpoints must be bound to their round

`ContributionHeader`, `ChallengeOrder` and `SlashEvidence` are all checked against
the node's `round_id`. `CheckpointHeader` carries a `round_id` **and nobody checks
it** — not `Participant::OnCheckpoint`, not `CheckpointHeader::Validate`, not
`CheckpointChain::Append`. A checkpoint labelled with any round at all is accepted
as long as its parent, step and parameter count line up.

That is harmless while there is exactly one round in existence. It stops being
harmless the moment there are two.

### 5.4 The switch has to be scheduled, not announced

Nodes must not switch at the moment they hear about it, or the network splits into
"switched" and "not yet" for as long as gossip takes. The switch point has to be a
value every node computes identically — an outer step in round 0 after which round
1 begins — pinned in the round-1 descriptor, so a node that receives it early
knows exactly when to act and a node that receives it late catches up to the same
answer.

---

## 6. What to do in the meantime

For the first corpus, none of this is on the critical path: round 0 has never
trained, so there is nothing to preserve. Pin `dataset_root`, re-emit the anchors,
re-derive the weights, and start.

The transition machinery becomes necessary the first time the network has trained
something worth keeping. Building it before then is premature; building it after
the corpus is exhausted in the statistical sense — several times `W` draws — is
too late, because by then stopping the network is the only way to change anything.
