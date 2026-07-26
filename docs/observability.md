# Watching what a node is doing

## The gap, stated first

**There is no status query command.** No `rnetd status`, no RPC port, no metrics
endpoint, no way to ask a running node anything from another terminal. If you
want to know what your node is doing, you read its log or you attach a worker.

This page documents what exists. The last section describes what a status command
would have to expose, because that is the shape of the missing piece rather than
a wish list.

---

## The log

`--log info` is the default.

### The summary line, every sixty seconds

```
peers 5 (5 ready, 2 in, 3 out) — addresses 143 (61 tried) — objects 892
consensus: step 47, 3 contributions this round, 12 objects
```

| Field | Meaning |
| --- | --- |
| `peers N` | Connections currently held, including ones still handshaking. |
| `N ready` | Peers past the handshake. Only these count for gossip. |
| `N in` / `N out` | Inbound vs. outbound. All-inbound is the shape an eclipse attack wants; a healthy node dials out. |
| `addresses N` | Size of the address database. |
| `N tried` | How many of those the node has actually connected to successfully. |
| `objects N` | Objects in the store. Capped at 10,000, oldest evicted. |
| `step N` | The node's current outer step — the height of its chain. |
| `N contributions this round` | Accepted so far in the round now open. |

The second line appears only on a participating node (one started with
`--worker-id`).

### Consensus events

These are the lines worth grepping for. Each is printed once, when it happens.

```
participant: staged step 12 from 3 contributions, awaiting the worker
```
This node was elected producer, aggregated, and is waiting for its worker to
apply the update on the GPU. If you see this and then nothing for a full round
deadline, the worker never came back — the node drops the staged update and says
so rather than holding it forever.

```
participant: tip is now step 12 (30895c111a150266)
participant: produced checkpoint 30895c111a150266 for step 12 from 3 contributions
```
A checkpoint became canonical. The first line appears on every node; the second
only on the one that produced it.

```
participant: round at step 11 expired with 1 of 2 contributions; reopening
```
Not enough workers contributed before the deadline. Normal on a small or quiet
network — this is the round restarting, not an error.

```
participant: step 11 closed but the elected producer published nothing; reopening
rather than waiting forever
```
The elected producer went away. The election rotates on a wall-clock slot, so the
next attempt picks someone else.

```
participant: reorganising to 9042821a83d4162a at height 12, replacing d325e533dae200ca
```
Two producers were elected for the same step — the usual cause is clock skew
across a slot boundary — and both published. The fork rule keeps the lower id.
Rare; if you see it often, check the machine's clock.

```
participant: accepting checkpoint at step 12 without verifying its optimizer state
(it aggregated a different set of contributions than this node did); this node
cannot produce until it reproduces one
```
Ordinary asynchrony: the producer saw a contribution this node did not, or the
reverse. The node follows the chain but will not produce from momentum it cannot
justify. Persistent, on this build, because there is no way to re-derive optimizer
state — see the README's status section.

```
ipc: worker 3 asked to apply the update for step 12 (3 contributions)
ipc: releasing the apply slot held by worker 3 for assignment 7
```
The second line means the worker holding the exclusive apply slot vanished, and
the slot was released so another worker can take it.

### Peer scoring

At `--log debug`:

```
peer 4 misbehaving (+50 = 50): checkpoint id does not match its content
```

The threshold is 100. Scores are graded deliberately — 20 for malformed, 50 for a
forged identity, 0 for "refused, but not your fault" — so a node that is merely
behind the tip does not ban the peers that could bring it up to date.

---

## The one machine-readable interface

The worker socket. A process holding an IPC connection can call `status()`:

```python
client.status()
# {'peers': 5, 'objects': 892, 'outer_step': 47, 'contributions': 3}
```

Four fields, canonical CBOR. That is the entire programmatic surface, and it
exists because the worker needs to know what to train on, not because anyone
designed it as a monitoring interface.

`get_assignment()` returns rather more, and is the closest thing to a view of
consensus state:

```
round_id, outer_step, assignment_id, base_checkpoint, base_weights_hash
```

Both require the caller to have completed the handshake, which requires it to
hold the correct anchors. There is no unauthenticated read path, by design: the
socket is owner-only and the node verifies the peer's uid.

---

## Following a run

Since there is no query interface, the practical approach is to keep the log and
filter it:

```bash
./build/rnetd --network regtest --worker-id 1 --datadir /tmp/rnet-a --log info \
  2>&1 | tee /tmp/rnet-a/log
```

```bash
# Just consensus progress
grep -E "tip is now|produced checkpoint|reorganising" /tmp/rnet-a/log

# Why is nothing happening?
grep -E "expired with|published nothing|without verifying" /tmp/rnet-a/log

# Peer health
grep -E "misbehaving|disconnected|connected" /tmp/rnet-a/log
```

Note that the log is written to stdout, not to a file in the datadir. If you run
the daemon detached, redirect it yourself, and use `setsid nohup … < /dev/null &`
rather than plain `nohup` — the latter does not survive the shell exiting.

---

## What a status command would need to expose

Written down because it is the specification of the missing piece, not a wish
list. Anything less and an operator still cannot answer "is my node healthy?"

**Chain**
- tip id, height, outer step
- whether the node considers itself synced
- `optimizer_desynced` — because a node with this set follows the chain but can
  never produce, and today nothing surfaces it except one log line at the moment
  it happens

**Round**
- state (open, closed, abandoned) and when the deadline falls
- contributions accepted, and how many payloads are still outstanding
- whether this node is the elected producer for the current slot, and the slot
  number

**Peers**
- per peer: direction, ready, misbehaviour score, bytes moved, last message
- the address database's size and tried count

**Worker**
- whether a worker is attached, its id, and what it was last asked to do
- whether the apply slot is held, and by whom

The natural shape is a second command on the existing Unix socket, reusing the
CBOR framing, with a read-only role that does not require the caller to be a
worker. That keeps it owner-only and adds no network surface.
