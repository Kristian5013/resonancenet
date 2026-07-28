# The corpus, and how a worker is told what to train on

> **Status: the reader exists; the corpus it points at cannot be rebuilt.**
> `LocalCorpus`, `RemoteCorpus`, the Merkle index and `rnet corpus-build` are
> all here. What is missing is upstream: the pin names a FineWeb-Edu snapshot
> by content and not by revision, and the dataset has moved since. See
> [Building one](#building-one) — it is the one thing standing between this
> document and a `main` that trains.

## Text, not tokens

The corpus is raw UTF-8, addressed in document-aligned chunks. Nothing is
tokenized in advance, so changing the tokenizer does not mean re-preparing seven
terabytes, and a worker tokenizes the chunk it is assigned with the artifact the
round pins.

Boundaries are **derived, not stored**: a chunk ends at the first `\n\n` at or
after `target_chunk_bytes` from its start. So the manifest is three numbers
rather than an offset table for 150,093 chunks, and any node that scans the
corpus reaches the same boundaries or disagrees loudly about what chunk seven is.

```
dataset_root    2831ce67f1079928e2afa5279462651b29bf9cde066ba9644d6dbd3e4bfd2f9b
n_bytes         7,359,506,899,436
target_chunk    1,048,576
n_chunks        150,093
```

## Which window a worker gets

A pure function of protocol state — corpus root, round, the worker's assigned
id, the outer step, the index within it:

```python
seed   = sha3(BatchSeed{dataset_root, round_id, worker_id, outer_step, inner_index})
chunk  = seed[:8]                                        % n_chunks
offset = sha3("rnet/window-offset" ‖ seed)[:8]           % (tokens_in_chunk - seq_len)
window = tokens[offset : offset + seq_len + 1]
```

**Two draws, domain separated.** Which chunk and where in it come from two
different hashes of one seed rather than two halves of one. Halves would be
cheaper and would leak: a worker learning the chunk index would learn something
about the offset, and one that could search over a draw could steer the other.

**The worker id is an input, which is why it is assigned.** Two workers in a
round must see different data or the second teaches the round nothing. The
daemon hands out the id; a worker that could name itself could take another's
assignment, or fish for one whose data it liked.

## Why the boundaries are addressed at all

So that a chunk arriving from an untrusted peer can be placed. Each chunk is a
leaf of a Merkle tree rooted at `dataset_root`, and a chunk without a proof is
bytes nobody can put anywhere.

Leaves carry a `0x00` prefix and interior nodes `0x01`. Without that, a 64-byte
"chunk" that happens to be two concatenated hashes would verify as the interior
node over them, and a prover could claim any subtree as data.

**The width comes from the manifest, never from the message.** A proof does not
pin the width it was built at — for a given index, several widths produce an
identical walk — so trusting the sender's `leaf_count` would let it choose which
tree its proof is against.

## Reading one

Two implementations of one interface, in `rnet/dataset/corpus.py`:

```python
def window_for_seed(self, seed: bytes, length: int) -> list[int]: ...
def get(self, index: int) -> bytes | None: ...
```

`LocalCorpus` reads the file, derives the boundaries by the rule above,
tokenizes with the pinned artifact, and takes the window. `RemoteCorpus` asks
its daemon, which fetches from a peer and verifies against `dataset_root`
before handing anything over — which is what lets the corpus live on a machine
with the disk for it while training happens on a machine with the GPU for it.

A round that pins a root and cannot read it **fails rather than
inventing**. That guard is not cosmetic: without it the trainer silently
substituted hash noise, the daemon accepted the contribution, the chain advanced
on it, and the verifier — replaying with the same absent corpus — reproduced the
identical noise and returned MATCH. The anti-cheating machinery certified work
that could not possibly have learned anything.

## Building one

```bash
pip install -e '.[corpus]'
rnet corpus-build --out /mnt/data/fineweb-edu.txt --revision <sha>
```

The build is: download, extract the text column, write documents separated by
blank lines, then index — seek to each candidate boundary, read a 64 KiB window,
find the separator, SHA3 the chunk into the tree. Resumable: interrupt it and
run it again, and it truncates to exactly the bytes its state file claims before
appending, so an interrupted write never leaves a torn record inside the corpus.

**`--revision` is not optional if the root has to mean anything.** Without it
both Hugging Face calls track a moving branch, so the file list and the contents
are whatever is there today, and a build a month later differs for reasons
nothing can distinguish from a bug in this code. The pin that preceded this one
named no revision and could not be rebuilt.

**`--include` is not optional either, for this repository.** FineWeb-Edu holds
3,036 parquet files: 2,410 under `data/` and 626 under `sample/` that are
copies of subsets of the first 2,410. Without the filter the corpus gets about
two terabytes of text it already contains — documents the model would see
twice, and a root matching nothing. This is what `main` is pinned to:

```bash
rnet corpus-build --out CORPUS \
  --include data/CC-MAIN-2025-21/ --include data/CC-MAIN-2025-26/ \
  --revision 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9
rnet corpus-index --corpus CORPUS --network main
```

Measured: 100 files, 147.6 GiB of text, 29,642,369 documents, 15 minutes at
170 MiB/s. Indexing it is 60.9 seconds and 150,093 chunks. The whole `data/`
tree is 4.52 TB of parquet and about a day of downloading.

## Moving a network to a different corpus

Changing the corpus is a change to the manifest, not to the code — but it is a
consensus change, so it moves three of the four anchors and every node has to
be rebuilt. The sequence below is the whole of it, and each step refuses rather
than warns if the one before it was skipped.

```bash
rnet corpus-build --out CORPUS --include data/... --revision SHA   # 1
rnet corpus-index --corpus CORPUS                                  # 2
```

Step 2 prints the root and the chunk count. Those two numbers, plus the
repository and revision they came from, go into `rnet/consensus/params.py`.
Nothing else in that file changes.

```bash
rnet genesis-anchors        # 3 — prints GENESIS_HASH; paste it into genesis.py
rnet genesis-weights        # 4 — prints what the weights now derive to
rnet genesis-emit           # 5 — rewrite share/genesis/*.rnet
rnet genesis-verify         # 6 — every artifact against every anchor
./ci/run_tests.sh           # 7
```

**The order is not a suggestion.** `genesis-anchors` reads the tables, so it
must run after params.py. The initial weights are derived from the genesis
hash, so step 4 must run after step 3 — and `verify_build()`, which is what
the cheap checks call, does NOT check the weights root. That is deliberate:
deriving 29.4 billion parameters takes eighty seconds, which is too slow for
something a node does at startup. It does mean a re-pin that skips step 4
passes every quick check and fails only when somebody runs `genesis-weights`.

`POLICY_HASH` does not move — the policy says nothing about the corpus — and
`regtest` does not move either, because it pins no corpus at all. So a re-pin
touches `main`, `test` and `moe`, and a diff showing anything else is a diff
that changed something it did not mean to.

**What a re-pin costs.** Every checkpoint on the old chain is orphaned: a new
genesis is a new set of initial weights, and there is no way to carry trained
weights across, because "the weights are a pure function of the genesis hash"
is the property that lets a stranger check where the network started. A
network that has been training for a month cannot be re-pinned without
throwing that month away. Decide the corpus before the training, not after.
