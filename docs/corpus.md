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
rather than an offset table for 6,956,933 chunks, and any node that scans the
corpus reaches the same boundaries or disagrees loudly about what chunk seven is.

```
dataset_root    7195da139188f4a1b779bd380562718e080959531aee0cabbf777cd13501a3b8
n_bytes         7,359,506,899,436
target_chunk    1,048,576
n_chunks        6,956,933
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
are whatever is there today. That is how the current pin came to be
unreproducible: `7195da13…` was built against a snapshot nobody named, and
FineWeb-Edu has since grown from 4.52 TB of parquet to 5.84 TB across 3,036
files. A build today produces a different root, and nothing distinguishes that
from a bug in this code. Any future pin must name the commit it was built from.

Budget: 7.4-9.4 TB of output, ~23 GB of transient cache at `--parallel 8`, and
roughly a day of wall time — download-bound, not CPU-bound.
