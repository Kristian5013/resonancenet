# Addressing the corpus by chunk of text

This is the specification for how a worker finds the data it must train on, when
the corpus is stored as raw text rather than as tokens. It has to be precise
enough that the C++ node and the Python worker, written separately, produce
identical bytes — a disagreement here is not a bug, it is a network split.

## Why it changed

The corpus used to be a flat file of token ids, and a training window was a token
offset into it: `offset * 4` bytes from the start. That has one virtue, which is
that reading a window is a seek, and three costs:

- **Publishing a corpus meant tokenizing it first.** Measured on FineWeb-Edu:
  810 GB of parquet takes about 23 CPU-hours to tokenize and produces a 1.28 TB
  file that then has to be stored and served alongside the text it came from.
- **That cost is paid again on every corpus change**, and this project exists to
  train on fresh data, so corpora change often.
- **The token file has to exist before anything can be addressed**, because the
  window count `W` is derived from the total token count, which nobody knows until
  the whole corpus has been tokenized.

A byte-level vocabulary was measured as an alternative and rejected: it removes
the pipeline entirely, but at 5.06 characters per token it needs five times the
positions for the same text, and that lands on context length.

So the corpus is raw UTF-8, and the schedule addresses chunks of it. A worker
fetches the chunk it was assigned, verifies it against `dataset_root`, tokenizes
it with the tokenizer pinned in the round descriptor, and trains. A verifier does
exactly the same and compares.

## The corpus file

A single file of UTF-8 text: documents concatenated, each followed by a blank line
(`\n\n`). The separator is text rather than a reserved token id because there is
no reserved id — the corpus is not tokenized when it is written.

## Chunks are cut on document boundaries

Chunks are **variable-sized**, and each holds whole documents.

The alternative — fixed-size byte ranges — cuts documents and, worse, cuts UTF-8
characters. A chunk that begins mid-character is not a `str` in Python and is a
byte sequence in C++, and any rule for reconciling the two ("decode with
`errors='replace'`") is a rule that must be reimplemented identically in both, on
a path nobody exercises. Cutting on document boundaries removes the question.

The builder starts a new chunk when adding the next document would take the
current one past `target_chunk_bytes`. A document larger than
`target_chunk_bytes` occupies a chunk of its own.

Consequences:

- Every chunk is valid UTF-8 and a whole number of documents.
- Chunk sizes vary, so the manifest carries a byte offset per chunk. At a target
  of 1 MiB over 810 GB that is about 810,000 chunks, 6.5 MB of offsets — small
  enough for a seed to hold in memory, which it must anyway to serve proofs.
- `CorpusSource::ChunkBytes(index)` already exists in the transport, so
  variable-sized chunks need no change there.

## The manifest

| Field | Was | Is |
| --- | --- | --- |
| `dataset_root` | Merkle root over token chunks | Merkle root over text chunks |
| `n_tokens` | total tokens | **removed** — unknown until tokenized |
| `n_bytes` | — | total corpus bytes |
| `chunk_tokens` | tokens per chunk | **removed** |
| `target_chunk_bytes` | — | the size chunks aim for |
| `n_chunks` | ceil(n_tokens / chunk_tokens) | number of chunks |
| `chunk_offsets` | — | `n_chunks + 1` byte offsets, ascending |
| `dtype` | uint16 / uint32 | **removed** — the file is text |
| `tokenizer_hash` | unchanged | unchanged |

`chunk_offsets[i]` is the first byte of chunk `i`; `chunk_offsets[n_chunks]` is
`n_bytes`. Ascending and strictly increasing: an empty chunk is invalid.

## The schedule

For inner step `i` of outer step `step`, worker `worker_id`:

```
h_chunk  = SHA3-256(canon(BatchSeed{dataset_root, round_id, worker_id, step, i, 0}))
chunk    = be64(h_chunk[0..8]) mod n_chunks

tokens   = tokenize(chunk_bytes(chunk))          // the pinned tokenizer
T        = len(tokens)

h_offset = SHA3-256(canon(BatchSeed{dataset_root, round_id, worker_id, step, i, 1}))
start    = be64(h_offset[0..8]) mod (T - seq_len)

window   = tokens[start : start + seq_len + 1]
```

The trailing `0` and `1` are a domain byte, so the chunk and the offset within it
are drawn from independent hashes rather than from one value used twice.

`W`, the quantity a worker cannot influence, is `n_chunks`.

## Every case the rule has to answer

**A chunk yields fewer than `seq_len + 1` tokens.** Forbidden by construction,
and checked when the manifest is built rather than handled at training time. A
BPE token covers at most `max_token_bytes` bytes (a property of the tokenizer
artifact, computable once), so a chunk of `B` bytes yields at least
`B / max_token_bytes` tokens. The manifest is invalid unless

```
min_chunk_bytes / max_token_bytes >= seq_len + 1
```

At `seq_len` 16384 and a longest token of 32 bytes that is 512 KiB, comfortably
under a 1 MiB target. A corpus containing one short document as its own chunk
therefore fails to build, and says so, rather than producing a schedule with a
window nobody can read.

**A chunk yields exactly `seq_len + 1`.** `T - seq_len` is 1, `start` is 0. Fine.

**The last chunk.** No different from any other. It is shorter, and the same
minimum applies to it, so a corpus whose final document is small must have that
document merged into the previous chunk. The builder does this.

**Document separators.** They are bytes in the file, so they tokenize like any
other text. Nothing counts them or skips them, and a window may begin or end
inside one.

**Padding.** None, ever. Every window is exactly `seq_len + 1` real tokens taken
from a chunk that is guaranteed to be long enough, so there is nothing to pad and
nothing to mask.

**Tokenizer determinism.** `tokenizer_hash` in the round descriptor pins the
artifact and both implementations refuse one that does not match. Tokenizing the
same bytes with the same artifact is deterministic — that is the property the
whole scheme rests on, and it is the one property BPE gives for free.

## What a verifier does

A challenge names a contribution, and the contribution header names the worker,
the outer step and the base checkpoint. That is everything the schedule takes, so
the verifier derives the same chunk indices and offsets, fetches the same chunks
by index, verifies each against `dataset_root`, tokenizes them with the same
pinned artifact, and recomputes. It needs no state from the worker beyond what is
already published.

Cost per verification: the same chunks the worker fetched — 250 chunks of about
1 MiB for one outer step — plus the tokenization of them, which the measurement
puts at 26.3 MB/s on 24 cores, so about ten seconds.

## What a worker fetches

250 inner steps, each naming one chunk, so at most 250 distinct chunks of about
1 MiB — **250 MB per outer step**, against a ten-minute round. A chunk is
verified as a whole against its inclusion proof, so there is no way to fetch less
than one; the useful fraction is `(seq_len + 1) * 5.06` bytes of the roughly
1 MiB fetched, about 8%.

That amplification is the price of not pre-tokenizing, and it is worth naming
rather than hiding. It can be reduced later by drawing several of a step's
windows from one chunk, at the cost of correlating the samples within a step.

## What is not decided here

- **`target_chunk_bytes`.** 1 MiB is a placeholder chosen so the minimum-size
  check passes at `seq_len` 16384 with room. Larger chunks mean less
  amplification and coarser sampling.
- **Whether windows within one outer step may share a chunk.** As specified they
  are drawn independently, so collisions are possible but rare, and each is
  fetched once.
- **Whether `max_token_bytes` is recorded in the round descriptor** or recomputed
  from the tokenizer artifact. Recording it is one more consensus value;
  recomputing it is one more thing two implementations must agree on.
