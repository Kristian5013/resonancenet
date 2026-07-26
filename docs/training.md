# Running training

Read this first, because it decides whether the rest is useful to you:

> **There is no trainer command, and there is no network to join.** The worker is
> a Python library. What you can run today is the whole protocol on your own
> machine — real training, real aggregation, real verification — plus a node that
> talks to a worker over a socket. What you cannot do is contribute to a shared
> model, because no corpus is pinned and no seed resolves.

Everything below was run against this tree. Where output is shown, it is copied.

---

## What "training" means here

A round is two nested loops.

The **inner loop** is ordinary training: a worker takes the current weights, runs
`inner_steps` optimizer steps on batches it did not choose, and computes the
difference between the weights it started with and the weights it ended with.
That difference — quantised to one byte per parameter — is its *contribution*.

The **outer loop** is consensus: contributions are aggregated with integer
arithmetic, one elected node applies the result, and the new weights become a
checkpoint every other node can check by recomputing.

The part that makes it verifiable rather than merely distributed: a worker's
batches are a pure function of protocol state, so anyone holding the same corpus
can reproduce them exactly. See [corpus-addressing.md](corpus-addressing.md).

---

## 0. Build, and prove the build

```bash
cmake -B build -S .
cmake --build build -j"$(nproc)"
./build/rnet_tests
```

Last two lines must be:

```
317 passed, 0 failed
suite: consensus + transport (complete)
```

If the second says `consensus ONLY`, the transport suites were not compiled and
the green number above describes a third less code than you think.

Then check that this build agrees with the network about what is being trained:

```bash
./build/rnet-tool genesis-show --network main
```

`genesis_hash` must equal `pinned_anchor`. Round 0 is:

| | |
| --- | --- |
| Parameters | 397,728,768 |
| Shape | `d_model` 1024, 24 layers, 8 heads (head_dim 128), GQA 4:1, `d_ff` 4096 |
| Vocabulary | 32,000 — byte-level BPE, pinned by hash |
| Context | 16,384 tokens (~76 KB of text) |
| Inner steps | 200 per round |

---

## 1. Python, and why it needs a GPU

```bash
python3 -m venv worker/.venv
worker/.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
worker/.venv/bin/pip install numpy tokenizers pyarrow
```

Training needs CUDA. The consensus half — `rnet-tool`, the node, the
cross-language checks — needs nothing but the standard library, which is
deliberate: verifying what a network claims should not require the hardware to
participate in it.

Deterministic mode is not optional, and one thing must be set before torch
initialises CUDA:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Set it in your shell, not in the script. By the time Python can set it, the
decision it affects has been made.

---

## 2. The whole protocol on one machine

This is the one to run first. Genesis to a verified checkpoint, no networking
involved:

```bash
worker/.venv/bin/python worker/tests/test_local_simulation.py --workers 4
```

```
[honest round]
  ok    every contribution was aggregated
  ok    all challenges were answered
  ok    every honest update reproduced bit for bit
  ok    the checkpoint advanced
        round took 16.0s, loss 5.7101
[dishonest workers]
  ok    data-poisoned worker caught
  ok    data: honest workers not accused
  ok    delta-poisoned worker caught
  ok    delta: honest workers not accused
[multi-round]
  ok    each round produced a distinct checkpoint
  ok    outer optimizer advanced
```

Four claims worth understanding, because they are the protocol:

- **every contribution was aggregated** — a sum, not a race. No worker's work is
  dropped for arriving second.
- **reproduced bit for bit** — a verifier re-ran a worker's training from the same
  weights and the same batches and got byte-identical output. This is what makes
  cheating detectable rather than merely improbable.
- **data-poisoned worker caught** — a worker that trained on data it chose itself
  is caught, because the batches it was supposed to use are derivable.
- **honest workers not accused** — the check has no false positives, which matters
  more than the detection: a protocol that slashes honest work is worse than one
  that misses some cheating.

This uses regtest — a 3M-parameter model with a 256-token context — so it runs in
seconds on any CUDA card.

---

## 3. Training the real model

```bash
worker/.venv/bin/python worker/tests/simulate_rn1b.py --workers 2 --rounds 1
```

```
network test: d_model=1024 layers=24 vocab=32000 seq=16384
policy: inner_steps=50 micro_batch=1 challenge=25%
workers: 2
```

This is the round-0 architecture at 397,728,768 parameters, on the `test` network
(same model, lighter schedule). It needs about 8 GB of VRAM at `seq_len` 16384 in
bf16 with activation checkpointing.

Useful flags:

```
--workers N     how many workers contribute to each round
--rounds N      how many outer steps
--network       test | regtest
--challenge     issue verification challenges and answer them
--out PATH      write the run's measurements as JSON
```

**On the corpus.** Without one, it synthesises tokens for the round's vocabulary
and says so. That trains a real model on nonsense — fine for measuring throughput
and memory, useless for measuring loss. A corpus tokenized for a different round
is refused rather than used:

```
worker/data/rn1b_tokens.bin holds token id 127999, and this round's vocabulary is
32000. That file was tokenized for a different round.
```

---

## 4. A node and a worker over the real socket

The previous two run everything in one process. This one starts an actual `rnetd`
and drives it through the IPC client — the real binary, not a reimplementation:

```bash
python3 worker/tests/test_ipc_roundtrip.py
```

It exercises the handshake with anchor verification, getting an assignment,
submitting a contribution through a sealed `memfd`, receiving an aggregated
update, applying it, and reporting the resulting weights hash.

To watch a node yourself:

```bash
./build/rnetd --network regtest --worker-id 1 --datadir /tmp/rnet-a --log debug
```

and in another terminal, a second one that connects to it:

```bash
./build/rnetd --network regtest --worker-id 2 --datadir /tmp/rnet-b \
              --port 19556 --connect 127.0.0.1:19555 --log debug
```

Neither will close a round: regtest requires two contributors and neither node has
a worker attached. What you can see is the handshake, the gossip, and the
sixty-second status line.

---

## 5. What a worker actually does, in order

If you are writing your own worker rather than running the harnesses, this is the
sequence. `worker/rn_worker/ipc/client.py` implements it.

1. **Connect and handshake.** The daemon hands over the round and policy
   containers as raw bytes. Hash them against your own compiled-in anchors before
   believing anything in them; a mismatch is `AnchorMismatch` and the connection
   is over. The daemon assigns your worker id — you do not choose it, because a
   worker that could name itself could submit as someone else.

2. **Ask for an assignment.** You get `round_id`, `outer_step`, `assignment_id`,
   `base_checkpoint` and `base_weights_hash`. Your model must be at those weights.

3. **Derive your batches.** For inner step `i`:

   ```python
   seed  = scheduler.window_seed(dataset_root, round_id, worker_id, step, i)
   chunk = scheduler.chunk_for_window(seed, n_chunks)
   toks  = tokenize(corpus_chunk(chunk))        # the pinned tokenizer
   start = scheduler.offset_in_chunk(seed, len(toks), seq_len)
   window = toks[start : start + seq_len + 1]
   ```

   You do not choose any of it. That is the point.

4. **Train.** `inner_steps` steps, deterministic mode on, attention through
   `deterministic_attention(dtype)` — not a backend you picked yourself. Flash and
   math are each reproducible and are not bit-identical to each other, so a worker
   choosing its own would compute a different update and look like a cheat.

5. **Quantise the difference.** `weights_before - weights_after`, flattened in
   canonical parameter order, scaled by a power of two and rounded to int8. The
   scale is an integer exponent so aggregation is integer arithmetic and every node
   gets the same sum.

6. **Submit** through a sealed `memfd`, and be ready to answer a challenge: a
   verifier may ask you to prove you did step 3 the way you claimed.

---

## What stops you joining a network

| | |
| --- | --- |
| **No corpus is pinned** | `dataset_root` is all-zero on every network, so no worker can be told which text to train on. A 4.5 TB corpus is being built; the root gets pinned when it is verified. |
| **No seed resolves** | `seed.resonancenet.org` has no address. Nodes find each other only through `--connect`. |
| **A node cannot catch up** | There is no chain sync. A node joining an established network receives checkpoints whose parents it does not hold and stays where it started. Joining works from genesis, alongside everyone else. |
| **A node cannot re-derive the optimizer** | Momentum is committed to as a hash, not distributed. A node that missed a step follows the chain but will not produce. |
| **Verification is not wired up** | Challenge selection and bit-exact replay work as libraries and are exercised by the simulations. In a running node nothing answers a challenge: `AssignVerify` and `SubmitVerdict` are message types with no handler. |
| **No incentive layer** | Contributions are not rewarded. |

---

## When it goes wrong

**`CUDA error: device-side assert triggered`** — almost always a token id past the
end of the embedding table, meaning a corpus from a different vocabulary. The
loader now refuses those by name; if you built your own path, check
`tokens.max() < vocab_size`.

**`No available kernel`** — flash attention was asked for with fp32 inputs, and
flash kernels are 16-bit only. Real rounds are bf16; fp32 is a test convenience.

**`AnchorMismatch` at the handshake** — your worker and the daemon disagree about
what is being trained. One of them is on a different commit. Compare
`rnet-tool genesis-show` against `genesis.GENESIS_HASH` in the Python mirror.

**Out of memory at `seq_len` 16384** — check that activation checkpointing is on
and the model is bf16. Measured on a 24 GB card: 3.2 GiB for the attention and
activations at this length, against roughly 4 GiB for weights, gradients and the
optimizer.
