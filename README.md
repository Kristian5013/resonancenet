# ResonanceNet

A protocol for training a language model the way Bitcoin mines a chain: many
untrusted machines, one agreed result, and every step checkable by anyone who
holds the same inputs.

The distinguishing bet is **bit-exact deterministic replay**. A verifier re-runs
somebody else's training round from the same weights and the same derived
batches and must get byte-identical output. Comparable networks check
*similarity* against empirically-tuned thresholds and call it an open problem;
here the comparison is equality.

```
$ python -m rnet daemon --network regtest &
$ CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m rnet train --network regtest

worker id 1 (assigned by the daemon, not chosen here)
step 1: 4 inner steps from f1f0c82d26889543…
  [####################]  4/4  loss 5.684
step 1: submitted 1d70be0b1a9bd878…, loss 5.6843, 1s
```

---

## What is true today, and what is not

Read this before anything else. It is the part most projects leave out.

| | |
| --- | --- |
| ✅ The protocol runs | rounds close, checkpoints chain, challenges replay and verdicts publish — between two daemons and two workers, on real sockets |
| ✅ Four anchors, self-certifying | genesis, policy, corpus root and initial weights, all checkable with the standard library alone |
| ✅ IPv4 and IPv6 | one address type, two listening sockets, no branch that could treat the families differently |
| ✅ 464 tests | including an adversarial audit's findings, as tests |
| ✅ A corpus reader | chunking, a Merkle index, and a resumable builder. A corpus that indexes to the wrong root is refused, not warned about |
| ❌ **Nobody can reproduce the pinned corpus** | `main`, `test` and `moe` pin FineWeb-Edu at a snapshot that names no revision, and the dataset has since grown from 4.52 TB to 5.84 TB. Root `7195da13…` cannot be rebuilt from upstream today. **Only `regtest` runs end to end**, on synthetic tokens |
| ❌ No chain persistence | a restarted node rebuilds from genesis and cannot rejoin a network past its retention window |
| ❌ No mixture-of-experts training | the 29.4B shape is described, pinned and tested; how workers divide a sharded mixture is unsolved |
| ❌ No seed, no network | `seed.resonancenet.org` resolves to a released address. Nodes find each other only with `--connect` |
| ❌ No incentive layer | contributions are not rewarded. This is not a code problem |

If you want to see the protocol work, `regtest` does that today. If you want to
contribute compute to a shared model, there is nothing to contribute to yet.

---

## Install

Python 3.12 or newer. The consensus half needs **nothing outside the standard
library** — that is deliberate: checking what a network claims should not
require the hardware to participate in it.

```bash
git clone https://github.com/Kristian5013/resonancenet rnet && cd rnet
python3.12 -m venv .venv                # name the interpreter — see below
.venv/bin/pip install -e .              # numpy only
```

Name the version rather than relying on `python3`. On a machine whose `python3`
is 3.14 the venv step fails outright with `ensurepip is not available`, which
looks like this project's problem and is not.

Training additionally needs PyTorch with CUDA:

```bash
.venv/bin/pip install -e '.[train]'     # torch, tokenizers
# or, for a specific CUDA build:
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Building a corpus needs two more, and only building does:

```bash
.venv/bin/pip install -e '.[corpus]'    # huggingface_hub, pyarrow
```

**Hardware.** `regtest` runs on anything, including CPU. Round 0 of `main` is a
397,728,768-parameter model at a 16,384-token context: **24 GB of VRAM**, bf16,
with activation checkpointing. Less will not fit.

---

## Check what the network claims, without running it

This needs no GPU, no torch, and no trust in this program beyond the anchors it
is holding itself to.

```bash
python -m rnet genesis-show main
```

```
network        main  (magic 0x524e4d31)
genesis        adfd6082694c614cf44b2490df78ba15efc51ddc1174dd5740506c80ff4f9597
policy         d0de7064ef6f9ae67133958c9d8a93df06b42c72f19789d557adee4243e39431
weights        26358eaeb57666cf9e5d5fa59106ab407e5dbbd0f67da925f040abd064bdb37d  (derived, not shipped)
model          d_model 1024, 24 layers, 8 heads (head_dim 128), GQA 4:1, seq_len 16384, vocab 32000 — dense, 397,728,768 parameters
arithmetic     bf16 params, bf16 compute, fp32 accumulate, flash attention, int8 contributions (class 0x1730f203)
corpus         7195da139188f4a1… (6,956,933 chunks)
tokenizer      1f8d0c4bc23d000c…
schedule       200 inner steps, 2 contributor(s) minimum, 1200s deadline
verification   25% challenged, quorum 3, shadow mode
```

Every one of those hashes is derivable rather than announced:

```bash
python -m rnet genesis-verify        # artifacts on disk against the anchors
python -m rnet genesis-weights       # DERIVE the initial weights and check them
```

`genesis-weights` is the interesting one. Nobody ships the starting weights —
they are a pure function of the genesis hash, so anyone can recompute them and
check the network started where it says it did. 1.5 seconds for the dense 400M,
85 seconds for the 29.4-billion-parameter mixture.

---

## Run a node

```bash
python -m rnet daemon --network regtest --datadir ~/.rnet/regtest
```

```
rnet regtest: listening on [::]:9444, 0.0.0.0:9444
  workers  /home/you/.rnet/regtest/worker.sock
  datadir  /home/you/.rnet/regtest
  genesis  153cc31ec891b6dc0ff66107bc0d7c70…
  weights  f1f0c82d26889543586f6a5bbe76d301…
  peers    0 known
[   1.0m] peers 0 (0 ready, 0 in, 0 out) — addresses 0 — chain 0 — objects 0
```

Two nodes on one machine, over IPv6:

```bash
python -m rnet daemon --datadir /tmp/a --port 19444
python -m rnet daemon --datadir /tmp/b --port 19445 --connect '[::1]:19444'
```

Useful flags: `--connect HOST:PORT` (repeatable; brackets required for IPv6),
`--no-v4` / `--no-v6`, `--max-outbound`, `--max-inbound`, `--status-interval`.

---

## Attach a worker

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8      # MUST be set before python starts
python -m rnet train --network regtest --datadir ~/.rnet/regtest
```

That variable is not optional and cannot be set from inside the program: torch
reads it when CUDA initialises, which happens at import. Without it cuBLAS picks
workspaces by heuristic and the same batch gives different bytes on the same
card — which is precisely the property everything here rests on.

**`regtest` needs two contributors** to close a round. Start a second worker in
another terminal; both attach to the same daemon.

Useful flags: `--device cpu` (no GPU needed, and then the variable above is not
needed either), `--rounds N` to stop after N, `--lr`, and `--corpus PATH` for a
network that pins one.

```
step 1: applying 2 contribution(s)
step 1: checkpoint c0a52a89cbb050f4 (extended), weights 740d844658cc9f31
step 3: challenging 971bfbad39849fd7 from worker 1
verdict MATCH from worker 2 on 2ecfc67bd5689cca (shadow mode)
```

---

## How it works

Five things, each documented in `docs/`:

**[The four anchors](docs/anchors.md)** — a node is compiled knowing two hashes
per network and believes an artifact only if it hashes to one of them.
Everything else it accepts arrives inside an artifact that cleared that check.

**[Deriving the batches](docs/corpus.md)** — a worker's training data is a pure
function of protocol state: corpus root, round, its own assigned id, the step.
Nobody chooses it, so "you trained on data you picked yourself" stops being an
accusation nobody can settle.

**[The round](docs/rounds.md)** — inner loop, int8 pseudo-gradients, integer
aggregation, a Q16 outer step, and a chain whose fork-choice rule is the
tie-break because there is no proof of work to break ties with.

**[Verification](docs/verification.md)** — who gets challenged is derived from
the checkpoint id, so it is unpredictable in advance, checkable afterwards, and
chosen by nobody. A quorum of distinct verifiers agreeing on a mismatch becomes
evidence.

**[The numerics](docs/numerics.md)** — bf16 through fp8, int4 through int8, flash
or math attention, all pinned per round. Flash and math are each reproducible
and are *not* bit-identical to each other, so a worker choosing its own would
produce an update nobody could reproduce.

---

## Tests

```bash
./ci/run_tests.sh
```

```
python: /path/to/rnet/.venv/bin/python 3.12.13
Ran 464 tests in 55.5s
OK
```

A green run needs the `train` extra — three modules test the training half and
skip without torch (`OK (skipped=3)` on a numpy-only install, 425 tests).

The script exists for one thing a test cannot arrange for itself: exporting
`CUBLAS_WORKSPACE_CONFIG` before python starts. Without a CUDA device the
round tests fall back to CPU and take about three minutes.

---

## Contributing

Read `CONTRIBUTING.md`. The short version: every asserting comment names the
test that proves it, and a comment naming a test that passes vacuously is worse
than no comment at all — it buys false confidence. Several of the tests in this
tree exist because an audit found a claim that was not true.

## License

MIT. See `LICENSE`.
