# The command line

Two binaries, both built by `cmake --build build`:

- **`rnet-tool`** — offline. Never touches the network. Portable: it builds and
  runs anywhere, including Windows, which is how you verify what a network claims
  without running a node.
- **`rnetd`** — the node. Linux only.

Every command below was run against this build; the outputs are copied, not
paraphrased.

---

## `rnet-tool`

### Common options

```
--network <main|test|regtest>   which consensus rules (default: main)
--log <error|warn|info|debug|trace>   default: info
```

The three networks differ in more than a magic number:

| | main | test | regtest |
| --- | --- | --- | --- |
| Parameters | 983,635,968 | same as main | 2,968,320 |
| `d_model` × `n_layers` | 2048 × 16 | 2048 × 16 | 256 × 4 |
| `seq_len` | 8192 | 8192 | 256 |
| `vocab_size` | 128000 | 128000 | 256 |
| Round deadline | 600 s | 120 s | 1 s |
| Default port | 9444 | 19444 | 19555 |
| DNS seeds | `seed.resonancenet.org` | none | none |

Regtest exists so a full protocol cycle takes seconds on a CPU. Use it for
everything except actually training.

---

### `genesis-show` — what this build believes

```bash
./build/rnet-tool genesis-show --network main
```

Prints the entire consensus state as JSON: model dimensions, policy constants,
and the anchors. The fields you will actually check:

```json
{
  "genesis_hash":  "888a43d234425ca6efb9b802b9faf369beddfea55f3de82513471811b75f75c4",
  "pinned_anchor": "888a43d234425ca6efb9b802b9faf369beddfea55f3de82513471811b75f75c4",
  "policy_hash":   "58d7f7b520ea639c891ed6552c40d8a5809931c9b4a0642d666f67157229dc5c",
  "pinned_policy_anchor": "58d7f7b520ea639c891ed6552c40d8a5809931c9b4a0642d666f67157229dc5c",
  "genesis_weights_hash": "57ecba31c9a2242b62299364040cf3c4bb0dbf9339ead4b4b58bbdbd4a694e2e",
  "parameters": 983635968,
  "round_id": 0,
  "dataset_root": "0000…0000",
  "tokenizer_hash": "11878ae15ef43a42a92e5d02231b780731b39c0067868d66c29f12042919febc",
  "inner_steps": 250,
  "micro_batch": 1,
  "challenge_percent": 10,
  "challenge_deadline_steps": 3,
  "slash_quorum": 3,
  "shadow_mode": true,
  "retained_checkpoints": 5,
  "chunk_tokens": 1048576,
  "default_port": 9444
}
```

**`genesis_hash` must equal `pinned_anchor`, and `policy_hash` must equal
`pinned_policy_anchor`.** The first is computed from the round descriptor this
binary was compiled with; the second is the constant compiled into it. If they
differ, someone changed a consensus parameter without re-pinning, and this build
would train a different model from everyone else.

`dataset_root` being all-zero is the current, expected state: no corpus is pinned
yet, so no worker can be told what to train on.

---

### `genesis-emit` — write the artifacts

```bash
./build/rnet-tool genesis-emit --network regtest --out /tmp/gen
```

```
/tmp/gen/regtest.rnet   48babab7e69b3433a96563af65913423571af5b7be9caf78910c667215d22e51
/tmp/gen/regtest.rnpol  d6783edd2b5f1996a677995219f888335bf8f91d40cd3485a87f1fb5822db8d2

Pin both hashes as the trust anchors for 'regtest'.
```

Two files: `.rnet` is the round descriptor (the model), `.rnpol` is the policy
(the rules). Both are canonical byte containers, and their SHA3-256 is their
identity. A worker verifies the bytes it was handed against the hash compiled
into it — it never trusts a field someone typed out.

CI re-emits all three networks on every run and fails if a hash moves. That is
what makes an accidental parameter change impossible to land quietly.

---

### `genesis-weights` — prove the initial weights are derived

```bash
./build/rnet-tool genesis-weights --network regtest
```

```
deriving 2968320 parameters for 'regtest' — this is real work, not a lookup
derived  f9da0b51a6dda8325d525a1365c053fe33ba456de428d290e901c68a2649e110
anchor   f9da0b51a6dda8325d525a1365c053fe33ba456de428d290e901c68a2649e110

match: this node starts from the state the network agreed on.
```

Initial weights are **derived from the genesis anchor, not distributed**. There is
no file to download and no one to download it from — every node computes the same
tensor from a counter-based SHA3 stream and gets the same hash. Two independent
implementations (C++ and the Python worker) produce byte-identical output, which
the cross-language suite asserts.

On main this derives 983 million parameters and takes a while. That is the point:
it is real work, not a lookup.

---

### `genesis-check` — verify a file you were given

```bash
./build/rnet-tool genesis-check --file /tmp/gen/regtest.rnet --hash 48babab7e69b…
```

Exits non-zero and says what it saw when the file does not hash to the expected
value:

```
error: genesis hash mismatch — refusing untrusted genesis
  file:   48babab7e69b3433a96563af65913423571af5b7be9caf78910c667215d22e51
  anchor: 48babab7
```

Use this when someone hands you an artifact. Both arguments are required; there
is no "trust the file" mode.

---

### `dataset-build` — pin a corpus

```bash
./build/rnet-tool dataset-build \
  --file corpus.bin \
  --out data/corpus \
  --chunk-tokens 65536 \
  --dtype uint32
```

`--chunk-tokens` is the Merkle leaf size. Main's policy fixes it at 1,048,576;
the small value above is what produced the output below, on a 300k-token test
corpus. `--tokenizer-hash` defaults to the network's own and only needs giving
when building a corpus for a tokenizer that is not the pinned one.

```json
{
  "chunk_tokens": 65536,
  "dataset_root": "b4ec7fd56ce566df49c662e4b322f19b18831e9b1a892166dfe59aee89b7c4b2",
  "dtype": "uint32",
  "manifest_id": "da7c9a93bba4632d006f73051f40516d7b2dae0c0468c35f9e69bb32d38f1cf6",
  "n_chunks": 5,
  "n_tokens": 300000,
  "tokenizer_hash": "11878ae15ef43a42a92e5d02231b780731b39c0067868d66c29f12042919febc"
}
```

Writes `<out>.rnds` (the canonical manifest) and `<out>.json` (the same thing,
readable).

`--file` is a flat array of token ids. **This is the one place in the project
that is little-endian**: the tokenizer writes `uint32` little-endian and the C++
side reads it directly, because byte-swapping a multi-terabyte corpus to satisfy
a convention would be a real cost for no benefit. Everything else — every hashed
structure, every wire format — is big-endian.

`dataset_root` is the Merkle root over the chunks. That value is what goes into
the round descriptor, which means **pinning a corpus changes the genesis anchor**.

---

### `dataset-check` — verify a corpus against its manifest

```bash
./build/rnet-tool dataset-check --file corpus.bin --manifest data/corpus.rnds
```

```
OK  300000 tokens, 5 chunks, root b4ec7fd56ce566df49c662e4b322f19b18831e9b1a892166dfe59aee89b7c4b2
```

Re-hashes every chunk and rebuilds the tree. A corpus that has lost a byte
anywhere fails here, naming the chunk.

---

### `schedule-show` — what a worker is told to train on

```bash
./build/rnet-tool schedule-show \
  --manifest data/corpus.rnds --worker 7 --step 100 --network regtest
```

```
worker=7 step=100 seq_len=256 micro_batch=1
  window[0] offset=141145
```

This is the anti-poisoning mechanism made inspectable. The offsets are a pure
function of `(dataset_root, round_id, worker_id, step, index)`, so:

- a worker cannot choose its own data, and
- a verifier can reproduce the exact batch a worker claims it trained on.

Run it for another worker id and the offsets change completely; run it again for
the same one and they are identical, on any machine, in either implementation.

---

## `rnetd`

```
--network <main|test|regtest>   which consensus rules (default: main)
--datadir <path>                state directory (default: ~/.rnet)
--worker-id <n>                 this node's worker identity; required to participate
--connect <host:port>           connect only to this peer; may be given repeatedly
--bind <addr>                   listen address (default: 0.0.0.0)
--port <n>                      listen port (default: the network's own)
--no-listen                     do not accept inbound connections
--no-seeds                      do not query DNS seeds
--max-inbound <n>               inbound connection slots (default: 64)
--max-outbound <n>              outbound connection slots (default: 8)
--corpus <path>                 tokenized corpus file to serve (needs --manifest)
--manifest <path>               the .rnds manifest describing that corpus
--verify                        answer verification challenges assigned to this node
--log <error|warn|info|debug|trace>
--help
```

### `--worker-id` decides what the node is

Without it, `rnetd` is a **relay**: it completes handshakes, gossips objects, and
serves bulk transfers, but constructs no participant, opens no worker socket, and
takes no position on consensus. Useful for a well-connected machine with no GPU.

With it, the node participates: it opens `<datadir>/rnet.sock` for a worker,
accepts contributions, closes rounds, runs the election, and publishes checkpoints
when elected.

The id comes from the node, never from the worker. A worker that could name itself
could submit under someone else's identity, so the daemon assigns it during the
handshake and the worker is told what it is.

### `--connect` disables discovery

Given `--connect`, the node dials exactly those peers and does not query DNS
seeds or its address database. That is what you want for a local test; it is not
what you want in production, where a node that only accepts inbound connections
lets other people decide who it knows — the position an eclipse attack wants it
in.

### `--verify`

Marks this node willing to answer verification challenges assigned to it. Note
the status section of the README: nothing currently sends a verify assignment to
a worker, so this flag reserves the role without yet exercising it.

---

## `ci/run_tests.sh`

```bash
ci/run_tests.sh              # release build, full checks
ci/run_tests.sh debug        # debug build
ci/run_tests.sh sanitize     # address + undefined-behaviour sanitizers
```

In order, it: configures and builds; **diffs the registered test list against
`test/registered_tests.txt`**; runs the suite under ctest; re-emits the genesis
and policy artifacts for all three networks and checks every hash reproduces;
re-derives the genesis weights; and runs the Python cross-language suite against
the real `rnet-tool` binary.

The test-list diff exists because a count is a poor canary. Three tests were once
deleted here by a file being restored over them, and it was nearly missed — the
count went 300 → 297 and someone happened to look. The diff names the casualty.
When you add a test, regenerate the file in the same commit:

```bash
./build/rnet_tests --list > test/registered_tests.txt
```

The sanitized suite takes about six minutes, against roughly forty seconds
uninstrumented, and its ctest timeout is set separately for exactly that reason.
