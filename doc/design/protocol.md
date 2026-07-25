# Protocol design

This document describes the parts that are implemented. Layers still in progress
(networking, aggregation, incentives) are sketched at the end and marked as such.

## 1. Objects and identity

Every object two nodes must agree on is serialized through `canon` and identified
by the SHA3-256 of its container.

```
magic[4]="RNET" | version:u32 | obj_type:u16 | content_len:u32
                | content[content_len]
                | content_hash:SHA3-256[32] | crc32c:u32
```

`crc32c` catches corruption cheaply before structural parsing; `content_hash`
gives content-addressed identity; the container hash is the artifact's name.

Implemented object types:

| Type | Purpose |
| --- | --- |
| `RoundDescriptor` (1) | what to train, how, on what data |
| `DatasetManifest` (2) | a corpus: Merkle root, chunking, tokenizer binding |
| `BatchSeed` (3) | pre-image whose hash selects one training window |

## 2. Trust bootstrap

A node trusts exactly one value a priori: the genesis hash compiled in for its
network. It reads the artifact, hashes it, refuses it unless it matches, and
derives every parameter from the verified contents.

The descriptor carries `network_magic`, so an artifact is bound to its network.
Without that binding, two networks training the same architecture would produce
byte-identical artifacts and one's genesis would be accepted by the other.

## 3. Corpus integrity

A corpus is a flat array of token ids. It is split into fixed **token-count**
chunks and hashed into a Merkle tree with RFC-6962 domain separation:

```
leaf(d)       = SHA3-256(0x00 || d)
internal(l,r) = SHA3-256(0x01 || l || r)
```

Domain separation makes leaves and internal nodes structurally distinct, which
blocks second-preimage substitution and removes the duplicate-node ambiguity
behind Bitcoin's CVE-2012-2459. An odd node is promoted unchanged — safe here,
and unambiguous.

The root is a consensus value referenced by the round descriptor, so a seed node
cannot serve altered data: any consumer verifies a served chunk against the root
with an inclusion proof.

## 4. Deterministic batch derivation

This is the security core.

```
offset_i = be64(SHA3-256(canon(BatchSeed{dataset_root, round_id, worker_id, step, i}))) mod W
W        = n_tokens - seq_len - 1
```

Two properties follow:

**A worker cannot choose its data.** The corpus is pinned by root and the windows
are dictated by the schedule.

**A verifier can reproduce the batch exactly**, which is what makes recomputation
verification possible at all.

### Why not detect bad data instead

An experiment in this project measured whether a held-out validation gate catches
a poisoned contribution. It does not:

| attacker | poison fraction | attack success | held-out loss | gate verdict |
| --- | --- | --- | --- | --- |
| 1 of 4 workers | 20% | 92% | *improved* | accepted |
| 2 of 6, colluding | 10% | 91-94% | improved | accepted (mean, trimmed-mean and centered-clipping alike) |

The backdoor improved the metric the gate watches, because a rare trigger barely
moves clean-language loss. Robust aggregation caught a single scaled outlier but
not colluding submitters who each look ordinary.

So the protocol does not try to detect poisoned data after the fact. It removes
the worker's ability to choose data, and verifies the computation instead.

## 5. Verification model (design, partially implemented)

With data fixed by the schedule, the remaining attack is a dishonest computation:
correct batch, wrong update. Because the batch, the starting weights and the seed
are all known, an honest update is reproducible — so a challenged worker's update
can be recomputed and compared bit for bit.

Constraints that are honest about the limits:

- **Same determinism class only.** Different GPU architectures are not guaranteed
  to produce identical floating-point results. Verification is therefore sharded
  by class; cross-architecture reproducibility is being measured, not assumed.
- **Probabilistic, not absolute.** Only a sampled fraction (`challenge_percent`)
  is recomputed; the guarantee is economic — stake at risk — rather than total.
- **Tolerance comparison is not a substitute.** A loose epsilon reintroduces
  exactly the hiding place the poison experiment exposed, so comparison is
  bit-exact or it is not verification.

## 6. In progress

- **Aggregation (DiLoCo).** Infrequent synchronisation: workers take `inner_steps`
  local steps, then submit a quantised pseudo-gradient for an outer optimizer
  step.
- **Networking.** Peer discovery, corpus and checkpoint distribution, NAT
  traversal.
- **Checkpoint consensus.** A canonical chain of accepted checkpoints, each gated
  on not regressing a secret held-out set, with rollback on later disagreement.
- **Incentives.** Rewards for verifiable actions, settled on an external ledger,
  deliberately last: no token before convergence is demonstrated.
