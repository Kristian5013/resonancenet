# Security policy

ResonanceNet coordinates untrusted peers and will hold value through its
incentive layer. Treat consensus code accordingly.

## Reporting a vulnerability

**Do not open a public issue for a security problem.** Report privately to the
maintainers; you will get an acknowledgement within 72 hours and an assessment
within seven days.

Please include: affected component, an assessment of impact, and reproduction
steps or a proof of concept if you have one.

## What we consider a vulnerability

Anything that lets a participant:

- make two honest nodes disagree about a consensus object (a training-network
  equivalent of a chain split),
- have a contribution accepted that was not produced by the prescribed
  computation on the prescribed data,
- influence which data a worker trains on outside the deterministic schedule,
- serve corpus content that does not match the published `dataset_root`,
- crash or exhaust the resources of a node with remote input,
- extract or corrupt another participant's stake or rewards.

## Known limitations (by design, not bugs)

These are documented trade-offs, not accepted vulnerabilities to be reported:

- **Spot-recompute verification holds within a determinism class.** Two GPUs of
  different architectures are not guaranteed to produce bit-identical updates, so
  verification is sharded by class. Cross-architecture reproducibility is an open
  measurement, not an assumption.
- **Verification is probabilistic.** Only a sampled fraction of updates is
  recomputed; the guarantee is economic (stake at risk) rather than absolute.
- **Content is not filtered.** The protocol constrains *which* data a worker
  trains on and *how* it computes, not what the corpus says. Corpus curation is a
  separate, social process.

## Cryptography

SHA3-256 (FIPS 202) is the single hash primitive, vendored from tiny_sha3 and
covered by known-answer tests in the suite. Any change to a hashed layout
requires bumping the protocol version — it is a hard fork.
