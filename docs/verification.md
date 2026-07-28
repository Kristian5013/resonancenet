# Verification: proving somebody did the work

A worker claimed that starting from these weights, with this id, at this step,
for this many inner steps, it produced an update hashing to X. Every one of
those inputs is public and derivable, so a verifier does the same work and sees
whether it gets X.

```
step 3: challenging 971bfbad39849fd7 from worker 1
verdict MATCH from worker 2 on 2ecfc67bd5689cca (shadow mode: recorded, nothing punished)
```

## What makes it decidable

The batches are a pure function of protocol state, so the verifier trains on
exactly the same text without being told what it is — which is why a worker
cannot pick data it likes. And the arithmetic is pinned by the round's numerics,
so two machines in the same determinism class produce identical bytes rather
than similar ones.

Comparable networks compare *similarity* — Jaccard, Manhattan and Hamming
distances against empirically-set thresholds, which their own documentation
calls an open problem. Here the comparison is equality.

## Who gets checked, and why nobody can arrange to be the exception

The draw comes from the checkpoint id and the contribution id:

```python
should_challenge = sha3("rnet/challenge/v1" ‖ checkpoint_id ‖ contribution_id)[:8] % 100 < percent
```

Three properties at once, which is harder than any one of them:

- **Unpredictable in advance.** The checkpoint id is a hash over every
  contribution in it and does not exist until they all do, so no worker can
  steer whether it is picked.
- **Checkable afterwards.** Anyone can say whether a contribution *should* have
  been challenged — otherwise "nobody challenged it" and "the challenge was
  quietly dropped" look the same.
- **Chosen by nobody.** A node would pick its rivals; a worker would pick itself
  out.

The verifier is drawn the same way and for the same reason.

## The payload does not travel

Only the claimed hash. The verifier computes its own and compares — so a
challenge costs 32 bytes rather than 400 MB, and a verifier cannot be fooled by
a payload that hashes to the claim without being what was computed.

## INDETERMINATE is an answer

A verifier in a different determinism class, holding different base weights, or
without the corpus the round pins, says so. Reporting MISMATCH instead would
slash an honest worker for the verifier's own circumstances, and a scheme that
does that punishes the wrong people while looking rigorous.

An indeterminate answer **leaves the challenge open** for somebody who can judge
it. Closing it would let a verifier retire a challenge by being unable to answer
— the cheapest way to protect a friend.

## Nobody judges their own work

Refusing to hand a worker its own challenge is not enough: a verdict is a
message it can send unprompted, so without an explicit check it clears itself of
a question it was never asked.

## From verdicts to evidence

One verdict is never enough. A verifier can be wrong, malicious, or on hardware
that rounds differently, and a protocol that acted on one report is one where
accusing costs less than working.

A quorum of **distinct verifiers** agreeing on MISMATCH becomes `SlashEvidence`.
Three rules, each closing a way to cheat the count:

| rule | what it closes |
| --- | --- |
| only MISMATCH counts | INDETERMINATE convicts a worker for its verifiers' hardware |
| distinct **verifiers**, not verdict objects | one verifier can make many objects; a quorum of three would be a quorum of one |
| same determinism class | two machines that were never going to agree, counted as though they had |

A challenge settles when a quorum agrees **either way** — guilt or innocence —
and otherwise stays open until it expires. Closing on the first answer makes the
quorum unreachable by construction: the second verifier is told there is no such
challenge.

## A two-worker network cannot prove anything

The quorum is two and nobody judges their own work, so every challenge has
exactly one eligible verifier. That is a property of the network, not of any
test.

## What slashing means today

There is no stake, so there is nothing to take. What evidence buys is that a
node stops spending its aggregate on a worker proven to have submitted work it
did not do — and stops handing it challenges to judge, because a worker that
fabricated its own round is exactly the one that will fabricate a verdict.

That is real. It is **not** economic punishment, and calling it slashing without
saying so is the kind of overclaim that survives until somebody depends on it.

## Shadow mode

Every shipped network records verdicts and punishes nothing. `shadow_mode` lives
in the hashed policy, so turning enforcement on is a fork rather than a flag —
which is the right shape for a rule whose first act, historically, is to be
wrong about somebody.
