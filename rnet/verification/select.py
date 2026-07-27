"""Who gets checked, and why nobody can arrange to be the exception.

A network that verified everything would spend as much compute checking as
training, and one that verified nothing would be a network where lying is free.
So a fraction is checked — and the fraction has to be chosen in a way that
satisfies three things at once, which is harder than it looks:

  * UNPREDICTABLE IN ADVANCE. A worker that knew it would not be checked would
    have no reason to do the work.
  * CHECKABLE AFTERWARDS. Anyone must be able to say whether a given
    contribution SHOULD have been challenged, or "nobody challenged it" and
    "the challenge was quietly dropped" look the same.
  * CHOSEN BY NOBODY. If a node picks, it picks its rivals; if a worker picks,
    it picks itself out.

All three fall out of deriving the decision from the checkpoint id and the
contribution id. The checkpoint id is not known until after every contribution
in it was made — it is a hash over all of them — so no worker can steer it, and
a producer who ground the header to dodge a challenge would be grinding SHA3
against the same rule that decides every other contribution in the same block.

WHY THE CHECKPOINT AND NOT A BEACON. A separate randomness beacon would be one
more thing to agree about and one more thing to attack. The checkpoint is
already agreed, already unpredictable to the people it judges, and already
something every node holds.
"""

from __future__ import annotations

import hashlib

DOMAIN = b"rnet/challenge/v1"


def challenge_draw(checkpoint_id: bytes, contribution_id: bytes) -> int:
    """A number in [0, 100) for this contribution under this checkpoint."""
    if len(checkpoint_id) != 32 or len(contribution_id) != 32:
        raise ValueError("select: ids are 32 bytes")
    digest = hashlib.sha3_256(DOMAIN + checkpoint_id + contribution_id).digest()
    return int.from_bytes(digest[:8], "big") % 100


def should_challenge(checkpoint_id: bytes, contribution_id: bytes,
                     percent: int) -> bool:
    """Whether this contribution is one of the checked ones.

    `percent` comes from the hashed policy, so the rate is consensus too — a
    node that wanted to check less could not do so without being on a different
    network.
    """
    if not 0 <= percent <= 100:
        raise ValueError(f"select: challenge_percent {percent} is out of range")
    if percent == 0:
        return False
    return challenge_draw(checkpoint_id, contribution_id) < percent


def challenged_in(checkpoint_id: bytes, contribution_ids, percent: int) -> list:
    """Every contribution in a checkpoint that should be checked.

    Returned in the order given rather than sorted, because the caller knows
    which order it cares about and sorting here would hide a caller that had
    none.
    """
    return [i for i in contribution_ids
            if should_challenge(checkpoint_id, i, percent)]


def verifier_for(checkpoint_id: bytes, contribution_id: bytes,
                 candidates: list) -> bytes | None:
    """Which worker should check it, from those able to.

    Derived the same way and for the same reason: a verifier that volunteered
    could volunteer for its own accomplice's work, and one that was assigned by
    a node would be assigned by whoever wanted the answer.

    `candidates` must be in an order every node agrees on — sorted ids — or two
    nodes would name different verifiers for the same challenge.
    """
    if not candidates:
        return None
    digest = hashlib.sha3_256(
        DOMAIN + b"/verifier" + checkpoint_id + contribution_id).digest()
    return candidates[int.from_bytes(digest[:8], "big") % len(candidates)]
