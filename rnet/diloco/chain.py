"""The checkpoint chain, and how a node picks between two of them.

FORK CHOICE WITHOUT PROOF OF WORK. Bitcoin breaks ties with accumulated work,
which exists because blocks arrive at unpredictable times and anyone may make
one. Here neither is true: a round has a deadline, every node knows when it
closes, and heights advance in lockstep. Two checkpoints at one height are not a
race that one of them won — they are two producers who both applied the round's
contributions and disagreed, or one producer who lied.

So the tie-break IS the rule, and it is: THE LOWER CHECKPOINT ID WINS. Not the
first seen, which depends on the network; not the one with more contributions,
which a producer chooses; not the earliest timestamp, which a producer writes.
The id is the hash of the whole header, so it is self-certifying — every node
computes the same answer from bytes it already has, with no extra round trip and
nothing to trust. A producer who wants its checkpoint to win must find a header
that hashes low, which is grinding a SHA3 for no reward, since the honest
producers' contributions are the same as its own.

A LONGER CHAIN STILL WINS OVER A SHORTER ONE. Height first, id second. A node
that fell behind and receives two steps at once must follow them rather than sit
on a lower head it prefers the hash of.

ORPHANS ARE KEPT, BOUNDED. A checkpoint whose parent is unknown is the normal
first thing a joining node sees, and dropping it — which the implementation this
replaces did — means a node that starts late never catches up, because the
thing it needs in order to ask for the parent is the very message it threw away.
They are held, capped, and connected when the parent arrives.

see: docs/fork-choice.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..consensus.objects import CheckpointHeader

# How many parentless checkpoints to hold while their ancestors are fetched.
# Enough for a deep sync, small enough that a peer cannot make a node hold an
# unbounded amount of unverifiable data by sending headers that lead nowhere.
MAX_ORPHANS = 4096


class Outcome(Enum):
    """What happened to a checkpoint that was offered."""

    EXTENDED = "extended"          # it became the new head
    REORGANISED = "reorganised"    # it replaced the head at the same height
    SIDE_BRANCH = "side"           # valid, connected, not preferred
    ORPHANED = "orphaned"          # parent unknown; held for later
    KNOWN = "known"                # already had it
    REFUSED = "refused"            # malformed: the sender is at fault
    NO_ROOM = "no-room"            # well-formed, but this node is full

    @property
    def blames_the_sender(self) -> bool:
        """Whether this outcome says anything about the peer.

        NO_ROOM does not, and conflating it with REFUSED was a way for a node to
        eclipse itself: an attacker filled the orphan pool once, and from then
        on every honest peer offering a checkpoint this node could not yet
        connect was scored as misbehaving and banned by netblock for a day.
        Proved by ChainTests.AFullPoolDoesNotBlameThePeer.
        """
        return self is Outcome.REFUSED


class ChainError(Exception):
    pass


@dataclass(frozen=True)
class Entry:
    header: CheckpointHeader
    id: bytes
    height: int


@dataclass
class Chain:
    """Every checkpoint this node holds, and which of them is the head."""

    genesis: CheckpointHeader
    retained: int = 16

    _by_id: dict[bytes, Entry] = field(default_factory=dict)
    _head: bytes = field(default=b"")
    _orphans: dict[bytes, list[CheckpointHeader]] = field(default_factory=dict)
    _orphan_count: int = 0

    def __post_init__(self):
        if self.genesis.outer_step != 0 or self.genesis.parent != bytes(32):
            raise ChainError("chain: genesis must be step 0 with no parent")
        if self.retained < 2:
            # One retained checkpoint cannot answer a challenge about the step
            # before it, which is every challenge worth issuing.
            raise ChainError(f"chain: retained {self.retained} must be at least 2")
        entry = Entry(self.genesis, self.genesis.id, 0)
        self._by_id[entry.id] = entry
        self._head = entry.id

    # -- reading -----------------------------------------------------------

    @property
    def head(self) -> Entry:
        return self._by_id[self._head]

    @property
    def height(self) -> int:
        return self.head.height

    def get(self, checkpoint_id: bytes) -> Entry | None:
        return self._by_id.get(checkpoint_id)

    def has(self, checkpoint_id: bytes) -> bool:
        return checkpoint_id in self._by_id

    def at_height(self, height: int) -> Entry | None:
        """The checkpoint at `height` on the chain leading to the head.

        Walked from the head rather than kept as an index, because an index
        would have to be maintained across reorganisations and the walk is at
        most `retained` steps.
        """
        entry = self.head
        if height > entry.height or height < 0:
            return None
        while entry.height > height:
            parent = self._by_id.get(entry.header.parent)
            if parent is None:
                return None     # pruned below this point
            entry = parent
        return entry

    def is_ancestor_of_head(self, checkpoint_id: bytes) -> bool:
        entry = self.get(checkpoint_id)
        return entry is not None and self.at_height(entry.height) is entry

    def missing_parents(self) -> list[bytes]:
        """Parents this node is holding orphans for and does not have.

        What a syncing node asks its peers for. Sorted so two nodes in the same
        state ask in the same order, which makes a stuck sync reproducible.
        """
        return sorted(p for p in self._orphans if p not in self._by_id)

    # -- writing -----------------------------------------------------------

    def add(self, header: CheckpointHeader) -> Outcome:
        """Offer a checkpoint. Never raises for an unwelcome one — the outcome
        is the answer, because "this peer sent something we do not want" is
        ordinary and must not be an exception path."""
        checkpoint_id = header.id
        if checkpoint_id in self._by_id:
            return Outcome.KNOWN

        parent = self._by_id.get(header.parent)
        if parent is None:
            return self._hold_orphan(header)

        if header.outer_step != parent.height + 1:
            # The step is not a hint, it is part of the hashed header, and a
            # header whose step disagrees with its parent's is malformed rather
            # than merely unwanted.
            return Outcome.REFUSED

        entry = Entry(header, checkpoint_id, parent.height + 1)
        self._by_id[checkpoint_id] = entry
        outcome = self._consider(entry)
        self._connect_orphans(checkpoint_id)
        self._prune()
        return outcome

    def _consider(self, entry: Entry) -> Outcome:
        head = self.head
        if entry.height > head.height:
            self._head = entry.id
            return Outcome.EXTENDED
        if entry.height < head.height:
            return Outcome.SIDE_BRANCH
        # Same height. The tie-break is the rule.
        if preferred(entry.id, head.id):
            self._head = entry.id
            return Outcome.REORGANISED
        return Outcome.SIDE_BRANCH

    def _hold_orphan(self, header: CheckpointHeader) -> Outcome:
        if self._orphan_count >= MAX_ORPHANS:
            # Not the sender's fault, and saying so matters: scored as
            # misbehaviour, a full pool turns every honest peer offering an
            # unconnectable checkpoint into a ban.
            return Outcome.NO_ROOM
        waiting = self._orphans.setdefault(header.parent, [])
        # Compared as headers, not as ids. `Object.id` re-serialises the whole
        # container and runs SHA3 on every access, so scanning the list by id
        # made holding N orphans cost N**2 hashes — an unsolicited run of
        # headers froze the event loop for minutes, answering no pings, serving
        # no peers and ignoring the worker socket, for a kilobyte of traffic.
        # Dataclass equality compares eight fields and no bytes are hashed.
        # Proved by ChainTests.HoldingOrphansDoesNotCostQuadraticHashing.
        if header in waiting:
            return Outcome.KNOWN
        waiting.append(header)
        self._orphan_count += 1
        return Outcome.ORPHANED

    def _connect_orphans(self, parent_id: bytes) -> None:
        """Attach anything that was waiting on this checkpoint, and anything
        waiting on those, until nothing more connects."""
        pending = [parent_id]
        while pending:
            waiting = self._orphans.pop(pending.pop(), [])
            for header in waiting:
                self._orphan_count -= 1
                parent = self._by_id.get(header.parent)
                if parent is None or header.outer_step != parent.height + 1:
                    continue
                entry = Entry(header, header.id, parent.height + 1)
                self._by_id[entry.id] = entry
                self._consider(entry)
                pending.append(entry.id)

    def _prune(self) -> None:
        """Drop everything more than `retained` steps below the head.

        The floor is what bounds memory on a long-running node, and the policy
        invariant — a challenge expires before its inputs are pruned — is what
        makes it safe. Anything at or above the floor stays, including side
        branches, because a reorganisation may still need them.
        """
        floor = self.height - self.retained
        if floor <= 0:
            return
        for checkpoint_id, entry in list(self._by_id.items()):
            if entry.height < floor:
                del self._by_id[checkpoint_id]
        for parent_id in [p for p in self._orphans
                          if p not in self._by_id and self._depth_unknown(p)]:
            self._orphan_count -= len(self._orphans.pop(parent_id))

    def _depth_unknown(self, parent_id: bytes) -> bool:
        """An orphan waiting on a parent nobody will ever send again.

        Only the ones whose children are below the retention floor: a parent
        that is simply not here yet must keep its orphans, or a syncing node
        would discard exactly what it is waiting for.
        """
        floor = self.height - self.retained
        return all(h.outer_step <= floor for h in self._orphans.get(parent_id, []))

    def rollback_to(self, checkpoint_id: bytes) -> None:
        """Make `checkpoint_id` the head, discarding what descended from it.

        Used when a reorganisation means work in progress was building on a
        branch that lost.
        """
        entry = self._by_id.get(checkpoint_id)
        if entry is None:
            raise ChainError(f"chain: cannot roll back to an unknown checkpoint")
        self._head = checkpoint_id

    def fork_point(self, a: bytes, b: bytes) -> bytes | None:
        """The deepest checkpoint both branches share.

        What a node needs in order to know how much of its work to abandon.
        """
        left, right = self.get(a), self.get(b)
        if left is None or right is None:
            return None
        while left.height > right.height:
            left = self._by_id.get(left.header.parent)
            if left is None:
                return None
        while right.height > left.height:
            right = self._by_id.get(right.header.parent)
            if right is None:
                return None
        while left.id != right.id:
            left = self._by_id.get(left.header.parent)
            right = self._by_id.get(right.header.parent)
            if left is None or right is None:
                return None
        return left.id


def preferred(candidate: bytes, incumbent: bytes) -> bool:
    """Whether `candidate` beats `incumbent` at the same height.

    Bytewise comparison of the two ids, lower wins. Total, symmetric, and
    computable by anyone holding both headers — which is the entire point:
    every node reaches the same answer without asking anyone.
    """
    if len(candidate) != 32 or len(incumbent) != 32:
        raise ChainError("chain: a checkpoint id is 32 bytes")
    return candidate < incumbent
