"""Which peers this node knows, and which it will try next.

The job is not storage. It is resisting the one attack that matters against a
gossip network: filling a node's address table with addresses the attacker
controls, so that every connection it makes goes to the attacker and it sees
whatever the attacker wants it to see. An eclipse.

FOUR THINGS MAKE THAT EXPENSIVE, and none of them is a filter on who may speak.

  * TWO TABLES. Addresses merely heard about live in NEW; addresses this node
    has actually connected to live in TRIED. Gossip can fill NEW for free and
    can never put anything in TRIED, because getting there costs a completed
    handshake from an address that really answers.

  * BUCKETS BY GROUP, NOT BY ADDRESS. A /16 of IPv4 or a /32 of IPv6 shares one
    group, so buying the second address in a netblock buys nothing. Tunnelled
    addresses are grouped by the IPv4 underneath, or a 6to4 range would be a
    free identity factory.

  * THE BUCKET KEY IS SECRET AND PER-NODE. Which bucket an address lands in
    depends on a key this node generated and never sends. Without that an
    attacker computes, offline, the exact set of addresses that collide in the
    buckets it wants to own.

  * A NEW ENTRY DOES NOT DISPLACE AN OLD ONE FOR FREE. A full bucket evicts by
    position, and position is another keyed hash, so an attacker cannot choose
    what it displaces.

WHAT IS DELIBERATELY NOT HERE. No reputation, no scoring by behaviour, no
allowlist. Those need a notion of identity this network does not have, and a
table that could be poisoned by claims about behaviour is worse than one that
can only be poisoned by owning netblocks.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field

from .address import NetAddress, Services, TimestampedAddress

NEW_BUCKETS = 1024
TRIED_BUCKETS = 256
BUCKET_SIZE = 64

# How many groups one source may spread its addresses across in NEW. Without
# this a single peer that gossips ten thousand addresses reaches every bucket.
SOURCE_SPREAD = 64

# Beyond this an address is stale enough that trying it wastes a slot.
HORIZON_MS = 30 * 24 * 60 * 60 * 1000

# A timestamp further ahead than this is clamped. An address stamped in the
# future would sort to the front of every table forever, which is a free way to
# monopolise a node's outbound attempts.
FUTURE_SLACK_MS = 10 * 60 * 1000


class AddrManError(Exception):
    pass


@dataclass
class Entry:
    address: NetAddress
    services: Services
    last_seen_ms: int
    source_group: bytes
    attempts: int = 0
    last_attempt_ms: int = 0
    last_success_ms: int = 0

    @property
    def tried(self) -> bool:
        return self.last_success_ms > 0


@dataclass
class AddrMan:
    """The address table. Deterministic given its key, which is what makes it
    testable and what makes it unpredictable to anyone who does not have it."""

    key: bytes = field(default_factory=lambda: os.urandom(32))
    _entries: dict[tuple, Entry] = field(default_factory=dict)
    _new: dict[int, list[tuple]] = field(default_factory=dict)
    _tried: dict[int, list[tuple]] = field(default_factory=dict)

    # -- keyed placement ----------------------------------------------------

    def _h(self, *parts: bytes) -> int:
        h = hashlib.sha3_256()
        h.update(self.key)
        for p in parts:
            h.update(len(p).to_bytes(4, "big"))
            h.update(p)
        return int.from_bytes(h.digest()[:8], "big")

    def _new_bucket(self, address: NetAddress, source_group: bytes) -> int:
        """Where a heard-about address goes.

        Keyed by BOTH the address's group and the group it was heard from, so
        one gossiping peer cannot place addresses into buckets of its choosing,
        and one netblock cannot occupy many buckets by being announced from many
        places.
        """
        spread = self._h(b"new-spread", address.group, source_group) % SOURCE_SPREAD
        return self._h(b"new", source_group, spread.to_bytes(8, "big")) % NEW_BUCKETS

    def _tried_bucket(self, address: NetAddress) -> int:
        group_slot = self._h(b"tried-slot", address.ip, address.port.to_bytes(2, "big"))
        return self._h(b"tried", address.group,
                       (group_slot % 8).to_bytes(8, "big")) % TRIED_BUCKETS

    def _position(self, table: bytes, bucket: int, address: NetAddress) -> int:
        return self._h(table, bucket.to_bytes(4, "big"), address.ip,
                       address.port.to_bytes(2, "big")) % BUCKET_SIZE

    # -- adding -------------------------------------------------------------

    def add(self, entry: TimestampedAddress, source: NetAddress | None,
            now_ms: int, *, explicit: bool = False) -> bool:
        """Record an address heard from `source`. True if it was new to us.

        A gossiped unroutable address is refused outright rather than stored and
        skipped later: keeping it would let a peer spend our table space on
        addresses that can never be dialled.

        `explicit` is for an address an operator named on the command line. That
        is not hearsay — it is an instruction, and it is often a LAN address or
        one reachable through a tunnel, so the routability filter would refuse
        exactly the case the flag exists for. Such addresses are kept and dialled
        but never gossiped: relaying them would tell strangers the shape of
        somebody's private network. Proved by
        AddrManTests.AnExplicitAddressIsKeptButNeverGossiped.
        """
        address = entry.address
        if address.port == 0:
            return False
        if not explicit and not address.is_routable:
            return False

        stamp = min(entry.last_seen_ms, now_ms + FUTURE_SLACK_MS)
        if stamp < now_ms - HORIZON_MS:
            return False

        key = _key(address)
        existing = self._entries.get(key)
        if existing is not None:
            # Only ever moves forward, and only within what we would accept.
            existing.last_seen_ms = max(existing.last_seen_ms, stamp)
            existing.services |= entry.services
            return False

        source_group = source.group if source is not None else b"\x00local"
        record = Entry(address=address, services=entry.services, last_seen_ms=stamp,
                       source_group=source_group)
        self._entries[key] = record
        self._place(self._new, self._new_bucket(address, source_group), key, b"new")
        return True

    def _place(self, table: dict[int, list], bucket: int, key: tuple,
               label: bytes) -> None:
        slots = table.setdefault(bucket, [])
        if key in slots:
            return
        if len(slots) < BUCKET_SIZE:
            slots.append(key)
            return
        # Full. Evict by keyed position, so an attacker cannot choose its victim.
        victim_slot = self._position(label, bucket, self._entries[key].address)
        evicted = slots[victim_slot % len(slots)]
        slots[victim_slot % len(slots)] = key
        if not self._in_any_table(evicted):
            self._entries.pop(evicted, None)

    def _in_any_table(self, key: tuple) -> bool:
        return (any(key in s for s in self._new.values())
                or any(key in s for s in self._tried.values()))

    # -- results ------------------------------------------------------------

    def attempted(self, address: NetAddress, now_ms: int) -> None:
        entry = self._entries.get(_key(address))
        if entry is not None:
            entry.attempts += 1
            entry.last_attempt_ms = now_ms

    def connected(self, address: NetAddress, now_ms: int) -> None:
        """A completed handshake. This is the only way into TRIED."""
        key = _key(address)
        entry = self._entries.get(key)
        if entry is None:
            return
        entry.last_success_ms = now_ms
        entry.last_seen_ms = max(entry.last_seen_ms, now_ms)
        entry.attempts = 0
        for slots in self._new.values():
            if key in slots:
                slots.remove(key)
        self._place(self._tried, self._tried_bucket(address), key, b"tried")

    def forget(self, address: NetAddress) -> None:
        """Remove an address outright.

        Used when a node discovers it has dialled itself: the address is real,
        answers, and must never be tried again, which is not a state the two
        tables can express.
        """
        key = _key(address)
        self._entries.pop(key, None)
        for table in (self._new, self._tried):
            for slots in table.values():
                if key in slots:
                    slots.remove(key)

    # -- choosing -----------------------------------------------------------

    def select(self, now_ms: int, *, prefer_tried: bool = True,
               exclude: set | None = None) -> NetAddress | None:
        """An address to try next.

        Biased toward TRIED because those are known to answer, but never only
        TRIED: a node that stopped drawing from NEW would keep the same peers
        forever and never notice the network changing around it.
        """
        exclude = exclude or set()
        order = ((self._tried, self._new) if prefer_tried else (self._new, self._tried))
        for table in order:
            picked = self._pick_from(table, now_ms, exclude)
            if picked is not None:
                return picked
        return None

    def _pick_from(self, table: dict[int, list], now_ms: int,
                   exclude: set) -> NetAddress | None:
        best, best_score = None, -1.0
        for bucket, slots in table.items():
            for key in slots:
                entry = self._entries.get(key)
                if entry is None or _key(entry.address) in exclude:
                    continue
                score = self._score(entry, now_ms)
                if score > best_score:
                    best, best_score = entry.address, score
        return best

    def _score(self, entry: Entry, now_ms: int) -> float:
        """Fresher is better, and repeated failure is worse, fast.

        Halving per failed attempt rather than a linear penalty because an
        address that has refused ten times is not ten times worse than one that
        refused once — it is almost certainly gone.
        """
        age_ms = max(0, now_ms - entry.last_seen_ms)
        score = 1.0 / (1.0 + age_ms / 3_600_000.0)
        if entry.attempts:
            score /= float(1 << min(entry.attempts, 16))
        if entry.tried:
            score *= 4.0
        return score

    def to_gossip(self, now_ms: int, limit: int = 1000) -> list[TimestampedAddress]:
        """What to answer a GETADDR with.

        Only addresses seen recently, because relaying a stale one spends
        someone else's table space on a peer that is probably gone.
        """
        fresh = [e for e in self._entries.values()
                 if now_ms - e.last_seen_ms < HORIZON_MS and e.address.is_routable]
        fresh.sort(key=lambda e: e.last_seen_ms, reverse=True)
        return [TimestampedAddress(e.address, e.services, e.last_seen_ms)
                for e in fresh[:limit]]

    # -- accounting ---------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def tried_count(self) -> int:
        return sum(1 for e in self._entries.values() if e.tried)

    @property
    def new_count(self) -> int:
        return len(self._entries) - self.tried_count

    @property
    def groups(self) -> set[bytes]:
        return {e.address.group for e in self._entries.values()}


def _key(address: NetAddress) -> tuple:
    return (address.ip, address.port)
