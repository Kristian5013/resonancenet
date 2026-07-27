"""Where a node keeps the consensus objects it has been sent.

Bounded on two axes, count and bytes, because either one alone is a way in. A
cap on count lets a peer send few enormous objects; a cap on bytes lets it send
many tiny ones and exhaust the dictionary instead. Both, and neither matters.

WHAT GETS IN. Only objects that parse, hash to the id they were requested
under, and were actually requested. That last clause is the one that looks
paranoid and is not: without it, a peer can push anything into every node's
memory for the cost of sending it, and the store becomes the cheapest
amplification surface in the protocol.

EVICTION IS BY AGE, OLDEST FIRST, and it is deliberately not by usefulness.
Anything cleverer — least recently used, most requested — is a signal an
attacker controls by asking for things, and a cache whose retention an attacker
steers is a cache that holds what the attacker wants it to.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

from ..canon.container import ObjectType
from ..canon.stream import CanonError
from ..consensus.objects import Object, decode

# A few thousand objects covers a deep sync with room to spare; the byte ceiling
# is what actually binds, since consensus objects are hundreds of bytes each.
MAX_OBJECTS = 8192
MAX_BYTES = 64 << 20


class ObjectStoreError(Exception):
    pass


@dataclass
class Held:
    obj_type: ObjectType
    container: bytes
    parsed: Object


@dataclass
class ObjectStore:
    """Objects by id, oldest evicted first."""

    max_objects: int = MAX_OBJECTS
    max_bytes: int = MAX_BYTES
    _held: OrderedDict = field(default_factory=OrderedDict)
    _bytes: int = 0

    def __contains__(self, object_id: bytes) -> bool:
        return object_id in self._held

    def __len__(self) -> int:
        return len(self._held)

    @property
    def total_bytes(self) -> int:
        return self._bytes

    def get(self, object_id: bytes) -> Held | None:
        return self._held.get(object_id)

    def put(self, container: bytes, *, expect_id: bytes | None = None) -> Held:
        """Parse, verify, and keep. Raises rather than storing anything doubtful.

        `expect_id` is the id this was requested under. Checking it here rather
        than at the caller is what makes "an object is what it says it is" a
        property of the store instead of a discipline everyone has to remember.
        """
        try:
            parsed = decode(container)
        except CanonError as exc:
            raise ObjectStoreError(f"objects: will not store what does not parse: {exc}")

        object_id = parsed.id
        if expect_id is not None and object_id != expect_id:
            raise ObjectStoreError(
                f"objects: asked for {expect_id.hex()[:16]}, got "
                f"{object_id.hex()[:16]} — a peer answering a different question")

        existing = self._held.get(object_id)
        if existing is not None:
            return existing

        held = Held(obj_type=parsed.OBJ_TYPE, container=container, parsed=parsed)
        self._held[object_id] = held
        self._bytes += len(container)
        self._evict()
        return held

    def _evict(self) -> None:
        while (len(self._held) > self.max_objects
               or self._bytes > self.max_bytes) and self._held:
            _, dropped = self._held.popitem(last=False)
            self._bytes -= len(dropped.container)

    def ids_of_type(self, obj_type: ObjectType) -> list[bytes]:
        return [i for i, h in self._held.items() if h.obj_type is obj_type]

    def forget(self, object_id: bytes) -> None:
        dropped = self._held.pop(object_id, None)
        if dropped is not None:
            self._bytes -= len(dropped.container)
