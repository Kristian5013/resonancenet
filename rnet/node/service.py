"""What a node does with what its peers say.

The transport layer knows about sockets and framing and nothing about what a
checkpoint is; the consensus layer knows what a checkpoint is and nothing about
sockets. This is the seam, and keeping it thin is what stops a networking bug
from becoming a consensus bug.

THREE CONVERSATIONS, and they are separate because they fail differently:

  * RELAY — inv, getdata, object. Announcements are cheap and data is not, so a
    peer announces and this node decides. It never fetches the same object from
    two peers at once, and it never accepts one it did not ask for.

  * SYNC — getheaders, headers. A node that fell behind walks back through a
    locator to find the last checkpoint it shares with a peer, then takes the
    chain forward from there. This is what makes joining late possible, and its
    absence is what made the implementation this replaces unable to.

  * CORPUS — getchunk, chunk. Bulk, verified against the round's pinned root
    before a single byte of it is believed. The proof travels with the data
    because a chunk without one is bytes nobody can place.

NOTHING ARRIVING HERE IS TRUSTED. An object is a candidate until it hashes to
what was asked for; a header is a candidate until its parent is known; a chunk
is a candidate until its proof verifies against the manifest's root and the
manifest's width. Peers that send otherwise are scored, not argued with.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..canon.container import ObjectType
from ..consensus.objects import CheckpointHeader
from ..crypto import merkle
from ..diloco.chain import Chain, Outcome
from ..net import protocol as P
from ..net.peer import Peer, Score
from .objects import ObjectStore, ObjectStoreError

# How long a getdata may stay outstanding before the object may be asked of
# somebody else. Long enough for a slow peer, short enough that one peer cannot
# stall a sync by promising and not delivering.
REQUEST_TIMEOUT_S = 60.0

# One peer may owe this many objects at once. Beyond it the node is not fetching
# from a peer, it is being led by one.
MAX_IN_FLIGHT_PER_PEER = 128

# A locator goes back exponentially, so a few dozen entries reach genesis from
# any height a node will ever be at.
LOCATOR_DENSE = 10


@dataclass
class InFlight:
    peer_id: int
    requested_at: float
    inv_type: P.InvType


@dataclass
class Service:
    """Binds a transport Node to a Chain and an ObjectStore."""

    node: object
    chain: Chain
    objects: ObjectStore = field(default_factory=ObjectStore)
    corpus: object = None
    dataset_root: bytes = bytes(32)
    dataset_chunks: int = 0

    _in_flight: dict = field(default_factory=dict)
    _announced: dict = field(default_factory=dict)

    def install(self) -> None:
        self.node.on_message = self.handle

    # -- dispatch ------------------------------------------------------------

    async def handle(self, peer: Peer, message: P.Message) -> None:
        if isinstance(message, P.Inv):
            self._on_inv(peer, message)
        elif isinstance(message, P.GetData):
            self._on_getdata(peer, message)
        elif isinstance(message, P.Object):
            self._on_object(peer, message)
        elif isinstance(message, P.NotFound):
            self._on_notfound(peer, message)
        elif isinstance(message, P.GetHeaders):
            self._on_getheaders(peer, message)
        elif isinstance(message, P.Headers):
            self._on_headers(peer, message)
        elif isinstance(message, P.GetChunk):
            self._on_getchunk(peer, message)
        elif isinstance(message, P.Chunk):
            self._on_chunk(peer, message)

    # -- relay ----------------------------------------------------------------

    def _on_inv(self, peer: Peer, message: P.Inv) -> None:
        """"I have these." Decide what is worth asking for."""
        if len(message.entries) > P.MAX_INV:
            self.node.penalise(peer, Score.FLOOD, "oversized inv")
            return
        outstanding = sum(1 for f in self._in_flight.values()
                          if f.peer_id == getattr(peer, "id", -1))
        wanted = []
        for entry in message.entries:
            if entry.hash in self.objects or entry.hash in self._in_flight:
                continue
            if self.chain.has(entry.hash):
                continue
            if outstanding + len(wanted) >= MAX_IN_FLIGHT_PER_PEER:
                break
            wanted.append(entry)
        if not wanted:
            return
        now = time.monotonic()
        for entry in wanted:
            self._in_flight[entry.hash] = InFlight(
                peer_id=getattr(peer, "id", -1), requested_at=now,
                inv_type=entry.type)
        peer.send(P.GetData(tuple(wanted)))

    def _on_getdata(self, peer: Peer, message: P.GetData) -> None:
        """"Send me these." Answer with what is held, and say so when it is not.

        Answering NotFound explicitly matters: silence leaves the requester
        unable to tell a peer that went quiet from one that never had the
        object, so it waits out a timeout for both.
        """
        missing = []
        for entry in message.entries:
            held = self.objects.get(entry.hash)
            if held is not None:
                peer.send(P.Object(container=held.container))
                continue
            chain_entry = self.chain.get(entry.hash)
            if chain_entry is not None:
                peer.send(P.Object(container=chain_entry.header.to_container()))
                continue
            missing.append(entry)
        if missing:
            peer.send(P.NotFound(tuple(missing)))

    def _on_object(self, peer: Peer, message: P.Object) -> None:
        """An object arrived. It is a candidate until it hashes right."""
        try:
            held = self.objects.put(message.container)
        except ObjectStoreError as exc:
            self.node.penalise(peer, Score.UNKNOWN_OBJECT, str(exc))
            return

        object_id = held.parsed.id
        flight = self._in_flight.pop(object_id, None)
        if flight is None:
            # Not asked for. Kept — it parsed and hashes to itself, so it costs
            # what it costs — but scored, because a peer that pushes objects is
            # using this node's memory as its own.
            self.objects.forget(object_id)
            self.node.penalise(peer, Score.UNSOLICITED_OBJECT,
                               f"unsolicited {held.obj_type.name}")
            return

        if held.obj_type is ObjectType.CHECKPOINT_HEADER:
            self._offer_checkpoint(peer, held.parsed)

        # Anything accepted is worth announcing onward, but only the fact of it.
        self.node.broadcast(
            P.Inv((P.InvEntry(flight.inv_type, object_id),)), skip=peer)

    def _on_notfound(self, peer: Peer, message: P.NotFound) -> None:
        """A peer withdrew an announcement. Free the slot so another may serve."""
        for entry in message.entries:
            flight = self._in_flight.get(entry.hash)
            if flight is not None and flight.peer_id == getattr(peer, "id", -1):
                del self._in_flight[entry.hash]

    def expire_requests(self, now: float | None = None) -> int:
        """Release objects a peer promised and did not send.

        Without this one peer stalls a sync permanently by announcing
        everything and delivering nothing, and the node never asks anyone else.
        """
        now = time.monotonic() if now is None else now
        stale = [h for h, f in self._in_flight.items()
                 if now - f.requested_at > REQUEST_TIMEOUT_S]
        for h in stale:
            del self._in_flight[h]
        return len(stale)

    # -- chain ----------------------------------------------------------------

    def _offer_checkpoint(self, peer: Peer, header: CheckpointHeader) -> None:
        outcome = self.chain.add(header)
        if outcome is Outcome.ORPHANED:
            # The parent is unknown, so ask for the chain rather than for the
            # one parent: a node that fell behind by a hundred steps would
            # otherwise need a hundred round trips.
            peer.send(P.GetHeaders(locators=tuple(self.locator()),
                                   stop=bytes(32)))
        elif outcome in (Outcome.EXTENDED, Outcome.REORGANISED):
            self.node.height = self.chain.height

    def locator(self) -> list[bytes]:
        """Checkpoint ids going back exponentially from the head.

        Dense near the tip, sparse further back, so two nodes find their common
        ancestor in one round trip whether they diverged one step ago or a
        thousand.
        """
        out = []
        height = self.chain.height
        step = 1
        seen = 0
        while height >= 0:
            entry = self.chain.at_height(height)
            if entry is None:
                break
            out.append(entry.id)
            if len(out) >= P.MAX_LOCATORS:
                break
            seen += 1
            if seen > LOCATOR_DENSE:
                step *= 2
            height -= step
        genesis = self.chain.at_height(0)
        if genesis is not None and genesis.id not in out:
            if len(out) >= P.MAX_LOCATORS:
                out[-1] = genesis.id
            else:
                out.append(genesis.id)
        return out

    def _on_getheaders(self, peer: Peer, message: P.GetHeaders) -> None:
        """Answer from the first locator this node recognises."""
        start = 0
        for candidate in message.locators:
            entry = self.chain.get(candidate)
            if entry is not None and self.chain.is_ancestor_of_head(candidate):
                start = entry.height + 1
                break
        headers = []
        height = start
        while height <= self.chain.height and len(headers) < P.MAX_HEADERS:
            entry = self.chain.at_height(height)
            if entry is None:
                break
            headers.append(entry.header.to_container())
            if entry.id == message.stop:
                break
            height += 1
        if headers:
            peer.send(P.Headers(tuple(headers)))

    def _on_headers(self, peer: Peer, message: P.Headers) -> None:
        """Take a run of checkpoints, in order.

        In order matters: each one's parent should be the one before it, so a
        run that connects does so in a single pass instead of leaving the middle
        of it orphaned.
        """
        accepted = 0
        for raw in message.headers:
            try:
                header = CheckpointHeader.from_container(raw)
            except Exception as exc:
                self.node.penalise(peer, Score.UNKNOWN_OBJECT, f"bad header: {exc}")
                return
            outcome = self.chain.add(header)
            if outcome is Outcome.REFUSED:
                self.node.penalise(peer, Score.WRONG_HASH, "header refused by the chain")
                return
            if outcome in (Outcome.EXTENDED, Outcome.REORGANISED, Outcome.SIDE_BRANCH):
                accepted += 1
        self.node.height = self.chain.height
        # A full batch means there is probably more; ask again from the new tip.
        if accepted and len(message.headers) >= P.MAX_HEADERS:
            peer.send(P.GetHeaders(locators=tuple(self.locator()), stop=bytes(32)))

    # -- corpus ----------------------------------------------------------------

    def _on_getchunk(self, peer: Peer, message: P.GetChunk) -> None:
        if self.corpus is None or message.dataset_root != self.dataset_root:
            peer.send(P.Reject(P.Command.GETCHUNK, P.RejectCode.UNAVAILABLE,
                               "no corpus for that root"))
            return
        try:
            data, proof = self.corpus.chunk_with_proof(message.index)
        except Exception:
            peer.send(P.Reject(P.Command.GETCHUNK, P.RejectCode.UNAVAILABLE,
                               f"chunk {message.index} unavailable"))
            return
        peer.send(P.Chunk(dataset_root=self.dataset_root, index=message.index,
                          leaf_count=proof.leaf_count, proof=proof.path, data=data))

    def _on_chunk(self, peer: Peer, message: P.Chunk) -> bool:
        """Corpus bytes. Verified before they are believed, never after.

        The width comes from the manifest rather than from the message, because
        a proof does not pin the width it was built at — different widths can
        produce an identical walk for a given index — so taking it from the peer
        would let it choose which tree its proof is against.
        """
        if message.dataset_root != self.dataset_root:
            self.node.penalise(peer, Score.BAD_PROOF, "a chunk for another corpus")
            return False
        if self.dataset_chunks and message.leaf_count != self.dataset_chunks:
            self.node.penalise(
                peer, Score.BAD_PROOF,
                f"claims {message.leaf_count} chunks, the manifest says "
                f"{self.dataset_chunks}")
            return False
        proof = merkle.Proof(index=message.index,
                             leaf_count=self.dataset_chunks or message.leaf_count,
                             path=tuple(message.proof))
        if not merkle.verify_proof(merkle.leaf_hash(message.data), proof,
                                   self.dataset_root):
            self.node.penalise(peer, Score.BAD_PROOF,
                               f"chunk {message.index} does not prove")
            return False
        if self.corpus is not None:
            self.corpus.store(message.index, message.data)
        return True

    # -- announcing ------------------------------------------------------------

    def announce(self, inv_type: P.InvType, object_id: bytes) -> int:
        return self.node.broadcast(P.Inv((P.InvEntry(inv_type, object_id),)))

    def publish(self, obj) -> int:
        """Store an object this node made and tell everyone it exists."""
        self.objects.put(obj.to_container())
        kind = {
            ObjectType.CHECKPOINT_HEADER: P.InvType.CHECKPOINT,
            ObjectType.CONTRIBUTION_HEADER: P.InvType.CONTRIBUTION,
            ObjectType.CHALLENGE_ORDER: P.InvType.CHALLENGE,
            ObjectType.VERIFICATION_VERDICT: P.InvType.VERDICT,
            ObjectType.SLASH_EVIDENCE: P.InvType.SLASH_EVIDENCE,
            ObjectType.EXPERT_SHARD: P.InvType.EXPERT_SHARD,
        }.get(obj.OBJ_TYPE)
        if kind is None:
            raise ValueError(f"service: {obj.OBJ_TYPE.name} is not relayed")
        return self.announce(kind, obj.id)

    def status(self) -> str:
        return (f"chain {self.chain.height} — objects {len(self.objects)} "
                f"({self.objects.total_bytes // 1024} KB) — "
                f"{len(self._in_flight)} in flight")
