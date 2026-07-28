"""Weights on disk, named by what they hash to.

The chain records what the weights became; nothing held the weights themselves,
so a machine that rebooted after a week of training had a chain saying where it
got to and no way to be there. This is the other half.

CONTENT ADDRESSED, and the file name is the verification. A file called
`0ea5ef97….rnw` claims to be the weights whose Merkle root is `0ea5ef97…`, and
loading it recomputes that root and compares. There is no separate checksum
because there is nothing a separate checksum would catch that this does not.

A CACHE, UNLIKE THE CHAIN. A corrupt or missing weights file is recoverable —
derive them again from the genesis hash, or ask a peer that has the checkpoint
— so this discards quietly and says so, where the chain store refuses to start.
The distinction is whether the thing lost can be got back.

THE BYTES ARE THE ONES THE COMMITMENT IS OVER. Layout order, bf16 words,
big-endian: exactly what `hash_of_values` builds its Merkle leaves from. That is
not a coincidence to be maintained, it is the point — a file whose bytes differ
from what the root is computed over would hash correctly and load wrongly, and
serving a tensor to a peer would mean re-deriving what to send.
"""

from __future__ import annotations

import os
import re

import numpy as np

from ..canon.stream import Reader, Writer
from ..consensus.init import hash_of_values
from ..consensus.model_spec import ModelSpec
from ..crypto import merkle
from .layout import layout

MAGIC = b"RNWT"
VERSION = 1
SUFFIX = ".rnw"

# How many checkpoints' weights to keep. Two is the floor that means anything:
# the head, and the one before it, which is what a challenge about the last
# round has to be replayed against. Each is 759 MiB for the dense 400M.
DEFAULT_RETAIN = 3

_NAME = re.compile(r"^[0-9a-f]{64}\.rnw$")


class StoreError(Exception):
    pass


def canonical_bytes(spec: ModelSpec, tensors: dict) -> bytes:
    """One tensor's worth at a time is what `write` does; this is for tests."""
    return b"".join(tensor_bytes(tensors, t) for t in layout(spec))


def tensor_bytes(tensors: dict, descriptor) -> bytes:
    """The canonical bytes of one tensor, which is also its Merkle leaf's
    pre-image."""
    value = tensors.get(descriptor.name)
    if value is None:
        raise StoreError(f"weights: {descriptor.name} is missing")
    if value.size != descriptor.numel:
        raise StoreError(
            f"weights: {descriptor.name} has {value.size} elements, the layout "
            f"says {descriptor.numel}")
    return np.ascontiguousarray(value, dtype=np.uint16).astype(">u2").tobytes()


def path_for(directory: str, root: bytes) -> str:
    return os.path.join(directory, root.hex() + SUFFIX)


def write(directory: str, spec: ModelSpec, tensors: dict) -> tuple[str, bytes]:
    """Store `tensors`, returning where they went and what they hashed to.

    Streamed tensor by tensor. The straightforward version built the whole
    759 MiB as one bytes object first, which is the same mistake that had a
    worker OOM-killed at the submit — and here it would be paid on every round.
    """
    order = layout(spec)
    root = hash_of_values(spec, tensors)
    path = path_for(directory, root)
    if os.path.exists(path):
        return path, root                      # already have exactly these

    os.makedirs(directory, exist_ok=True)
    tmp = path + ".tmp"
    header = (Writer().raw(MAGIC).u16(VERSION).raw(root).u32(len(order))).take()
    try:
        with open(tmp, "wb") as f:
            f.write(header)
            for descriptor in order:
                f.write(tensor_bytes(tensors, descriptor))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        # A half-written file whose name promises a root it does not hash to is
        # worse than no file: the next load would spend a minute proving it.
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return path, root


def read(path: str, spec: ModelSpec, *, expect: bytes | None = None) -> dict:
    """Load weights and prove they are what the name says.

    Raises `StoreError` for anything wrong, which the caller is expected to
    treat as "derive or fetch instead" rather than as fatal.
    """
    order = layout(spec)
    expected_bytes = sum(t.numel for t in order) * 2

    try:
        with open(path, "rb") as f:
            head = f.read(4 + 2 + 32 + 4)
            r = Reader(head)
            if r._take(4) != MAGIC:
                raise StoreError(f"weights: {path} is not a weights file")
            version = r.u16()
            if version != VERSION:
                raise StoreError(
                    f"weights: {path} is version {version}, this build reads "
                    f"{VERSION}")
            claimed = r._take(32)
            count = r.u32()
            if count != len(order):
                raise StoreError(
                    f"weights: {path} holds {count} tensors and this model has "
                    f"{len(order)}")
            if expect is not None and claimed != expect:
                raise StoreError(
                    f"weights: {path} claims {claimed.hex()[:16]}… and "
                    f"{expect.hex()[:16]}… was wanted")

            body = f.read(expected_bytes + 1)
    except OSError as exc:
        raise StoreError(f"weights: {path} cannot be read: {exc}") from exc

    if len(body) != expected_bytes:
        raise StoreError(
            f"weights: {path} holds {len(body)} bytes of tensors, the layout "
            f"needs {expected_bytes}")

    tensors, at = {}, 0
    for descriptor in order:
        end = at + descriptor.numel * 2
        tensors[descriptor.name] = np.frombuffer(
            body[at:end], dtype=">u2").astype(np.uint16)
        at = end

    actual = hash_of_values(spec, tensors)
    if actual != claimed:
        raise StoreError(
            f"weights: {path} hashes to {actual.hex()[:16]}… and its name says "
            f"{claimed.hex()[:16]}…. The file is damaged.")
    return tensors


def find(directory: str, root: bytes, spec: ModelSpec) -> dict | None:
    """The weights for `root` if this node has them, else None."""
    path = path_for(directory, root)
    if not os.path.exists(path):
        return None
    try:
        return read(path, spec, expect=root)
    except StoreError:
        # A cache entry that does not verify is one to throw away, not one to
        # stop for. Leaving it would mean failing the same way on every start.
        try:
            os.remove(path)
        except OSError:
            pass
        return None


def has(directory: str, root: bytes) -> bool:
    return os.path.exists(path_for(directory, root))


def prune(directory: str, keep: list, retain: int = DEFAULT_RETAIN) -> int:
    """Delete every stored set except the `retain` most recent of `keep`.

    `keep` is roots in chain order, newest last, so the caller decides what
    recent means rather than this guessing from timestamps — a file's mtime says
    when it was written, not where it sits on the chain.
    """
    wanted = {root.hex() for root in keep[-retain:] if root}
    removed = 0
    try:
        names = os.listdir(directory)
    except OSError:
        return 0
    for name in names:
        if not _NAME.match(name):
            continue
        if name[:-len(SUFFIX)] in wanted:
            continue
        try:
            os.remove(os.path.join(directory, name))
            removed += 1
        except OSError:
            pass
    return removed


def tensor_proof(spec: ModelSpec, tensors: dict, name: str) -> merkle.Proof:
    """An inclusion proof for one tensor against the weights root.

    What lets a peer send 759 MiB one piece at a time and be caught on the first
    wrong one, rather than after all of it.
    """
    order = layout(spec)
    leaves = [merkle.leaf_hash(tensor_bytes(tensors, t)) for t in order]
    for index, descriptor in enumerate(order):
        if descriptor.name == name:
            return merkle.build_proof(leaves, index)
    raise StoreError(f"weights: this model has no tensor named {name}")
