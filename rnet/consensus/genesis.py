"""The anchors — the hashes a node trusts a priori, and nothing else.

This is the whole trust bootstrap. A node is compiled knowing two 32-byte
values per network and believes an artifact only if it hashes to one of them.
Everything else it will ever accept — a model shape, a corpus root, a challenge
rate — arrives inside an artifact that had to clear that check first.

WHY THE ANCHORS ARE HERE AND THE VALUES ARE IN params.py. Two descriptions of
the same thing, kept apart on purpose. params.py says what the network is;
this file says what its bytes must hash to. `verify_build` checks that the two
agree, so a value edited in the table without regenerating the anchor is caught
at import rather than at the first handshake with a stranger. Proved by
GenesisTests.TheTablesAgreeWithTheAnchors.

An anchor is not a version number. Changing one does not upgrade a network, it
names a different one — a node with different anchors will refuse every peer it
meets, which is the intended and only safe behaviour.
"""

from __future__ import annotations

import hashlib
import os

from ..canon.container import ObjectType, load
from ..canon.stream import CanonError
from .objects import PolicyDescriptor, RoundDescriptor
from .params import NETWORKS

# Protocol 2.0, generated from the tables in params.py. Regenerate with
# `python -m rnet genesis-anchors` after any change there, and understand that
# doing so forks the network.
GENESIS_HASH: dict[str, str] = {
    "main":    "adfd6082694c614cf44b2490df78ba15efc51ddc1174dd5740506c80ff4f9597",
    "moe":     "c8006631fd3952446636c88b6616b0455e388ce9bbb2504f4ed6756bf9efc7fe",
    "test":    "ce2413197ac4fe09034567d762c9f3a02726f488b5775a3ae1c629170f0e9ba5",
    "regtest": "153cc31ec891b6dc0ff66107bc0d7c70cc9dcbb3257b25294f8ad54e2aa70794",
}

POLICY_HASH: dict[str, str] = {
    "main":    "d0de7064ef6f9ae67133958c9d8a93df06b42c72f19789d557adee4243e39431",
    "moe":     "6447ae3f2daf9e203e8a4041fed10c66670c57cb5df704c52051d947baf90978",
    "test":    "73afd8acfd40ae5fd1da608dc19c7ae5c65aec26da275910fc10d3d9a9f44d79",
    "regtest": "9f64c73609b15760c2583a5df7e852608be67a78c00ceb4ded90cc8aff93cf50",
}


class GenesisError(Exception):
    """A refusal to trust something. Never recoverable by trying again."""


def networks() -> list[str]:
    return sorted(NETWORKS)


def verify_build(network: str | None = None) -> None:
    """Check that the tables still produce the anchors.

    Cheap — four SHA3s over a few hundred bytes — and it turns "somebody
    changed a constant" from a mystery at the first handshake into a message
    naming the network and both hashes.
    """
    for name in ([network] if network else networks()):
        if name not in NETWORKS:
            raise GenesisError(
                f"genesis: unknown network {name!r}; this build knows "
                f"{', '.join(networks())}")
        if name not in GENESIS_HASH or name not in POLICY_HASH:
            # A network in the table with no anchor is a build that can emit an
            # artifact it would then refuse to load. Caught here so it cannot
            # ship.
            raise GenesisError(f"genesis: network {name!r} is in the table with no anchor")
        round_desc, policy = NETWORKS[name]
        for kind, obj, expected in (("genesis", round_desc, GENESIS_HASH[name]),
                                    ("policy", policy, POLICY_HASH[name])):
            actual = obj.id.hex()
            if actual != expected:
                raise GenesisError(
                    f"genesis: {name} {kind} table hashes to {actual}, the anchor is "
                    f"{expected}. A consensus value was changed without regenerating "
                    f"the anchor — which forks the network, so do it deliberately.")


def round_descriptor(network: str) -> RoundDescriptor:
    """The round this build believes `network` is training."""
    verify_build(network)
    return NETWORKS[network][0]


def policy_descriptor(network: str) -> PolicyDescriptor:
    verify_build(network)
    return NETWORKS[network][1]


def emit(network: str, outdir: str) -> tuple[str, str]:
    """Write the two artifacts for `network` and return their paths.

    Written from the tables and verified against the anchors first, so an
    emitted artifact is never one this build would then refuse to load.
    """
    verify_build(network)
    round_desc, policy = NETWORKS[network]
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for suffix, obj in ((".rnet", round_desc), (".rnpol", policy)):
        path = os.path.join(outdir, network + suffix)
        raw = obj.to_container()
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        paths.append(path)
    return paths[0], paths[1]


def load_round(path: str, network: str) -> RoundDescriptor:
    """Read a round artifact from disk, believing it only if it anchors."""
    return _load(path, network, ObjectType.ROUND_DESCRIPTOR, RoundDescriptor,
                 GENESIS_HASH, "genesis")


def load_policy(path: str, network: str) -> PolicyDescriptor:
    policy = _load(path, network, ObjectType.POLICY_DESCRIPTOR, PolicyDescriptor,
                   POLICY_HASH, "policy")
    round_desc = round_descriptor(network)
    if policy.network_magic != round_desc.network_magic:
        # Both anchored and still mismatched means the anchors themselves were
        # generated from different networks — a build error, not a bad file.
        raise GenesisError(
            f"genesis: {path} is for network magic {policy.network_magic:#010x}, "
            f"the round is {round_desc.network_magic:#010x}")
    return policy


def _load(path, network, obj_type, cls, anchors, kind):
    if network not in anchors:
        raise GenesisError(f"genesis: unknown network {network!r}")
    with open(path, "rb") as f:
        raw = f.read()
    actual = hashlib.sha3_256(raw).hexdigest()
    if actual != anchors[network]:
        # Deliberately before parsing. An artifact that does not anchor is not
        # a malformed object to diagnose, it is somebody else's network.
        raise GenesisError(
            f"genesis: {path} hashes to {actual}, {network} expects {anchors[network]}")
    try:
        container = load(path, obj_type)
    except CanonError as exc:
        raise GenesisError(f"genesis: {path} anchors but does not parse: {exc}") from exc
    from ..canon.stream import Reader
    r = Reader(container.content)
    obj = cls.parse(r)
    r.expect_exhausted()
    return obj


def describe(network: str) -> str:
    round_desc = round_descriptor(network)
    policy = policy_descriptor(network)
    m = round_desc.model
    corpus = ("no corpus pinned" if round_desc.dataset_root == bytes(32)
              else f"{round_desc.dataset_root.hex()[:16]}… "
                   f"({round_desc.dataset_chunks:,} chunks)")
    return (
        f"network        {network}  (magic {round_desc.network_magic:#010x})\n"
        f"genesis        {GENESIS_HASH[network]}\n"
        f"policy         {POLICY_HASH[network]}\n"
        f"model          {m.describe()}\n"
        f"arithmetic     {round_desc.numerics.describe()}\n"
        f"corpus         {corpus}\n"
        f"tokenizer      {'byte-level' if round_desc.byte_level_tokenizer else round_desc.tokenizer_hash.hex()[:16] + '…'}\n"
        f"schedule       {policy.inner_steps} inner steps, "
        f"{policy.min_contributors} contributor(s) minimum, "
        f"{policy.round_deadline_ms / 1000:.0f}s deadline\n"
        f"verification   {policy.challenge_percent}% challenged, quorum "
        f"{policy.slash_quorum}, {'shadow mode' if policy.shadow_mode else 'ENFORCING'}"
    )
