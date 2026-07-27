"""Genesis loading — the worker's trust bootstrap.

The anchors below are the ONLY values this client trusts a priori, and they must
match src/consensus/params.cpp exactly. Everything else — every model dimension,
the tokenizer, the corpus root — is read from an artifact that hashes to one of
them. An artifact that does not is refused, so a peer (or a tampered local file)
cannot make this worker train a different architecture than the network agreed on.
"""

from __future__ import annotations

import os

from . import canon

# --- trust anchors: keep identical to src/consensus/params.cpp ---
#
# Two artifacts per network, because they change for different reasons: the round
# is frozen until a hard fork, the policy may be retuned from real network data.
GENESIS_HASH = {
    "main": "4de7a5892d63aa450ce871f136b2a3d2edcda0012a33a3bd2dd82a914ff82576",
    "test": "761bd498548c4d0b5eb7654702f334603ae2c09afa9546fbac4f844594eb90f4",
    "regtest": "fddde8262c8d23c30e4c0352bc77c0c58ffdc1fed3a20134c9fda395218d4438",
}

POLICY_HASH = {
    "main": "8a3426b6eee6c16c4c9d13aa12dae5bfc66ada1353aa0625243faebcc26a4e74",
    "test": "42b2726ab6163d8d1dc29b1ee532c36c13be4f3553e21482da7ff4bbed1e5546",
    "regtest": "3ef13240e05e7346ad9c173b050a16711239d9a58c21a0b2cb777293994d34ac",
}


class GenesisError(Exception):
    pass


def load_genesis(raw: bytes, expected_hash: str) -> canon.RoundDescriptor:
    if not expected_hash:
        raise GenesisError("genesis: no trust anchor given — refusing to load")
    actual = canon.container_id(raw)
    if actual != expected_hash:
        raise GenesisError(
            "genesis hash mismatch — refusing untrusted genesis\n"
            f"  file:   {actual}\n  anchor: {expected_hash}"
        )
    container = canon.parse_container(raw)
    if container.obj_type != canon.OBJ_ROUND_DESCRIPTOR:
        raise GenesisError(f"genesis: container holds object type {container.obj_type}")
    round_descriptor = canon.parse_round_descriptor(container.content)
    round_descriptor.genesis_hash = actual
    return round_descriptor


def load_policy(raw: bytes, expected_hash: str) -> canon.PolicyDescriptor:
    if not expected_hash:
        raise GenesisError("policy: no trust anchor given — refusing to load")
    actual = canon.container_id(raw)
    if actual != expected_hash:
        raise GenesisError(
            "policy hash mismatch — refusing untrusted policy\n"
            f"  file:   {actual}\n  anchor: {expected_hash}"
        )
    container = canon.parse_container(raw)
    if container.obj_type != canon.OBJ_POLICY_DESCRIPTOR:
        raise GenesisError(f"policy: container holds object type {container.obj_type}")
    policy = canon.parse_policy_descriptor(container.content)
    policy.policy_hash = actual
    return policy


def load_policy_for_network(network: str, genesis_dir: str) -> canon.PolicyDescriptor:
    if network not in POLICY_HASH:
        raise GenesisError(f"unknown network {network!r}")
    path = os.path.join(genesis_dir, f"{network}.rnpol")
    with open(path, "rb") as f:
        raw = f.read()
    policy = load_policy(raw, POLICY_HASH[network])
    round_descriptor = load_network(network, genesis_dir)
    # The two artifacts must belong to the same network, or a node could be run
    # with one network's model under another's rules.
    if policy.network_magic != round_descriptor.network_magic:
        raise GenesisError("policy and round belong to different networks")
    return policy


def load_network(network: str, genesis_dir: str) -> canon.RoundDescriptor:
    if network not in GENESIS_HASH:
        raise GenesisError(f"unknown network {network!r} (expected {sorted(GENESIS_HASH)})")
    path = os.path.join(genesis_dir, f"{network}.rnet")
    with open(path, "rb") as f:
        raw = f.read()
    return load_genesis(raw, GENESIS_HASH[network])
