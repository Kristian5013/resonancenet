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
    "main": "888a43d234425ca6efb9b802b9faf369beddfea55f3de82513471811b75f75c4",
    "test": "b05405fba484144102881b44497a3311283c5b95847a4d3520d2dbd4d348b84f",
    "regtest": "48babab7e69b3433a96563af65913423571af5b7be9caf78910c667215d22e51",
}

POLICY_HASH = {
    "main": "58d7f7b520ea639c891ed6552c40d8a5809931c9b4a0642d666f67157229dc5c",
    "test": "9a85901e93c71de37c6120fea8aac54e71c345407e574025a4c7367b353db56b",
    "regtest": "d6783edd2b5f1996a677995219f888335bf8f91d40cd3485a87f1fb5822db8d2",
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
