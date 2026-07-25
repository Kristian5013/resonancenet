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
    "main": "305780b8c751e8c9a540344de2eda40df7c2b84fb2cd7ed385ae9b6aaddcb78a",
    "test": "199fa0bd946726a0a416da7b5103891fc6302b337ba84f558740e0bd8338c61d",
    "regtest": "af61f8fdebe3b92867fdf54010c040363adadb4d32f4b76740b502c789f71928",
}

POLICY_HASH = {
    "main": "d9a253a89a6381cd8cfdfcb219b253b36d55e47d808c91452e2b03e9d30d88d7",
    "test": "ed4cfcfa75c2e2a29464dbc00bcc4882aac706d00be60e6ddd2278523b3a0992",
    "regtest": "a2d482f1f4d85fe786c259a4e38074691e830e2902e3f1ccd878ec35c0fa067f",
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
