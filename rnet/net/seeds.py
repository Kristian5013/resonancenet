"""Where a node with nobody to ask starts asking.

A node that has never run has an empty address table and no way to reach the
network. Somebody has to break that circle, and the answer everywhere is the
same: a short list, compiled in, of names and addresses the node will talk to
before it has talked to anyone.

THIS IS TRUSTED INPUT AND SHOULD BE READ THAT WAY. A seed cannot fork the
network — everything it says about the chain is checked against the anchors, and
a lying seed's checkpoints are refused by the same code that refuses any peer's.
What a seed CAN do is decide who a new node meets. Answer with addresses you
control and the node is eclipsed: it sees the network you show it and no other.
That is why the addresses a seed gives are put into the table as ORDINARY
HEARSAY, bucketed under the seed as their source, rather than being connected to
directly. The eclipse resistance that exists for gossip is exactly the defence
needed here, and routing around it for the bootstrap case would have removed it
from the one moment it matters most.

TWO KINDS OF ENTRY, and the second is not redundant. DNS is the maintainable
one: the operator changes a record and every future node finds the new address.
It is also the one that fails first — a resolver that is broken, filtered, or
lying leaves a new node with nothing. So each network also carries literal
addresses, which need a new release to change and work when nothing else does.

ONLY WHEN THE TABLE IS EMPTY. A node that has peers must never touch the seeds:
doing so on every start would put the whole network's reachability on whoever
runs them, which is the failure the persisted address table exists to avoid.

`regtest` HAS NONE, deliberately. It is the network for running two daemons on
one machine, and a local test that reaches out to the internet is a local test
that behaves differently depending on the room it is in.
"""

from __future__ import annotations

import asyncio
import socket

from .address import NetAddress

# How long to wait for all resolution together. A node that cannot start
# because a nameserver is slow is worse than a node that starts knowing nobody
# and tries the literals.
RESOLVE_TIMEOUT_S = 10.0

# Enough to reach the network without letting one seed fill the table. The
# bucketing already limits how much of NEW one source may occupy; this bounds
# the work before that.
MAX_PER_SEED = 32

# DNS names, per network. Changing a record here reaches every future node
# without a release.
DNS_SEEDS: dict[str, tuple[str, ...]] = {
    "main": ("seed.resonancenet.org",),
    "test": ("testseed.resonancenet.org",),
    "moe": ("seed.resonancenet.org",),
    "regtest": (),
}

# Literal addresses, for when DNS is not available or not honest. These need a
# release to change, which is the trade: they work when nothing else does.
FALLBACK_SEEDS: dict[str, tuple[str, ...]] = {
    "main": ("1.240.102.222:9444",),
    "test": (),
    "moe": (),
    "regtest": (),
}


async def resolve(network: str, port: int, *,
                  timeout: float = RESOLVE_TIMEOUT_S,
                  loop=None) -> list[NetAddress]:
    """Every address the seeds for `network` offer, deduplicated.

    Both families: a node listening on IPv6 and given only A records has been
    told about half the network. `getaddrinfo` through the loop's executor so a
    nameserver that never answers delays the bootstrap rather than blocking the
    node.

    Never raises for a seed that fails. A resolver being down is the ordinary
    case this is meant to survive, and one name answering is enough.
    """
    loop = loop or asyncio.get_running_loop()
    names = DNS_SEEDS.get(network, ())
    out: list[NetAddress] = []
    seen: set = set()

    if names:
        async def one(name: str) -> list:
            try:
                return await loop.getaddrinfo(name, port, type=socket.SOCK_STREAM)
            except (OSError, socket.gaierror):
                return []

        try:
            results = await asyncio.wait_for(
                asyncio.gather(*(one(n) for n in names)), timeout)
        except (asyncio.TimeoutError, OSError):
            results = []
        for infos in results:
            kept = 0
            for family, _type, _proto, _canon, sockaddr in infos:
                if kept >= MAX_PER_SEED:
                    break
                if family not in (socket.AF_INET, socket.AF_INET6):
                    continue
                try:
                    address = NetAddress.parse(
                        f"[{sockaddr[0]}]:{sockaddr[1]}" if family == socket.AF_INET6
                        else f"{sockaddr[0]}:{sockaddr[1]}", default_port=port)
                except Exception:                             # noqa: BLE001
                    continue
                if address in seen:
                    continue
                seen.add(address)
                out.append(address)
                kept += 1

    for literal in FALLBACK_SEEDS.get(network, ()):
        try:
            address = NetAddress.parse(literal, default_port=port)
        except Exception:                                     # noqa: BLE001
            continue
        if address in seen:
            continue
        seen.add(address)
        out.append(address)

    return out


def has_seeds(network: str) -> bool:
    return bool(DNS_SEEDS.get(network) or FALLBACK_SEEDS.get(network))
