"""Network addresses, one representation, IPv4 and IPv6 alike.

EVERY ADDRESS IS SIXTEEN BYTES. IPv4 is carried as an IPv4-mapped IPv6 address
— ::ffff:a.b.c.d — so there is one type on the wire, one type in the address
manager, and no branch anywhere that could handle the two families differently.
Dual-stack is not a feature added on top; it is the absence of a distinction
that was never made.

The cost of that choice, stated so nobody rediscovers it: an IPv4 address has
two spellings, 1.2.3.4 and ::ffff:1.2.3.4, and if they were allowed to differ
the address manager would hold both and count one peer as two. `parse` folds
them, and the folding is tested rather than assumed.

GROUPS ARE HOW DIVERSITY IS MEASURED. An attacker who wants to fill a node's
peer table cheaply buys many addresses in one netblock, so the address manager
buckets by group rather than by address: /16 for IPv4, /32 for IPv6. Tunnelled
addresses are mapped back to what they really are before grouping — a 6to4 or
Teredo address whose group was taken from the IPv6 prefix would give an attacker
a free extra identity per tunnel.
"""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from enum import IntFlag

from ..canon.stream import CanonError, Reader, Writer

# The 96-bit prefix that makes an IPv6 address an IPv4 one.
V4_MAPPED_PREFIX = bytes(10) + b"\xff\xff"


class Services(IntFlag):
    """What a peer says it can do. Advertised, never trusted — a peer that
    claims to serve the corpus and then does not is simply a peer that gets
    asked once."""

    NONE = 0
    CHAIN = 1 << 0      # serves headers and checkpoints
    CORPUS = 1 << 1     # serves corpus chunks against the pinned root
    WEIGHTS = 1 << 2    # serves checkpoint weights
    VERIFY = 1 << 3     # answers challenges


class AddressError(Exception):
    pass


@dataclass(frozen=True)
class NetAddress:
    """One host and port, family-agnostic."""

    ip: bytes           # always 16 bytes
    port: int

    def __post_init__(self):
        if len(self.ip) != 16:
            raise AddressError(f"address: {len(self.ip)} bytes, an address is 16")
        if not 0 <= self.port <= 0xFFFF:
            raise AddressError(f"address: port {self.port} is out of range")

    # -- construction -------------------------------------------------------

    @classmethod
    def parse(cls, text: str, default_port: int = 0) -> "NetAddress":
        """From "1.2.3.4:9444", "[2001:db8::1]:9444", or either without a port.

        The bracket form is required for IPv6 with a port, because a bare
        2001:db8::1:9444 is ambiguous and guessing would resolve it differently
        on different days.
        """
        text = text.strip()
        if not text:
            raise AddressError("address: empty")

        if text.startswith("["):
            end = text.find("]")
            if end < 0:
                raise AddressError(f"address: {text!r} opens a bracket and never closes it")
            host, rest = text[1:end], text[end + 1:]
            port = _port_from(rest, default_port, text)
        elif text.count(":") > 1:
            # Bare IPv6, no port possible without brackets.
            host, port = text, default_port
        else:
            host, _, port_text = text.partition(":")
            port = int(port_text) if port_text else default_port

        try:
            parsed = ipaddress.ip_address(host)
        except ValueError as exc:
            raise AddressError(f"address: {host!r} is not an IP address") from exc
        return cls(ip=_to16(parsed), port=port)

    @classmethod
    def from_sockaddr(cls, sockaddr: tuple, family: int) -> "NetAddress":
        host, port = sockaddr[0], sockaddr[1]
        if family == socket.AF_INET6 and "%" in host:
            host = host.split("%", 1)[0]     # strip the scope id
        return cls(ip=_to16(ipaddress.ip_address(host)), port=port)

    # -- what it is ---------------------------------------------------------

    @property
    def is_v4(self) -> bool:
        return self.ip.startswith(V4_MAPPED_PREFIX)

    @property
    def address(self) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
        if self.is_v4:
            return ipaddress.IPv4Address(self.ip[12:])
        return ipaddress.IPv6Address(self.ip)

    @property
    def host(self) -> str:
        return str(self.address)

    def __str__(self) -> str:
        return f"{self.host}:{self.port}" if self.is_v4 else f"[{self.host}]:{self.port}"

    @property
    def family(self) -> int:
        return socket.AF_INET if self.is_v4 else socket.AF_INET6

    @property
    def is_routable(self) -> bool:
        """Whether it is worth telling anyone about.

        Loopback, private ranges, link-local and the unspecified address are all
        real addresses that mean nothing to a stranger: gossiping them wastes
        everyone's table space and leaks the shape of the local network.
        """
        addr = self.address
        if addr.is_loopback or addr.is_link_local or addr.is_unspecified:
            return False
        if addr.is_multicast or addr.is_reserved:
            return False
        if addr.is_private:
            # is_private covers RFC1918 and the IPv6 unique-local range, and it
            # also covers a few blocks that are private for other reasons.
            return False
        return True

    @property
    def group(self) -> bytes:
        """The netblock this address is bucketed under.

        /16 for IPv4 and /32 for IPv6 — coarse enough that buying a second
        address in the same block gains an attacker nothing, fine enough that
        honest peers in different networks stay distinct.
        """
        if self.is_v4:
            return b"\x04" + self.ip[12:14]

        # A tunnelled address is grouped by the IPv4 it actually rides on.
        # Grouping it by the IPv6 prefix would hand out one free identity per
        # tunnel, which is the cheapest thing an attacker can buy.
        if self.ip[:2] == b"\x20\x02":          # 6to4: 2002:V4::/16
            return b"\x04" + self.ip[2:4]
        if self.ip[:4] == b"\x20\x01\x00\x00":  # Teredo: 2001::/32, v4 is inverted
            v4 = bytes(b ^ 0xFF for b in self.ip[12:16])
            return b"\x04" + v4[:2]
        return b"\x06" + self.ip[:4]

    # -- wire ---------------------------------------------------------------

    def serialize(self, w: Writer) -> Writer:
        return w.raw(self.ip).u16(self.port)

    @classmethod
    def parse_wire(cls, r: Reader) -> "NetAddress":
        ip = r._take(16)
        port = r.u16()
        try:
            return cls(ip=ip, port=port)
        except AddressError as exc:
            raise CanonError(str(exc)) from exc


@dataclass(frozen=True)
class TimestampedAddress:
    """An address with when it was last known good, and what it offers.

    The timestamp is the peer's claim, not ours, and it is clamped on receipt:
    an address stamped in the future would sort to the front of every table
    forever, which is a free way to monopolise a node's outbound slots.
    """

    address: NetAddress
    services: Services
    last_seen_ms: int

    def serialize(self, w: Writer) -> Writer:
        self.address.serialize(w)
        return w.u64(self.services).u64(self.last_seen_ms)

    @classmethod
    def parse_wire(cls, r: Reader) -> "TimestampedAddress":
        address = NetAddress.parse_wire(r)
        return cls(address=address, services=Services(r.u64()), last_seen_ms=r.u64())


def _to16(addr) -> bytes:
    if isinstance(addr, ipaddress.IPv4Address):
        return V4_MAPPED_PREFIX + addr.packed
    packed = addr.packed
    # An IPv6 literal that is really a mapped IPv4 folds to the same bytes, so
    # 1.2.3.4 and ::ffff:1.2.3.4 cannot become two entries for one peer.
    return packed


def _port_from(rest: str, default_port: int, whole: str) -> int:
    if not rest:
        return default_port
    if not rest.startswith(":"):
        raise AddressError(f"address: {whole!r} has trailing junk after the bracket")
    return int(rest[1:])
