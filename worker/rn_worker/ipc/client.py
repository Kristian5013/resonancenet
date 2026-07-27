"""The worker's end of the local channel.

The daemon is not trusted here either, and that symmetry is deliberate: the socket
lives in a directory only this user can enter, but a process running as the same
user could have created it first. So the worker checks the daemon's anchors
against its own compiled-in values exactly as the daemon checks the worker's, and
refuses to train for a daemon running different rules.

The payload crosses as a SEALED MEMFD rather than through the socket. Two reasons,
and the second is the one that matters:

  * Copying 983,635,968 bytes through a socket costs a second and doubles the peak
    memory on both sides.

  * A sealed memfd cannot be modified after it is handed over. A plain file, or a
    path passed by name, would leave the worker able to change the contents
    between the moment the daemon hashes them and the moment it serves them to
    peers — so the daemon would distribute bytes it never verified. Sealing is
    what makes "the bytes I hashed are the bytes I stored" a fact rather than an
    assumption.
"""

from __future__ import annotations

import array
import fcntl
import hashlib
import os
import socket
import struct

from . import cbor

FRAME_MAGIC = b"RNIP"
IPC_VERSION = 1
FRAME_HEADER_SIZE = 20
MAX_FRAME_PAYLOAD = 128 * 1024
MAX_FRAME_SIZE = FRAME_HEADER_SIZE + MAX_FRAME_PAYLOAD

HELLO = 0x0001
HELLO_OK = 0x0002
ERROR = 0x0003
GET_ASSIGNMENT = 0x0010
ASSIGN_TRAIN = 0x0011
ASSIGN_VERIFY = 0x0012
ASSIGN_NONE = 0x0013
ASSIGN_APPLY = 0x0014
GET_WEIGHTS = 0x0020
WEIGHTS = 0x0021
SUBMIT_CONTRIBUTION = 0x0030
SUBMIT_ACCEPTED = 0x0031
SUBMIT_VERDICT = 0x0040
VERDICT_ACCEPTED = 0x0041
SUBMIT_WEIGHTS = 0x0044
WEIGHTS_ACCEPTED = 0x0045
GET_CORPUS_CHUNK = 0x0060
CORPUS_CHUNK = 0x0061
GET_STATUS = 0x0050
STATUS = 0x0051

# fcntl sealing constants. Absent from Python's fcntl module on some versions, so
# they are written out rather than imported conditionally.
F_ADD_SEALS = 1033
F_GET_SEALS = 1034
F_SEAL_SEAL = 0x0001
F_SEAL_SHRINK = 0x0002
F_SEAL_GROW = 0x0004
F_SEAL_WRITE = 0x0008


class IpcError(RuntimeError):
    """The channel failed, or the daemon refused. Never recoverable in place."""


class AnchorMismatch(IpcError):
    """The daemon runs different consensus rules than this worker was built for.

    Fatal by design: training for it would produce work nobody else can use, and
    the worker would look healthy the entire time.
    """


def sealed_memfd(payload: bytes, name: str = "rnet-contribution") -> int:
    """A memfd holding `payload`, sealed so nothing can change it afterwards.

    The seals are applied before the descriptor is handed over, so there is no
    window in which the daemon could be given a writable view.
    """
    fd = os.memfd_create(name, os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    try:
        written = 0
        view = memoryview(payload)
        while written < len(view):
            written += os.write(fd, view[written:])
        # F_SEAL_SEAL last would be redundant here; what matters is that write,
        # shrink and grow are all closed. Any one left open lets the bytes the
        # daemon hashed differ from the bytes it stores.
        fcntl.fcntl(fd, F_ADD_SEALS, F_SEAL_WRITE | F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_SEAL)
        return fd
    except BaseException:
        os.close(fd)
        raise


class DaemonClient:
    """One connection to a local rnetd."""

    def __init__(self, socket_path: str, genesis_hash: bytes, policy_hash: bytes,
                 user_agent: str = "rn_worker/0.1.0") -> None:
        if len(genesis_hash) != 32 or len(policy_hash) != 32:
            raise IpcError("ipc: anchors must be 32 bytes each")
        self.socket_path = socket_path
        self._genesis_hash = genesis_hash
        self._policy_hash = policy_hash
        self._user_agent = user_agent
        self._sock: socket.socket | None = None
        self._next_request_id = 1

        # Filled by the handshake, from the containers the daemon returns.
        self.worker_id: int | None = None
        self.n_params: int | None = None
        self.round_container: bytes | None = None
        self.policy_container: bytes | None = None
        self.genesis_weights_hash: bytes | None = None

    # -- connection ---------------------------------------------------------

    def connect(self) -> None:
        # SEQPACKET, matching the daemon: records are preserved, so a message
        # cannot be split across reads and there is no reassembly to get wrong.
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        try:
            sock.connect(self.socket_path)
        except OSError as exc:
            sock.close()
            raise IpcError(f"ipc: cannot connect to {self.socket_path}: {exc}") from exc
        self._sock = sock

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def __enter__(self) -> "DaemonClient":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- framing ------------------------------------------------------------

    def _send(self, msg_type: int, payload: dict, fd: int | None = None) -> int:
        if self._sock is None:
            raise IpcError("ipc: not connected")
        body = cbor.dumps(payload)
        if len(body) > MAX_FRAME_PAYLOAD:
            # Truncating would produce a shorter message that still parses — the
            # worst outcome, because it looks like success.
            raise IpcError(f"ipc: payload of {len(body)} bytes exceeds the frame limit")

        request_id = self._next_request_id
        self._next_request_id += 1
        header = FRAME_MAGIC + struct.pack(">HHQI", IPC_VERSION, msg_type, request_id, len(body))
        record = header + body

        if fd is None:
            sent = self._sock.send(record)
        else:
            sent = self._sock.sendmsg(
                [record],
                [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [fd]))],
            )
        if sent != len(record):
            raise IpcError("ipc: the transport truncated a record")
        return request_id

    def _receive(self, expect_request_id: int, expect_fd: bool = False) -> tuple[int, dict, int | None]:
        if self._sock is None:
            raise IpcError("ipc: not connected")
        received_fd: int | None = None
        if expect_fd:
            record, ancdata, _flags, _addr = self._sock.recvmsg(
                MAX_FRAME_SIZE, socket.CMSG_SPACE(struct.calcsize("i"))
            )
            for level, ctype, data in ancdata:
                if level == socket.SOL_SOCKET and ctype == socket.SCM_RIGHTS:
                    fds = array.array("i")
                    fds.frombytes(data[: len(data) - (len(data) % fds.itemsize)])
                    for extra in fds[1:]:
                        os.close(extra)   # never leak what we will not use
                    if len(fds) >= 1:
                        received_fd = fds[0]
        else:
            record = self._sock.recv(MAX_FRAME_SIZE)
        if not record:
            if received_fd is not None:
                os.close(received_fd)
            raise IpcError("ipc: the daemon closed the connection")
        if len(record) < FRAME_HEADER_SIZE:
            raise IpcError(f"ipc: record of {len(record)} bytes is shorter than a header")
        if record[:4] != FRAME_MAGIC:
            raise IpcError("ipc: bad frame magic")

        version, msg_type, request_id, declared = struct.unpack(">HHQI", record[4:20])
        if version != IPC_VERSION:
            raise IpcError(f"ipc: unsupported frame version {version}")
        # The kernel preserved a record boundary; this checks the sender's own
        # description agrees with it.
        if FRAME_HEADER_SIZE + declared != len(record):
            raise IpcError(
                f"ipc: frame declares {declared} payload bytes but the record holds "
                f"{len(record) - FRAME_HEADER_SIZE}"
            )
        if request_id != expect_request_id:
            # A reply to something else, or a replay. Either way it is not an
            # answer to the question that was asked.
            raise IpcError(f"ipc: reply carries request id {request_id}, expected {expect_request_id}")

        body = cbor.require_map(cbor.loads(record[FRAME_HEADER_SIZE:]))
        if msg_type == ERROR:
            if received_fd is not None:
                os.close(received_fd)
            code = cbor.get_uint(body, "code")
            reason = cbor.get_text(body, "reason")
            if code == 3:
                raise AnchorMismatch(f"ipc: the daemon runs different rules: {reason}")
            raise IpcError(f"ipc: refused (code {code}): {reason}")
        return msg_type, body, received_fd

    def _request(self, msg_type: int, payload: dict, fd: int | None = None,
                 expect_fd: bool = False) -> tuple[int, dict, int | None]:
        request_id = self._send(msg_type, payload, fd)
        return self._receive(request_id, expect_fd)

    # -- protocol -----------------------------------------------------------

    def hello(self) -> dict:
        """Completes the handshake and checks the daemon against this worker's anchors.

        Note what is not sent: this worker does not state its own `worker_id`. That
        value belongs to the daemon's configuration — a worker that could choose it
        could submit under someone else's name.
        """
        msg_type, reply, _ = self._request(
            HELLO,
            {
                "genesis_hash": self._genesis_hash,
                "policy_hash": self._policy_hash,
                "user_agent": self._user_agent,
            },
        )
        if msg_type != HELLO_OK:
            raise IpcError(f"ipc: expected hello-ok, got message type 0x{msg_type:04x}")

        round_container = cbor.get_bytes(reply, "round")
        policy_container = cbor.get_bytes(reply, "policy")

        # Compared, never adopted — the same rule the daemon applies to this
        # worker. A squatted socket is a same-uid attack the OS cannot prevent, so
        # the containers are checked against the anchors compiled into THIS binary.
        if hashlib.sha3_256(round_container).digest() != self._genesis_hash:
            raise AnchorMismatch("ipc: the daemon's round descriptor does not match our genesis anchor")
        if hashlib.sha3_256(policy_container).digest() != self._policy_hash:
            raise AnchorMismatch("ipc: the daemon's policy descriptor does not match our policy anchor")

        self.round_container = round_container
        self.policy_container = policy_container
        self.worker_id = cbor.get_uint(reply, "worker_id")
        self.n_params = cbor.get_uint(reply, "n_params")
        self.genesis_weights_hash = cbor.get_hash(reply, "genesis_weights_hash")
        if self.worker_id == 0:
            raise IpcError("ipc: the daemon reported worker id 0, which is the unset value")
        return reply

    def get_assignment(self) -> dict:
        """Asks what to train.

        The reply says which step and which weights to start from — and nothing
        about which data to read. Batch windows are derived by this worker from
        values it verified; if the daemon supplied them, a colluding daemon and
        worker would regain exactly the freedom the derivation removes.
        """
        msg_type, reply, fd = self._request(GET_ASSIGNMENT, {}, expect_fd=True)
        if msg_type == ASSIGN_NONE:
            return {}
        if msg_type == ASSIGN_APPLY:
            # The node aggregated a round and needs the weights updated. It cannot
            # do this itself: the model lives here.
            if fd is None:
                raise IpcError("ipc: an apply assignment arrived without its update")
            return {
                "kind": "apply",
                "assignment_id": cbor.get_uint(reply, "assignment_id"),
                "outer_step": cbor.get_uint(reply, "outer_step"),
                "scale_exp": cbor.get_int(reply, "scale_exp"),
                "update_hash": cbor.get_hash(reply, "update_hash"),
                "base_weights_hash": cbor.get_hash(reply, "base_weights_hash"),
                "contributor_count": cbor.get_uint(reply, "contributor_count"),
                "update_fd": fd,
            }
        if fd is not None:
            os.close(fd)
        if msg_type != ASSIGN_TRAIN:
            raise IpcError(f"ipc: expected an assignment, got message type 0x{msg_type:04x}")
        return {
            "kind": "train",
            "assignment_id": cbor.get_uint(reply, "assignment_id"),
            "round_id": cbor.get_uint(reply, "round_id"),
            "outer_step": cbor.get_uint(reply, "outer_step"),
            "base_checkpoint": cbor.get_hash(reply, "base_checkpoint"),
            "base_weights_hash": cbor.get_hash(reply, "base_weights_hash"),
        }

    def submit_contribution(self, assignment_id: int, payload: bytes, scale_exp: int) -> dict:
        """Hands over the quantised pseudo-gradient.

        Only two values here are the worker's to state: the scale exponent it
        chose, and the hash of the bytes it produced. Everything else in the
        contribution header — round, step, base checkpoint, worker id, parameter
        count, inner steps — is rebuilt by the daemon from its own state, so
        nothing this worker could get wrong becomes consensus.

        There is deliberately no length field: the size is fixed by the model at
        one byte per parameter, and a second source for that number would be a
        second thing to disagree about.
        """
        if self.n_params is not None and len(payload) != self.n_params:
            raise IpcError(
                f"ipc: payload is {len(payload)} bytes, the model has {self.n_params} parameters "
                "and an int8 payload is exactly one byte each"
            )
        payload_hash = hashlib.sha3_256(payload).digest()
        fd = sealed_memfd(payload)
        try:
            msg_type, reply, _ = self._request(
                SUBMIT_CONTRIBUTION,
                {
                    "assignment_id": assignment_id,
                    "payload_hash": payload_hash,
                    "scale_exp": scale_exp,
                },
                fd=fd,
            )
        finally:
            os.close(fd)
        if msg_type != SUBMIT_ACCEPTED:
            raise IpcError(f"ipc: expected submit-accepted, got message type 0x{msg_type:04x}")
        return {
            "contribution_id": cbor.get_hash(reply, "contribution_id"),
            "outer_step": cbor.get_uint(reply, "outer_step"),
        }

    def read_update(self, assignment: dict) -> "np.ndarray":
        """Reads the staged update from the sealed descriptor and checks it.

        The descriptor is sealed by the daemon before it is handed over, so these
        bytes cannot change while they are being read. The size is derived from the
        model rather than taken from the message — four bytes per parameter — so a
        daemon cannot make this worker allocate by claiming a size.
        """
        import mmap

        import numpy as np

        if self.n_params is None:
            raise IpcError("ipc: the handshake has not run, so the model size is unknown")
        fd = assignment["update_fd"]
        expected = self.n_params * 4
        size = os.fstat(fd).st_size
        if size != expected:
            raise IpcError(f"ipc: update is {size} bytes, the model needs {expected}")

        with mmap.mmap(fd, expected, prot=mmap.PROT_READ) as mapping:
            raw = mapping.read(expected)
        # The daemon's claim about its own bytes, checked against the bytes. A
        # squatted socket is a same-uid attack the OS cannot prevent, so this is
        # checked rather than assumed.
        if hashlib.sha3_256(raw).digest() != assignment["update_hash"]:
            raise IpcError("ipc: the update does not hash to the value the assignment claims")
        return np.frombuffer(raw, dtype=">i4").astype(np.int64)

    def fetch_weights(self, weights_hash: bytes) -> bytes:
        """Asks the node for a checkpoint's weights and verifies them.

        This is how a worker joins a network that has moved past genesis. Deriving
        the initial weights is not enough once the chain has advanced: catching up
        needs either every historical payload, which the retention window
        deliberately does not keep, or the weights themselves.

        The node passes an ordinary read-only descriptor rather than a sealed one,
        because in this direction it is the RECEIVER that verifies — the hash comes
        from the checkpoint header, which this worker already holds.
        """
        if len(weights_hash) != 32:
            raise IpcError("ipc: a weights hash is 32 bytes")
        msg_type, reply, fd = self._request(
            GET_WEIGHTS, {"weights_hash": weights_hash}, expect_fd=True
        )
        if msg_type != WEIGHTS:
            if fd is not None:
                os.close(fd)
            raise IpcError(f"ipc: expected weights, got message type 0x{msg_type:04x}")
        if fd is None:
            raise IpcError("ipc: the node answered without the weights")
        try:
            size = cbor.get_uint(reply, "size")
            if self.n_params is not None and size != self.n_params * 2:
                raise IpcError(
                    f"ipc: weights are {size} bytes, the model needs {self.n_params * 2} "
                    "(two bytes per parameter in bf16)"
                )
            raw = b""
            while len(raw) < size:
                chunk = os.pread(fd, min(1 << 24, size - len(raw)), len(raw))
                if not chunk:
                    raise IpcError(f"ipc: weights ended after {len(raw)} of {size} bytes")
                raw += chunk
        finally:
            os.close(fd)

        # Checked against the hash the checkpoint committed to, not against
        # anything the node said. A squatted socket is a same-uid attack the OS
        # cannot prevent, so this is the check that makes the transfer safe.
        if hashlib.sha3_256(raw).digest() != weights_hash:
            raise IpcError("ipc: the weights do not hash to the checkpoint's value")
        return raw

    def submit_weights(self, assignment_id: int, weights_hash: bytes, weights: bytes) -> dict:
        """Reports what the updated weights hash to, and hands over the bytes.

        The hash is a computation, not a judgement: anyone holding the same base
        weights and the same update reaches the same value, which is what makes the
        checkpoint this completes checkable by every other node.

        The bytes travel with it because a checkpoint whose weights nobody holds is
        a checkpoint no new node can start from — publishing one without storing
        them would announce a state the network cannot reach.
        """
        if len(weights_hash) != 32:
            raise IpcError("ipc: a weights hash is 32 bytes")
        if hashlib.sha3_256(weights).digest() != weights_hash:
            raise IpcError("ipc: refusing to submit weights that do not match their own hash")
        fd = sealed_memfd(weights, "rnet-weights")
        try:
            msg_type, reply, _ = self._request(
                SUBMIT_WEIGHTS,
                {"assignment_id": assignment_id, "weights_hash": weights_hash},
                fd=fd,
            )
        finally:
            os.close(fd)
        if msg_type != WEIGHTS_ACCEPTED:
            raise IpcError(f"ipc: expected weights-accepted, got message type 0x{msg_type:04x}")
        return {
            "outer_step": cbor.get_uint(reply, "outer_step"),
            "checkpoint_id": cbor.get_hash(reply, "checkpoint_id"),
        }

    def corpus_chunk(self, chunk_index: int) -> bytes:  # noqa: D401
        """One chunk of the corpus, as raw UTF-8.

        The node answers from its own copy if it serves one, and otherwise fetches
        it from a peer and checks it against dataset_root before handing it over —
        so the bytes are trustworthy wherever they came from, which is the point of
        a seed.

        A fetch in flight is refused with Unavailable rather than waited on: the
        daemon serves every worker from one loop and must not block in it. The
        caller retries; RemoteCorpus does that.
        """
        import mmap

        msg_type, reply, fd = self._request(GET_CORPUS_CHUNK, {"chunk_index": chunk_index},
                                            expect_fd=True)
        if msg_type != CORPUS_CHUNK:
            raise IpcError(f"expected a corpus chunk, got message type {msg_type:#06x}")
        got = reply["chunk_index"]
        if got != chunk_index:
            raise IpcError(f"asked for chunk {chunk_index}, was given {got}")
        if fd is None:
            raise IpcError("ipc: the daemon sent no descriptor for the corpus chunk")

        # A chunk is a megabyte, which does not fit a frame, so it travels over a
        # sealed descriptor like every other bulk payload here. The size comes from
        # the descriptor rather than from the message: a daemon must not be able to
        # make this worker allocate by claiming a size.
        try:
            size = os.fstat(fd).st_size
            if size != reply["bytes"]:
                raise IpcError(f"ipc: chunk is {size} bytes, the message says {reply['bytes']}")
            with mmap.mmap(fd, size, prot=mmap.PROT_READ) as mapping:
                raw = mapping.read(size)
        finally:
            os.close(fd)

        # Checked against the hash the daemon stated. The daemon has already
        # checked the chunk against dataset_root; this catches the socket being
        # squatted by something with the same uid, which the OS cannot prevent.
        if hashlib.sha3_256(raw).digest() != reply["chunk_hash"]:
            raise IpcError("ipc: the corpus chunk does not hash to the value the daemon claims")
        return raw

    def status(self) -> dict:
        msg_type, reply, _ = self._request(GET_STATUS, {})
        if msg_type != STATUS:
            raise IpcError(f"ipc: expected status, got message type 0x{msg_type:04x}")
        return {
            "peers": cbor.get_uint(reply, "peers"),
            "objects": cbor.get_uint(reply, "objects"),
            "outer_step": cbor.get_uint(reply, "outer_step"),
            "contributions": cbor.get_uint(reply, "contributions"),
        }
