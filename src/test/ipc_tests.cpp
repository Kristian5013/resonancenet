// The local channel between a worker process and the node.
//
// These tests are almost entirely about what the decoder REFUSES. A CBOR decoder
// that accepts several spellings of one message is not a convenience — it is a
// place where two implementations can read the same bytes differently, and this
// channel carries consensus work.
//
// Each rejection below corresponds to a specific way RFC 8949 permits ambiguity,
// and the test names say which.
#include <format>

#include <fcntl.h>
#include <sys/mman.h>
#include "ipc/cbor.h"
#include "test/framework.h"

using namespace rnet;
using namespace rnet::ipc;

namespace {

std::vector<uint8_t> Raw(std::initializer_list<int> bytes) {
    std::vector<uint8_t> out;
    for (int b : bytes) out.push_back(static_cast<uint8_t>(b));
    return out;
}

std::string HexOf(const std::vector<uint8_t>& bytes) {
    std::string out;
    for (uint8_t b : bytes) out += std::format("{:02x}", b);
    return out;
}

}  // namespace

RNET_TEST(Cbor, EncodesIntegersInTheirShortestForm) {
    struct Case {
        uint64_t value;
        const char* encoding;
    };
    // The boundaries are where a lazy encoder starts producing a second spelling.
    const Case cases[] = {
        {0, "00"},
        {23, "17"},
        {24, "1818"},
        {255, "18ff"},
        {256, "190100"},
        {65535, "19ffff"},
        {65536, "1a00010000"},
        {4294967295ull, "1affffffff"},
        {4294967296ull, "1b0000000100000000"},
    };
    for (const auto& c : cases) {
        auto encoded = CborWriter().Uint(c.value).Take();
        RNET_CHECK_OK(encoded);
        RNET_CHECK_STREQ(HexOf(encoded.value()), c.encoding);

        auto parsed = CborParse(encoded.value());
        RNET_CHECK_OK(parsed);
        auto back = parsed.value().AsUint();
        RNET_CHECK_OK(back);
        RNET_CHECK_EQ(back.value(), c.value);
    }
}

// 1 must be 0x01, never 0x1801. Accepting both means accepting two byte strings
// that mean the same thing, and any check made on bytes rather than on meaning
// can then be walked around.
RNET_TEST(Cbor, RefusesNonMinimalIntegerEncodings) {
    const std::vector<std::vector<uint8_t>> padded = {
        Raw({0x18, 0x01}),                          // 1 in two bytes
        Raw({0x19, 0x00, 0x01}),                    // 1 in three
        Raw({0x1A, 0x00, 0x00, 0x00, 0x01}),        // 1 in five
        Raw({0x1B, 0, 0, 0, 0, 0, 0, 0, 0x01}),     // 1 in nine
        Raw({0x19, 0x00, 0x17}),                    // 23, which fits in one
        Raw({0x1A, 0x00, 0x00, 0x01, 0x00}),        // 256, which fits in three
    };
    for (const auto& bytes : padded) {
        auto parsed = CborParse(bytes);
        if (parsed) RNET_FAIL("accepted a non-minimal encoding: {}", HexOf(bytes));
    }
}

// Indefinite lengths are a second way to write what a definite length already
// says, and they let a sender stream past any bound checked on the head.
RNET_TEST(Cbor, RefusesIndefiniteLengthItems) {
    const std::vector<std::vector<uint8_t>> indefinite = {
        Raw({0x5F, 0x41, 0x61, 0xFF}),          // indefinite byte string
        Raw({0x7F, 0x61, 0x61, 0xFF}),          // indefinite text string
        Raw({0x9F, 0x01, 0xFF}),                // indefinite array
        Raw({0xBF, 0x61, 0x61, 0x01, 0xFF}),    // indefinite map
    };
    for (const auto& bytes : indefinite) {
        auto parsed = CborParse(bytes);
        if (parsed) RNET_FAIL("accepted an indefinite-length item: {}", HexOf(bytes));
    }
}

// Floats are the worst offender: half, single and double all encode 1.0, NaN
// carries a payload, and -0.0 is a distinct bit pattern that compares equal to
// 0.0. Anything needing a fraction crosses this boundary as a scaled integer.
RNET_TEST(Cbor, RefusesFloatsNullAndUndefined) {
    const std::vector<std::vector<uint8_t>> forbidden = {
        Raw({0xF9, 0x3C, 0x00}),                       // half 1.0
        Raw({0xFA, 0x3F, 0x80, 0x00, 0x00}),           // single 1.0
        Raw({0xFB, 0x3F, 0xF0, 0, 0, 0, 0, 0, 0}),     // double 1.0
        Raw({0xF9, 0x7E, 0x00}),                       // NaN
        Raw({0xF9, 0x80, 0x00}),                       // -0.0
        Raw({0xF6}),                                   // null
        Raw({0xF7}),                                   // undefined
        Raw({0xE0}),                                   // simple(0)
    };
    for (const auto& bytes : forbidden) {
        auto parsed = CborParse(bytes);
        if (parsed) RNET_FAIL("accepted a forbidden simple value: {}", HexOf(bytes));
    }

    // true and false are the only things permitted in major type 7.
    auto t = CborParse(Raw({0xF5}));
    RNET_CHECK_OK(t);
    RNET_CHECK(t.value().AsBool().value());
    auto f = CborParse(Raw({0xF4}));
    RNET_CHECK_OK(f);
    RNET_CHECK(!f.value().AsBool().value());
}

// A tag changes what a value means without changing its bytes — the exact shape
// of a hiding place this protocol removes everywhere else.
RNET_TEST(Cbor, RefusesTags) {
    auto tagged = CborParse(Raw({0xC0, 0x61, 0x61}));   // tag 0 over a text string
    RNET_CHECK_ERR(tagged);
}

// "Last one wins" and "first one wins" are both defensible, which is exactly why
// two implementations will disagree. A duplicate is refused outright.
RNET_TEST(Cbor, RefusesDuplicateMapKeys) {
    // {"a": 1, "a": 2}
    auto duplicated = CborParse(Raw({0xA2, 0x61, 0x61, 0x01, 0x61, 0x61, 0x02}));
    RNET_CHECK_ERR(duplicated);
}

// The order must be a property of the keys, not of whoever wrote them.
RNET_TEST(Cbor, RefusesMapKeysOutOfCanonicalOrder) {
    // {"b": 1, "a": 2} — correct contents, wrong order.
    auto unsorted = CborParse(Raw({0xA2, 0x61, 0x62, 0x01, 0x61, 0x61, 0x02}));
    RNET_CHECK_ERR(unsorted);

    // {"a": 1, "b": 2} is the same map written the one legal way.
    auto sorted = CborParse(Raw({0xA2, 0x61, 0x61, 0x01, 0x61, 0x62, 0x02}));
    RNET_CHECK_OK(sorted);
    RNET_CHECK_EQ(sorted.value().GetUint("a").value(), uint64_t{1});
    RNET_CHECK_EQ(sorted.value().GetUint("b").value(), uint64_t{2});
}

// Ordering is by ENCODED bytes, which puts shorter keys first regardless of how
// they compare as strings. "z" sorts before "aa" — a rule that surprises people
// and therefore needs a test rather than a comment.
RNET_TEST(Cbor, KeyOrderIsByEncodedBytesNotByString) {
    RNET_CHECK(CborKeyLess("z", "aa"));
    RNET_CHECK(!CborKeyLess("aa", "z"));
    RNET_CHECK(CborKeyLess("a", "b"));
    RNET_CHECK(!CborKeyLess("a", "a"));

    // And the writer enforces exactly that order.
    auto shorter_first = CborWriter().BeginMap(2).Key("z").Uint(1).Key("aa").Uint(2).Take();
    RNET_CHECK_OK(shorter_first);
    RNET_CHECK_OK(CborParse(shorter_first.value()));

    auto wrong_way = CborWriter().BeginMap(2).Key("aa").Uint(1).Key("z").Uint(2).Take();
    RNET_CHECK_ERR(wrong_way);
}

// Map keys are text only. Integer keys would be a second way to name the same
// field, and mixed key types make the ordering rule depend on which types happen
// to be present.
RNET_TEST(Cbor, RefusesNonTextMapKeys) {
    auto integer_keys = CborParse(Raw({0xA1, 0x01, 0x02}));   // {1: 2}
    RNET_CHECK_ERR(integer_keys);
}

// A length prefix is a claim by the sender. It is checked against the bytes
// actually present before anything is allocated, so a frame claiming four
// gigabytes costs nothing to refuse.
RNET_TEST(Cbor, RefusesLengthsThatExceedTheInput) {
    auto huge_string = CborParse(Raw({0x5A, 0xFF, 0xFF, 0xFF, 0xFF}));   // 4 GiB byte string
    RNET_CHECK_ERR(huge_string);

    auto huge_array = CborParse(Raw({0x9A, 0xFF, 0xFF, 0xFF, 0xFF}));
    RNET_CHECK_ERR(huge_array);

    auto huge_map = CborParse(Raw({0xBA, 0xFF, 0xFF, 0xFF, 0xFF}));
    RNET_CHECK_ERR(huge_map);

    auto short_string = CborParse(Raw({0x43, 0x01, 0x02}));   // says 3 bytes, gives 2
    RNET_CHECK_ERR(short_string);
}

// Deep nesting is either a bug or an attempt to exhaust the parser's stack.
RNET_TEST(Cbor, RefusesNestingPastTheLimit) {
    std::vector<uint8_t> deep;
    for (uint32_t i = 0; i <= kMaxCborDepth + 2; ++i) deep.push_back(0x81);   // array(1)
    deep.push_back(0x00);
    RNET_CHECK_ERR(CborParse(deep));

    // Something within the limit still works.
    std::vector<uint8_t> shallow;
    for (uint32_t i = 0; i < 3; ++i) shallow.push_back(0x81);
    shallow.push_back(0x00);
    RNET_CHECK_OK(CborParse(shallow));
}

// Trailing bytes are an error for the same reason they are in the canon container
// format: an attacker who can append data that some parsers ignore can make two
// readers see different messages in one frame.
RNET_TEST(Cbor, RefusesTrailingBytes) {
    RNET_CHECK_ERR(CborParse(Raw({0x01, 0x02})));
}

// Text must be valid UTF-8, or one implementation decodes while another compares
// bytes and they disagree about what the field said.
RNET_TEST(Cbor, RefusesInvalidUtf8) {
    const std::vector<std::vector<uint8_t>> invalid = {
        Raw({0x62, 0xC3, 0x28}),           // bad continuation
        Raw({0x62, 0xC0, 0xAF}),           // overlong '/'
        Raw({0x63, 0xED, 0xA0, 0x80}),     // surrogate
        Raw({0x61, 0x80}),                 // lone continuation
    };
    for (const auto& bytes : invalid) {
        auto parsed = CborParse(bytes);
        if (parsed) RNET_FAIL("accepted invalid utf-8: {}", HexOf(bytes));
    }
}

// The writer catches the author's mistakes rather than sending them: a container
// whose declared count does not match what follows would otherwise become the
// reader's problem.
RNET_TEST(Cbor, WriterRefusesMalformedConstruction) {
    // Declared three pairs, wrote one.
    RNET_CHECK_ERR(CborWriter().BeginMap(3).Key("a").Uint(1).Take());

    // A value where a key belongs.
    RNET_CHECK_ERR(CborWriter().BeginMap(1).Uint(1).Uint(2).Take());

    // A key outside a map.
    RNET_CHECK_ERR(CborWriter().Key("a").Take());

    // An array that declared fewer items than were written. The extra value lands
    // at the top level, where a second value is itself an error: a frame holds one
    // document, and a reader that stops at the first would ignore the rest.
    RNET_CHECK_ERR(CborWriter().BeginArray(1).Uint(1).Uint(2).Take());
    RNET_CHECK_ERR(CborWriter().Uint(1).Uint(2).Take());

    // Nothing at all is not a message.
    RNET_CHECK_ERR(CborWriter().Take());
}

RNET_TEST(Cbor, RoundTripsAMessageShapedValue) {
    const std::vector<uint8_t> hash(32, 0xAB);
    // Written in canonical order: shorter encoded keys first, then bytewise. That
    // puts "ok" (2 bytes) ahead of "base" (4) and "scale_exp" (9) last — an order
    // that reads oddly and is exactly why the writer enforces it rather than
    // trusting the author to sort by eye.
    auto encoded = CborWriter()
                       .BeginMap(5)
                       .Key("ok").Bool(true)
                       .Key("base").Bytes(hash)
                       .Key("step").Uint(7)
                       .Key("type").Text("work")
                       .Key("scale_exp").Int(-12)
                       .Take();
    RNET_CHECK_OK(encoded);

    auto parsed = CborParse(encoded.value());
    RNET_CHECK_OK(parsed);
    RNET_CHECK_STREQ(parsed.value().GetText("type").value(), "work");
    RNET_CHECK_EQ(parsed.value().GetUint("step").value(), uint64_t{7});
    RNET_CHECK_EQ(parsed.value().GetInt("scale_exp").value(), int64_t{-12});
    RNET_CHECK(parsed.value().GetBool("ok").value());
    RNET_CHECK_EQ(parsed.value().GetHash("base").value()[0], uint8_t{0xAB});

    // A missing field is an error rather than a zero: a message that forgot to say
    // which step it refers to must not be read as referring to step zero.
    RNET_CHECK_ERR(parsed.value().GetUint("absent"));

    // And a field of the wrong type is not silently converted.
    RNET_CHECK_ERR(parsed.value().GetUint("type"));
    RNET_CHECK_ERR(parsed.value().GetText("step"));
}

// Negative integers round-trip exactly, including the boundary where a decoder
// that wraps would turn a refused value into an accepted one.
RNET_TEST(Cbor, NegativeIntegersRoundTripAndDoNotWrap) {
    for (int64_t value : {int64_t{-1}, int64_t{-24}, int64_t{-25}, int64_t{-256}, int64_t{-65536},
                          int64_t{-2147483648LL}}) {
        auto encoded = CborWriter().Int(value).Take();
        RNET_CHECK_OK(encoded);
        auto parsed = CborParse(encoded.value());
        RNET_CHECK_OK(parsed);
        RNET_CHECK_EQ(parsed.value().AsInt().value(), value);
    }

    // -1 - 2^63 does not fit in int64_t; refused rather than wrapped.
    RNET_CHECK_ERR(CborParse(Raw({0x3B, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF})));
}

// A hash field of the wrong length is not a hash. Padding or truncating one would
// turn a malformed message into a valid-looking identity.
RNET_TEST(Cbor, RefusesHashFieldsOfTheWrongLength) {
    for (size_t length : {size_t{0}, size_t{31}, size_t{33}, size_t{64}}) {
        const std::vector<uint8_t> wrong(length, 0x11);
        auto encoded = CborWriter().BeginMap(1).Key("h").Bytes(wrong).Take();
        RNET_CHECK_OK(encoded);
        auto parsed = CborParse(encoded.value());
        RNET_CHECK_OK(parsed);
        RNET_CHECK_ERR(parsed.value().GetHash("h"));
    }
}

// ---------------------------------------------------------------------------
// The socket itself. These tests use a raw client socket rather than the C++
// helper, because the thing being tested is what arrives on the wire — a helper
// that made the same mistake on both sides would prove nothing.
// ---------------------------------------------------------------------------
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <cstring>
#include <filesystem>

#include "ipc/server.h"

namespace {

// Short by necessity: sun_path holds 107 bytes, and the project's own scratch
// directories are longer than that.
std::filesystem::path TestSocketDir(std::string_view name) {
    return std::filesystem::path("/tmp") / std::format("rnet-ipc-{}", name);
}

// A raw connected client socket, or -1.
int ConnectRaw(const std::filesystem::path& path) {
    const int fd = ::socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
    if (fd < 0) return -1;
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    const std::string text = path.string();
    std::memcpy(address.sun_path, text.c_str(), text.size());
    if (::connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
        ::close(fd);
        return -1;
    }
    return fd;
}

std::vector<uint8_t> HelloRecord(uint64_t request_id) {
    const std::vector<uint8_t> anchor(32, 0x11);
    auto payload = CborWriter()
                       .BeginMap(3)
                       .Key("policy_hash").Bytes(anchor)
                       .Key("genesis_hash").Bytes(anchor)
                       .Key("user_agent").Text("test/1")
                       .Take();
    Frame frame;
    frame.type = MessageType::Hello;
    frame.request_id = request_id;
    frame.payload = payload.ok() ? payload.value() : std::vector<uint8_t>{};
    auto encoded = EncodeFrame(frame);
    return encoded.ok() ? encoded.value() : std::vector<uint8_t>{};
}

}  // namespace

// The directory is what enforces access, and it must be restrictive BEFORE the
// socket exists — a mode applied after bind() leaves a window in which the socket
// is reachable with umask-derived permissions.
RNET_TEST(IpcServer, TheDirectoryIsRestrictedBeforeTheSocketExists) {
    const auto dir = TestSocketDir("perms");
    std::filesystem::remove_all(dir);

    Server server;
    RNET_CHECK_OK(server.Listen(dir / "rnet.sock"));

    struct stat info {};
    RNET_CHECK_EQ(::stat(dir.c_str(), &info), 0);
    // 0700: nobody but this user may even enter the directory.
    RNET_CHECK_EQ(int(info.st_mode & 0777), 0700);

    struct stat sock_info {};
    RNET_CHECK_EQ(::stat((dir / "rnet.sock").c_str(), &sock_info), 0);
    RNET_CHECK_EQ(int(sock_info.st_mode & 0777), 0600);

    server.Stop();
    // The socket is removed on the way out, so the next start need not decide
    // whether a leftover is stale.
    RNET_CHECK(!std::filesystem::exists(dir / "rnet.sock"));
    std::filesystem::remove_all(dir);
}

// A live socket and a stale one look identical on disk. Unlinking blindly would
// let a second instance quietly steal the channel from a running one.
RNET_TEST(IpcServer, RefusesToTakeOverALiveSocket) {
    const auto dir = TestSocketDir("live");
    std::filesystem::remove_all(dir);

    Server first;
    RNET_CHECK_OK(first.Listen(dir / "rnet.sock"));

    Server second;
    auto stolen = second.Listen(dir / "rnet.sock");
    RNET_CHECK_ERR(stolen);
    if (stolen.error().find("already has a daemon") == std::string::npos) {
        RNET_FAIL("the refusal should say why, got '{}'", stolen.error());
    }

    // But a leftover from a crashed daemon must not block a restart forever.
    first.Stop();
    RNET_CHECK_OK(second.Listen(dir / "rnet.sock"));
    second.Stop();
    std::filesystem::remove_all(dir);
}

// A path that does not fit sun_path is refused rather than truncated: a truncated
// path binds somewhere other than where the caller asked.
RNET_TEST(IpcServer, RefusesAPathThatWouldBeTruncated) {
    Server server;
    const std::string long_name(120, 'x');
    auto refused = server.Listen(std::filesystem::path("/tmp") / long_name / "rnet.sock");
    RNET_CHECK_ERR(refused);
    std::filesystem::remove_all(std::filesystem::path("/tmp") / long_name);
}

// The whole point: a frame written by a client is delivered whole.
RNET_TEST(IpcServer, DeliversAFrameFromAConnectedClient) {
    const auto dir = TestSocketDir("deliver");
    std::filesystem::remove_all(dir);

    Server server;
    RNET_CHECK_OK(server.Listen(dir / "rnet.sock"));

    const int client = ConnectRaw(dir / "rnet.sock");
    RNET_CHECK(client >= 0);

    // First poll accepts; the client is registered before it has said anything.
    auto first = server.Poll(1000);
    RNET_CHECK_OK(first);
    RNET_CHECK_EQ(server.client_count(), size_t{1});

    const auto record = HelloRecord(42);
    RNET_CHECK(!record.empty());
    RNET_CHECK_EQ(::send(client, record.data(), record.size(), 0),
                  static_cast<ssize_t>(record.size()));

    // A later poll must find it. This is the step that a daemon which accepts but
    // never reads would fail.
    bool delivered = false;
    for (int i = 0; i < 20 && !delivered; ++i) {
        auto polled = server.Poll(1000 + i * 10);
        RNET_CHECK_OK(polled);
        for (auto& d : polled.value()) {
            if (d.received.frame.type == MessageType::Hello && d.received.frame.request_id == 42) {
                delivered = true;
            }
        }
    }
    if (!delivered) RNET_FAIL("a hello frame was never delivered to the server");

    ::close(client);
    server.Stop();
    std::filesystem::remove_all(dir);
}

// A worker that connects and says nothing costs it nothing and costs the daemon a
// slot.
RNET_TEST(IpcServer, SilentClientsLoseTheirSlot) {
    const auto dir = TestSocketDir("silent");
    std::filesystem::remove_all(dir);

    Server server;
    RNET_CHECK_OK(server.Listen(dir / "rnet.sock"));
    const int client = ConnectRaw(dir / "rnet.sock");
    RNET_CHECK(client >= 0);

    RNET_CHECK_OK(server.Poll(1000));
    RNET_CHECK_EQ(server.client_count(), size_t{1});

    server.Sweep(1000 + kHandshakeTimeoutMs + 1);
    RNET_CHECK_EQ(server.client_count(), size_t{0});

    ::close(client);
    server.Stop();
    std::filesystem::remove_all(dir);
}

// A reply replayed at the daemon must not be mistaken for a request. Each type is
// legal in one direction only.
RNET_TEST(IpcServer, RefusesAMessageTravellingTheWrongWay) {
    const auto dir = TestSocketDir("direction");
    std::filesystem::remove_all(dir);

    Server server;
    RNET_CHECK_OK(server.Listen(dir / "rnet.sock"));
    const int client = ConnectRaw(dir / "rnet.sock");
    RNET_CHECK(client >= 0);
    RNET_CHECK_OK(server.Poll(1000));

    Frame reply;
    reply.type = MessageType::HelloOk;   // a daemon-to-worker message
    reply.request_id = 1;
    reply.payload = CborWriter().BeginMap(1).Key("a").Uint(1).Take().value();
    auto encoded = EncodeFrame(reply);
    RNET_CHECK_OK(encoded);
    RNET_CHECK(::send(client, encoded.value().data(), encoded.value().size(), 0) > 0);

    for (int i = 0; i < 20; ++i) (void)server.Poll(1000 + i * 10);
    server.Sweep(1100);
    RNET_CHECK_EQ(server.client_count(), size_t{0});

    ::close(client);
    server.Stop();
    std::filesystem::remove_all(dir);
}

// A descriptor that the sender can still write to would let the bytes change
// between being hashed and being served to peers.
RNET_TEST(IpcServer, RefusesADescriptorThatIsNotSealed) {
    // An ordinary pipe: not a regular file at all.
    int fds[2];
    RNET_CHECK_EQ(::pipe(fds), 0);
    RNET_CHECK_ERR(SealedSizeOf(fds[0]));
    ::close(fds[0]);
    ::close(fds[1]);

    // A memfd that was never sealed: a regular file the sender can still modify.
    const int unsealed = ::memfd_create("rnet-test-unsealed", MFD_ALLOW_SEALING);
    RNET_CHECK(unsealed >= 0);
    const std::vector<uint8_t> data(64, 0x7);
    RNET_CHECK_EQ(::write(unsealed, data.data(), data.size()),
                  static_cast<ssize_t>(data.size()));
    auto unsealed_size = SealedSizeOf(unsealed);
    RNET_CHECK_ERR(unsealed_size);
    if (unsealed_size.error().find("sealed") == std::string::npos) {
        RNET_FAIL("the refusal should name sealing, got '{}'", unsealed_size.error());
    }

    // Sealed against write, shrink and grow: now the bytes cannot change.
    RNET_CHECK_EQ(::fcntl(unsealed, F_ADD_SEALS, F_SEAL_WRITE | F_SEAL_SHRINK | F_SEAL_GROW), 0);
    auto sealed_size = SealedSizeOf(unsealed);
    RNET_CHECK_OK(sealed_size);
    RNET_CHECK_EQ(sealed_size.value(), uint64_t{64});
    ::close(unsealed);
}
