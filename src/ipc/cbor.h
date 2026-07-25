// CBOR, restricted to exactly one encoding per value.
//
// This project already has a canonical format for anything that gets hashed. CBOR
// is here only as an IPC envelope — but a second parser that accepts several
// spellings of the same message is a place for meaning to hide, and this codebase
// spends its whole design removing those. So what is implemented is not "CBOR",
// it is a profile with one legal encoding of every value:
//
//   * DEFINITE LENGTHS ONLY. Indefinite-length strings, arrays and maps are
//     rejected. They are a second way to write what a definite length already
//     says, and they let a sender stream an unbounded item past any size check
//     performed on the head.
//
//   * MINIMAL INTEGERS. 1 must be written as 0x01, never as 0x1801. A decoder
//     that accepts both accepts two byte strings that mean the same thing, and any
//     check performed on bytes rather than meaning can then be walked around.
//
//   * MAP KEYS ARE TEXT, SORTED, UNIQUE. Sorted by their encoded bytes, so the
//     order is a property of the keys rather than of whoever wrote them.
//     Duplicates are rejected outright: "last one wins" and "first one wins" are
//     both defensible, which is exactly why two implementations will disagree.
//
//   * NO TAGS, NO FLOATS, NO NULL, NO UNDEFINED. Floats are the worst of these:
//     half, single and double can all encode 1.0, NaN carries a payload, and -0.0
//     is a distinct bit pattern that compares equal to 0.0. Numbers that need a
//     fraction cross this boundary as integers with a fixed scale, as they do
//     everywhere else in the protocol.
//
// Everything is bounded before anything is allocated: a length prefix is a claim
// by the sender, and claims are checked against the bytes actually present.
#pragma once

#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "util/result.h"

namespace rnet::ipc {

namespace detail {
class CborParserImpl;
}  // namespace detail

// Nesting past this is refused. Our messages are a map of scalars with at most an
// array of maps inside; anything deeper is either a bug or an attempt to exhaust
// the parser's stack.
inline constexpr uint32_t kMaxCborDepth = 8;

// Items in one container. A bound the sender cannot argue with.
inline constexpr uint32_t kMaxCborItems = 65'536;

enum class CborType : uint8_t {
    Uint,
    Nint,   // negative integer; value() returns the mathematical value
    Bytes,
    Text,
    Array,
    Map,
    Bool,
};

// A decoded value. Owning and simple: IPC frames are small by construction, and
// bulk payloads never travel as a CBOR item.
class CborValue {
public:
    CborType type() const { return type_; }

    // Accessors that state what they require. Each fails rather than converting,
    // because a silent conversion is how a length gets read out of a boolean.
    Result<uint64_t> AsUint() const;
    Result<int64_t> AsInt() const;      // accepts Uint (if it fits) and Nint
    Result<std::span<const uint8_t>> AsBytes() const;
    Result<std::string_view> AsText() const;
    Result<bool> AsBool() const;

    const std::vector<CborValue>& items() const { return items_; }
    const std::map<std::string, CborValue>& entries() const { return entries_; }

    // Map lookup. `Get` fails when the key is absent; `Find` returns nullptr, for
    // the cases where absence is legitimate.
    Result<const CborValue*> Get(std::string_view key) const;
    const CborValue* Find(std::string_view key) const;

    // Typed field readers, so a message parser reads as a list of requirements
    // rather than a chain of checks that is easy to leave incomplete.
    Result<uint64_t> GetUint(std::string_view key) const;
    Result<int64_t> GetInt(std::string_view key) const;
    Result<std::vector<uint8_t>> GetBytes(std::string_view key) const;
    Result<std::array<uint8_t, 32>> GetHash(std::string_view key) const;
    Result<std::string> GetText(std::string_view key) const;
    Result<bool> GetBool(std::string_view key) const;

    static CborValue MakeUint(uint64_t v);
    static CborValue MakeNint(int64_t v);
    static CborValue MakeBool(bool v);

private:
    // The parser builds values field by field; it is the only thing that may.
    friend class detail::CborParserImpl;

    CborType type_{CborType::Uint};
    uint64_t uint_{0};
    int64_t nint_{0};
    bool bool_{false};
    std::vector<uint8_t> bytes_;              // Bytes and Text
    std::vector<CborValue> items_;            // Array
    std::map<std::string, CborValue> entries_;  // Map
};

// Parses one complete value and requires the input to be exhausted. Trailing
// bytes are an error: an attacker who can append data that some parsers ignore
// can make two nodes read the same frame differently.
Result<CborValue> CborParse(std::span<const uint8_t> raw);

// Writes the profile, and enforces it while writing rather than trusting the
// caller. Out-of-order or duplicate map keys, and containers whose declared count
// does not match what was written, are errors — caught here, at the author's
// desk, instead of at a peer's.
class CborWriter {
public:
    CborWriter& Uint(uint64_t v);
    CborWriter& Int(int64_t v);
    CborWriter& Bytes(std::span<const uint8_t> data);
    CborWriter& Text(std::string_view text);
    CborWriter& Bool(bool v);

    // Containers declare their length up front, which is what makes them
    // definite. `count` is checked against what actually follows.
    CborWriter& BeginArray(uint32_t count);
    CborWriter& BeginMap(uint32_t count);

    // A map key. Must be strictly greater than the previous key in the same map,
    // by encoded-byte order.
    CborWriter& Key(std::string_view key);

    // The finished bytes, or the first error encountered. Errors are held rather
    // than thrown so a message can be written as one expression.
    Result<std::vector<uint8_t>> Take();

private:
    void Head(uint8_t major, uint64_t value);
    void Fail(std::string reason);
    void CountItem();

    // True if a value may be written now. A frame holds exactly one top-level
    // value: two concatenated values are two documents, and a reader that stops
    // at the first would silently ignore the second.
    bool BeginItem();

    struct Frame {
        bool is_map{false};
        uint32_t remaining{0};      // items still expected (map: keys + values)
        bool expecting_key{false};
        std::vector<uint8_t> last_key;   // encoded, for the ordering check
    };

    std::vector<uint8_t> buf_;
    std::vector<Frame> stack_;
    std::string error_;
};

// Encoded-byte ordering for map keys, exposed because it is the rule two
// implementations must agree on and therefore deserves a direct test.
bool CborKeyLess(std::string_view a, std::string_view b);

}  // namespace rnet::ipc
