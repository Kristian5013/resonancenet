#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "script/opcodes.h"

namespace rnet::script {

/// Maximum script length in bytes.
static constexpr size_t MAX_SCRIPT_SIZE = 10000;

/// Maximum number of non-push operations per script.
static constexpr int MAX_OPS_PER_SCRIPT = 201;

/// Maximum number of elements on the stack.
static constexpr int MAX_STACK_SIZE = 1000;

/// Maximum size of a script element (push data).
static constexpr size_t MAX_SCRIPT_ELEMENT_SIZE = 520;

/// Threshold for OP_CHECKMULTISIG key count.
static constexpr int MAX_PUBKEYS_PER_MULTISIG = 20;

// ── Script number encoding ──────────────────────────────────────────

/// Default maximum size for script numbers (4 bytes).
static constexpr size_t DEFAULT_SCRIPTNUM_SIZE = 4;

/// Encode an integer as a script number byte vector.
/// Script numbers are little-endian, sign-magnitude.
std::vector<uint8_t> scriptnum_encode(int64_t value);

/// Decode a script number byte vector to an integer.
/// Returns 0 if the vector is empty.
/// max_size: maximum allowed byte vector size (default 4).
/// require_minimal: if true, reject non-minimal encodings.
int64_t scriptnum_decode(const std::vector<uint8_t>& data,
                         size_t max_size = DEFAULT_SCRIPTNUM_SIZE,
                         bool require_minimal = true);

/// Check if a byte vector is a minimal encoding of a number.
bool is_minimal_scriptnum(const std::vector<uint8_t>& data);

// ── CScript ─────────────────────────────────────────────────────────

/// Script is a sequence of opcodes and data pushes.
/// Inherits from vector<uint8_t> for direct byte access.
class CScript : public std::vector<uint8_t> {
public:
    using std::vector<uint8_t>::vector;

    CScript() = default;

    /// Construct from a byte vector.
    explicit CScript(const std::vector<uint8_t>& v)
        : std::vector<uint8_t>(v) {}

    explicit CScript(std::vector<uint8_t>&& v)
        : std::vector<uint8_t>(std::move(v)) {}

    /// Push an opcode.
    CScript& operator<<(Opcode op);

    /// Push data with automatic encoding (minimal push rules):
    ///   - 0 bytes: OP_0
    ///   - 1 byte in [1..16]: OP_1..OP_16
    ///   - 1 byte == 0x81: OP_1NEGATE
    ///   - 1..75 bytes: [len] [data]
    ///   - 76..255 bytes: OP_PUSHDATA1 [1-byte len] [data]
    ///   - 256..65535 bytes: OP_PUSHDATA2 [2-byte LE len] [data]
    ///   - larger: OP_PUSHDATA4 [4-byte LE len] [data]
    CScript& operator<<(const std::vector<uint8_t>& data);

    /// Push a number as a script number.
    CScript& operator<<(int64_t num);

    /// Check if this script matches the P2SH pattern:
    /// OP_HASH160 [20 bytes] OP_EQUAL
    bool is_pay_to_script_hash() const;

    /// Check if this is a witness program.
    /// Sets version (0-16) and program (2..40 bytes).
    /// Pattern: OP_n [2..40 bytes]
    bool is_witness_program(int& version,
                            std::vector<uint8_t>& program) const;

    /// Check if this script starts with OP_RETURN.
    bool is_unspendable() const;

    /// Count signature operations (for block sigop limit).
    /// If accurate is true, count P2SH subscript sigops accurately.
    unsigned int get_sig_op_count(bool accurate) const;

    /// Check if every element is a push operation (data or opcode push).
    bool is_push_only() const;

    /// Get the script as a human-readable string.
    std::string to_string() const;

    /// Get a subscript starting at the given position.
    CScript subscript(size_t start) const;

    /// Find and delete a specific byte sequence from the script.
    /// Used for OP_CODESEPARATOR removal during signature hashing.
    CScript& find_and_delete(const CScript& needle);

    /// Serialization support.
    template<typename Stream>
    void serialize(Stream& s) const {
        auto sz = static_cast<uint64_t>(size());
        // Write compact size
        uint8_t buf[9];
        size_t len = 0;
        if (sz < 253) {
            buf[0] = static_cast<uint8_t>(sz);
            len = 1;
        } else if (sz <= 0xFFFF) {
            buf[0] = 253;
            buf[1] = static_cast<uint8_t>(sz & 0xFF);
            buf[2] = static_cast<uint8_t>((sz >> 8) & 0xFF);
            len = 3;
        } else if (sz <= 0xFFFFFFFF) {
            buf[0] = 254;
            buf[1] = static_cast<uint8_t>(sz & 0xFF);
            buf[2] = static_cast<uint8_t>((sz >> 8) & 0xFF);
            buf[3] = static_cast<uint8_t>((sz >> 16) & 0xFF);
            buf[4] = static_cast<uint8_t>((sz >> 24) & 0xFF);
            len = 5;
        } else {
            buf[0] = 255;
            for (int i = 0; i < 8; ++i) {
                buf[1 + i] = static_cast<uint8_t>((sz >> (i * 8)) & 0xFF);
            }
            len = 9;
        }
        s.write(buf, len);
        if (!empty()) {
            s.write(data(), size());
        }
    }

    template<typename Stream>
    void unserialize(Stream& s) {
        // Read compact size
        uint8_t first = 0;
        s.read(&first, 1);
        uint64_t sz = 0;
        if (first < 253) {
            sz = first;
        } else if (first == 253) {
            uint8_t b[2];
            s.read(b, 2);
            sz = static_cast<uint64_t>(b[0])
               | (static_cast<uint64_t>(b[1]) << 8);
        } else if (first == 254) {
            uint8_t b[4];
            s.read(b, 4);
            sz = static_cast<uint64_t>(b[0])
               | (static_cast<uint64_t>(b[1]) << 8)
               | (static_cast<uint64_t>(b[2]) << 16)
               | (static_cast<uint64_t>(b[3]) << 24);
        } else {
            uint8_t b[8];
            s.read(b, 8);
            sz = 0;
            for (int i = 0; i < 8; ++i) {
                sz |= static_cast<uint64_t>(b[i]) << (i * 8);
            }
        }
        if (sz > MAX_SCRIPT_SIZE) {
            clear();
            return;
        }
        resize(static_cast<size_t>(sz));
        if (sz > 0) {
            s.read(data(), static_cast<size_t>(sz));
        }
    }
};

/// Iterator helper to walk through script opcodes and data.
class ScriptIterator {
public:
    explicit ScriptIterator(const CScript& script);

    /// Get the next opcode and its associated data push.
    /// Returns false when the end of the script is reached.
    bool next(Opcode& op, std::vector<uint8_t>& data);

    /// Current position in the script.
    size_t pos() const { return pos_; }

    /// Check if we have reached the end.
    bool done() const { return pos_ >= script_.size(); }

private:
    const CScript& script_;
    size_t pos_ = 0;
};

}  // namespace rnet::script
