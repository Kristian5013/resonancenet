// Copyright (c) 2024-2026 The ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "script/script.h"

#include <algorithm>
#include <cstring>
#include <sstream>

namespace rnet::script {

// ---------------------------------------------------------------------------
// scriptnum_encode
//
// Bitcoin-style variable-length signed integer encoding.
// Byte order is little-endian; sign lives in the MSB of the last byte.
//
//   value  0  -->  []  (empty)
//   value  5  -->  [0x05]
//   value -5  -->  [0x85]           (0x05 | 0x80)
//   value 255 -->  [0xFF, 0x00]     (need extra byte: 0xFF has sign bit set)
// ---------------------------------------------------------------------------

std::vector<uint8_t> scriptnum_encode(int64_t value) {
    // 1. Zero encodes as empty vector.
    if (value == 0) return {};

    std::vector<uint8_t> result;
    bool negative = value < 0;
    uint64_t absval = negative ? static_cast<uint64_t>(-value)
                               : static_cast<uint64_t>(value);

    // 2. Emit bytes least-significant first.
    while (absval > 0) {
        result.push_back(static_cast<uint8_t>(absval & 0xFF));
        absval >>= 8;
    }

    // 3. If the MSB of the last byte has the sign bit set,
    //    add an extra byte to carry the sign.
    if (result.back() & 0x80) {
        result.push_back(negative ? 0x80 : 0x00);
    } else if (negative) {
        result.back() |= 0x80;
    }

    return result;
}

// ---------------------------------------------------------------------------
// scriptnum_decode
//
// Inverse of scriptnum_encode.  Little-endian bytes -> int64_t.
// Sign bit is the MSB of the last byte.
// ---------------------------------------------------------------------------

int64_t scriptnum_decode(const std::vector<uint8_t>& data,
                         size_t max_size,
                         bool require_minimal) {
    if (data.empty()) return 0;
    if (data.size() > max_size) return 0;

    // 1. Reject non-minimal encodings if requested.
    if (require_minimal && !is_minimal_scriptnum(data)) {
        return 0;
    }

    // 2. Accumulate little-endian bytes.
    int64_t result = 0;
    for (size_t i = 0; i < data.size(); ++i) {
        result |= static_cast<int64_t>(data[i]) << (8 * i);
    }

    // 3. Extract sign from the MSB of the last byte.
    if (data.back() & 0x80) {
        result &= ~(static_cast<int64_t>(0x80) << (8 * (data.size() - 1)));
        result = -result;
    }

    return result;
}

// ---------------------------------------------------------------------------
// is_minimal_scriptnum
//
// A script number is minimal if:
//   - The last byte's non-sign bits are not all zero, OR
//   - The penultimate byte has its sign bit set (justifying the extra byte).
// ---------------------------------------------------------------------------

bool is_minimal_scriptnum(const std::vector<uint8_t>& data) {
    if (data.empty()) return true;

    // 1. Check for non-minimal trailing zero.
    if ((data.back() & 0x7F) == 0) {
        if (data.size() <= 1) return false;
        if ((data[data.size() - 2] & 0x80) == 0) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// CScript::operator<<(Opcode)
//
// Push a single opcode byte onto the script.
// ---------------------------------------------------------------------------

CScript& CScript::operator<<(Opcode op) {
    push_back(static_cast<uint8_t>(op));
    return *this;
}

// ---------------------------------------------------------------------------
// CScript::operator<<(vector<uint8_t>)
//
// Push data with the most compact encoding:
//   len == 0        -->  OP_0
//   len == 1, 1..16 -->  OP_1 .. OP_16
//   len == 1, 0x81  -->  OP_1NEGATE
//   len <= 75       -->  [len][data]          (direct push)
//   len <= 255      -->  OP_PUSHDATA1 [1-byte len][data]
//   len <= 65535    -->  OP_PUSHDATA2 [2-byte LE len][data]
//   else            -->  OP_PUSHDATA4 [4-byte LE len][data]
// ---------------------------------------------------------------------------

CScript& CScript::operator<<(const std::vector<uint8_t>& data) {
    if (data.empty()) {
        // 1. Empty data -> OP_0.
        push_back(static_cast<uint8_t>(Opcode::OP_0));
    } else if (data.size() == 1 && data[0] >= 1 && data[0] <= 16) {
        // 2. Single byte 1..16 -> OP_1..OP_16.
        push_back(static_cast<uint8_t>(Opcode::OP_1) + (data[0] - 1));
    } else if (data.size() == 1 && data[0] == 0x81) {
        // 3. Single byte 0x81 -> OP_1NEGATE.
        push_back(static_cast<uint8_t>(Opcode::OP_1NEGATE));
    } else if (data.size() <= 75) {
        // 4. Direct push: length byte is the opcode.
        push_back(static_cast<uint8_t>(data.size()));
        insert(end(), data.begin(), data.end());
    } else if (data.size() <= 255) {
        // 5. OP_PUSHDATA1 + 1-byte length.
        push_back(static_cast<uint8_t>(Opcode::OP_PUSHDATA1));
        push_back(static_cast<uint8_t>(data.size()));
        insert(end(), data.begin(), data.end());
    } else if (data.size() <= 65535) {
        // 6. OP_PUSHDATA2 + 2-byte little-endian length.
        push_back(static_cast<uint8_t>(Opcode::OP_PUSHDATA2));
        auto sz = static_cast<uint16_t>(data.size());
        push_back(static_cast<uint8_t>(sz & 0xFF));
        push_back(static_cast<uint8_t>((sz >> 8) & 0xFF));
        insert(end(), data.begin(), data.end());
    } else {
        // 7. OP_PUSHDATA4 + 4-byte little-endian length.
        push_back(static_cast<uint8_t>(Opcode::OP_PUSHDATA4));
        auto sz = static_cast<uint32_t>(data.size());
        push_back(static_cast<uint8_t>(sz & 0xFF));
        push_back(static_cast<uint8_t>((sz >> 8) & 0xFF));
        push_back(static_cast<uint8_t>((sz >> 16) & 0xFF));
        push_back(static_cast<uint8_t>((sz >> 24) & 0xFF));
        insert(end(), data.begin(), data.end());
    }
    return *this;
}

// ---------------------------------------------------------------------------
// CScript::operator<<(int64_t)
//
// Push a numeric literal.  Small values use dedicated opcodes.
// ---------------------------------------------------------------------------

CScript& CScript::operator<<(int64_t num) {
    if (num == -1 || (num >= 1 && num <= 16)) {
        // 1. OP_1NEGATE or OP_1..OP_16.
        push_back(static_cast<uint8_t>(num + static_cast<int>(Opcode::OP_1) - 1));
    } else if (num == 0) {
        // 2. OP_0.
        push_back(static_cast<uint8_t>(Opcode::OP_0));
    } else {
        // 3. Encode as script number and push.
        auto encoded = scriptnum_encode(num);
        *this << encoded;
    }
    return *this;
}

// ---------------------------------------------------------------------------
// CScript::is_pay_to_script_hash
//
// P2SH pattern: OP_HASH160 [20 bytes] OP_EQUAL  (exactly 23 bytes)
// ---------------------------------------------------------------------------

bool CScript::is_pay_to_script_hash() const {
    return size() == 23 &&
           (*this)[0] == static_cast<uint8_t>(Opcode::OP_HASH160) &&
           (*this)[1] == 0x14 &&
           (*this)[22] == static_cast<uint8_t>(Opcode::OP_EQUAL);
}

// ---------------------------------------------------------------------------
// CScript::is_witness_program
//
// Witness program: OP_n [2..40 byte push]
//   version = 0   for OP_0  (0x00)
//   version = 1..16 for OP_1..OP_16 (0x51..0x60)
// Total script size: 4..42 bytes.
// ---------------------------------------------------------------------------

bool CScript::is_witness_program(int& version,
                                 std::vector<uint8_t>& program) const {
    if (size() < 4 || size() > 42) return false;

    uint8_t first = (*this)[0];

    // 1. Decode version byte.
    if (first == 0x00) {
        version = 0;
    } else if (first >= 0x51 && first <= 0x60) {
        version = first - 0x50;
    } else {
        return false;
    }

    // 2. Second byte is the program length (direct push).
    uint8_t prog_len = (*this)[1];
    if (prog_len < 2 || prog_len > 40) return false;
    if (static_cast<size_t>(prog_len + 2) != size()) return false;

    program.assign(begin() + 2, end());
    return true;
}

// ---------------------------------------------------------------------------
// CScript::is_unspendable
//
// A script starting with OP_RETURN is provably unspendable.
// ---------------------------------------------------------------------------

bool CScript::is_unspendable() const {
    return !empty() && (*this)[0] == static_cast<uint8_t>(Opcode::OP_RETURN);
}

// ---------------------------------------------------------------------------
// CScript::get_sig_op_count
//
// Count signature-checking opcodes for consensus weight limits.
// When accurate=true, CHECKMULTISIG uses the actual N from the preceding
// OP_N; otherwise assumes MAX_PUBKEYS_PER_MULTISIG.
// ---------------------------------------------------------------------------

unsigned int CScript::get_sig_op_count(bool accurate) const {
    unsigned int count = 0;
    ScriptIterator it(*this);
    Opcode last_op = Opcode::OP_INVALIDOPCODE;
    Opcode op;
    std::vector<uint8_t> data;

    while (it.next(op, data)) {
        if (op == Opcode::OP_CHECKSIG ||
            op == Opcode::OP_CHECKSIGVERIFY) {
            ++count;
        } else if (op == Opcode::OP_CHECKMULTISIG ||
                   op == Opcode::OP_CHECKMULTISIGVERIFY) {
            if (accurate) {
                int n = decode_op_n(last_op);
                if (n >= 1 && n <= MAX_PUBKEYS_PER_MULTISIG) {
                    count += static_cast<unsigned int>(n);
                } else {
                    count += MAX_PUBKEYS_PER_MULTISIG;
                }
            } else {
                count += MAX_PUBKEYS_PER_MULTISIG;
            }
        }
        last_op = op;
    }
    return count;
}

// ---------------------------------------------------------------------------
// CScript::is_push_only
//
// Returns true if every opcode is a data-push (0x00..0x60).
// ---------------------------------------------------------------------------

bool CScript::is_push_only() const {
    ScriptIterator it(*this);
    Opcode op;
    std::vector<uint8_t> data;

    while (it.next(op, data)) {
        auto val = static_cast<uint8_t>(op);
        if (val > static_cast<uint8_t>(Opcode::OP_16)) {
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// CScript::to_string
//
// Human-readable disassembly: data pushes as hex, opcodes by name.
// ---------------------------------------------------------------------------

std::string CScript::to_string() const {
    std::ostringstream oss;
    ScriptIterator it(*this);
    Opcode op;
    std::vector<uint8_t> data;
    bool first = true;

    while (it.next(op, data)) {
        if (!first) oss << " ";
        first = false;

        auto val = static_cast<uint8_t>(op);
        if (!data.empty()) {
            for (auto b : data) {
                static constexpr char hex_chars[] = "0123456789abcdef";
                oss << hex_chars[(b >> 4) & 0x0F];
                oss << hex_chars[b & 0x0F];
            }
        } else {
            oss << opcode_name(op);
        }
    }
    return oss.str();
}

// ---------------------------------------------------------------------------
// CScript::subscript
//
// Return a new CScript from byte offset `start` to the end.
// ---------------------------------------------------------------------------

CScript CScript::subscript(size_t start) const {
    if (start >= size()) return CScript{};
    CScript result;
    result.assign(begin() + static_cast<ptrdiff_t>(start), end());
    return result;
}

// ---------------------------------------------------------------------------
// CScript::find_and_delete
//
// Remove all occurrences of `needle` from this script (in-place).
// ---------------------------------------------------------------------------

CScript& CScript::find_and_delete(const CScript& needle) {
    if (needle.empty()) return *this;

    auto it = std::search(begin(), end(), needle.begin(), needle.end());
    while (it != end()) {
        it = erase(it, it + static_cast<ptrdiff_t>(needle.size()));
        it = std::search(it, end(), needle.begin(), needle.end());
    }
    return *this;
}

// ---------------------------------------------------------------------------
// ScriptIterator
//
// Walks a CScript byte-by-byte, decoding opcodes and their data payloads.
//
// Encoding map:
//   0x00           OP_0            push empty
//   0x01..0x4b     direct push     next N bytes
//   0x4c           OP_PUSHDATA1    1-byte length, then data
//   0x4d           OP_PUSHDATA2    2-byte LE length, then data
//   0x4e           OP_PUSHDATA4    4-byte LE length, then data
//   0x4f..0xff     non-push ops    no data payload
// ---------------------------------------------------------------------------

ScriptIterator::ScriptIterator(const CScript& script)
    : script_(script) {}

bool ScriptIterator::next(Opcode& op, std::vector<uint8_t>& data) {
    data.clear();
    if (pos_ >= script_.size()) return false;

    uint8_t opcode = script_[pos_++];
    op = static_cast<Opcode>(opcode);

    // 1. OP_0: push empty.
    if (opcode == 0x00) {
        return true;
    }

    // 2. Direct push: 0x01..0x4b bytes follow.
    if (opcode >= 0x01 && opcode <= 0x4b) {
        size_t push_size = opcode;
        if (pos_ + push_size > script_.size()) {
            pos_ = script_.size();
            return false;
        }
        data.assign(script_.begin() + static_cast<ptrdiff_t>(pos_),
                     script_.begin() + static_cast<ptrdiff_t>(pos_ + push_size));
        pos_ += push_size;
        return true;
    }

    // 3. OP_PUSHDATA1: 1-byte length prefix.
    if (opcode == static_cast<uint8_t>(Opcode::OP_PUSHDATA1)) {
        if (pos_ >= script_.size()) return false;
        size_t push_size = script_[pos_++];
        if (pos_ + push_size > script_.size()) {
            pos_ = script_.size();
            return false;
        }
        data.assign(script_.begin() + static_cast<ptrdiff_t>(pos_),
                     script_.begin() + static_cast<ptrdiff_t>(pos_ + push_size));
        pos_ += push_size;
        return true;
    }

    // 4. OP_PUSHDATA2: 2-byte little-endian length prefix.
    if (opcode == static_cast<uint8_t>(Opcode::OP_PUSHDATA2)) {
        if (pos_ + 2 > script_.size()) return false;
        size_t push_size = static_cast<size_t>(script_[pos_])
                         | (static_cast<size_t>(script_[pos_ + 1]) << 8);
        pos_ += 2;
        if (pos_ + push_size > script_.size()) {
            pos_ = script_.size();
            return false;
        }
        data.assign(script_.begin() + static_cast<ptrdiff_t>(pos_),
                     script_.begin() + static_cast<ptrdiff_t>(pos_ + push_size));
        pos_ += push_size;
        return true;
    }

    // 5. OP_PUSHDATA4: 4-byte little-endian length prefix.
    if (opcode == static_cast<uint8_t>(Opcode::OP_PUSHDATA4)) {
        if (pos_ + 4 > script_.size()) return false;
        size_t push_size = static_cast<size_t>(script_[pos_])
                         | (static_cast<size_t>(script_[pos_ + 1]) << 8)
                         | (static_cast<size_t>(script_[pos_ + 2]) << 16)
                         | (static_cast<size_t>(script_[pos_ + 3]) << 24);
        pos_ += 4;
        if (pos_ + push_size > script_.size()) {
            pos_ = script_.size();
            return false;
        }
        data.assign(script_.begin() + static_cast<ptrdiff_t>(pos_),
                     script_.begin() + static_cast<ptrdiff_t>(pos_ + push_size));
        pos_ += push_size;
        return true;
    }

    // 6. Non-push opcode -- no data payload.
    return true;
}

} // namespace rnet::script
