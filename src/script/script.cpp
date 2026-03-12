#include "script/script.h"

#include <algorithm>
#include <cstring>
#include <sstream>

namespace rnet::script {

// ── Script number encoding ──────────────────────────────────────────

std::vector<uint8_t> scriptnum_encode(int64_t value) {
    if (value == 0) return {};

    std::vector<uint8_t> result;
    bool negative = value < 0;
    uint64_t absval = negative ? static_cast<uint64_t>(-value)
                               : static_cast<uint64_t>(value);

    while (absval > 0) {
        result.push_back(static_cast<uint8_t>(absval & 0xFF));
        absval >>= 8;
    }

    // If the most significant byte has the sign bit set,
    // add an extra byte for the sign.
    if (result.back() & 0x80) {
        result.push_back(negative ? 0x80 : 0x00);
    } else if (negative) {
        result.back() |= 0x80;
    }

    return result;
}

int64_t scriptnum_decode(const std::vector<uint8_t>& data,
                         size_t max_size,
                         bool require_minimal) {
    if (data.empty()) return 0;
    if (data.size() > max_size) return 0;

    if (require_minimal && !is_minimal_scriptnum(data)) {
        return 0;
    }

    int64_t result = 0;
    for (size_t i = 0; i < data.size(); ++i) {
        result |= static_cast<int64_t>(data[i]) << (8 * i);
    }

    // Check sign bit of the most significant byte.
    if (data.back() & 0x80) {
        // Negative: mask off the sign bit and negate.
        result &= ~(static_cast<int64_t>(0x80) << (8 * (data.size() - 1)));
        result = -result;
    }

    return result;
}

bool is_minimal_scriptnum(const std::vector<uint8_t>& data) {
    if (data.empty()) return true;

    // Check for non-minimal encoding: trailing zeros.
    // The last byte's non-sign bits should not be all zero unless
    // the penultimate byte has the sign bit set.
    if ((data.back() & 0x7F) == 0) {
        if (data.size() <= 1) return false;
        if ((data[data.size() - 2] & 0x80) == 0) return false;
    }
    return true;
}

// ── CScript ─────────────────────────────────────────────────────────

CScript& CScript::operator<<(Opcode op) {
    push_back(static_cast<uint8_t>(op));
    return *this;
}

CScript& CScript::operator<<(const std::vector<uint8_t>& data) {
    if (data.empty()) {
        push_back(static_cast<uint8_t>(Opcode::OP_0));
    } else if (data.size() == 1 && data[0] >= 1 && data[0] <= 16) {
        push_back(static_cast<uint8_t>(Opcode::OP_1) + (data[0] - 1));
    } else if (data.size() == 1 && data[0] == 0x81) {
        push_back(static_cast<uint8_t>(Opcode::OP_1NEGATE));
    } else if (data.size() <= 75) {
        push_back(static_cast<uint8_t>(data.size()));
        insert(end(), data.begin(), data.end());
    } else if (data.size() <= 255) {
        push_back(static_cast<uint8_t>(Opcode::OP_PUSHDATA1));
        push_back(static_cast<uint8_t>(data.size()));
        insert(end(), data.begin(), data.end());
    } else if (data.size() <= 65535) {
        push_back(static_cast<uint8_t>(Opcode::OP_PUSHDATA2));
        auto sz = static_cast<uint16_t>(data.size());
        push_back(static_cast<uint8_t>(sz & 0xFF));
        push_back(static_cast<uint8_t>((sz >> 8) & 0xFF));
        insert(end(), data.begin(), data.end());
    } else {
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

CScript& CScript::operator<<(int64_t num) {
    if (num == -1 || (num >= 1 && num <= 16)) {
        push_back(static_cast<uint8_t>(num + static_cast<int>(Opcode::OP_1) - 1));
    } else if (num == 0) {
        push_back(static_cast<uint8_t>(Opcode::OP_0));
    } else {
        auto encoded = scriptnum_encode(num);
        *this << encoded;
    }
    return *this;
}

bool CScript::is_pay_to_script_hash() const {
    return size() == 23 &&
           (*this)[0] == static_cast<uint8_t>(Opcode::OP_HASH160) &&
           (*this)[1] == 0x14 &&  // push 20 bytes
           (*this)[22] == static_cast<uint8_t>(Opcode::OP_EQUAL);
}

bool CScript::is_witness_program(int& version,
                                 std::vector<uint8_t>& program) const {
    if (size() < 4 || size() > 42) return false;

    uint8_t first = (*this)[0];

    // Version byte: OP_0 (0x00) or OP_1..OP_16 (0x51..0x60)
    if (first == 0x00) {
        version = 0;
    } else if (first >= 0x51 && first <= 0x60) {
        version = first - 0x50;
    } else {
        return false;
    }

    // Second byte is the program length (direct push).
    uint8_t prog_len = (*this)[1];
    if (prog_len < 2 || prog_len > 40) return false;
    if (static_cast<size_t>(prog_len + 2) != size()) return false;

    program.assign(begin() + 2, end());
    return true;
}

bool CScript::is_unspendable() const {
    return !empty() && (*this)[0] == static_cast<uint8_t>(Opcode::OP_RETURN);
}

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

bool CScript::is_push_only() const {
    ScriptIterator it(*this);
    Opcode op;
    std::vector<uint8_t> data;

    while (it.next(op, data)) {
        auto val = static_cast<uint8_t>(op);
        // Pushdata opcodes: 0x00..0x60 (OP_0 through OP_16)
        // Plus OP_PUSHDATA1/2/4 (0x4c..0x4e)
        // And OP_1NEGATE (0x4f)
        if (val > static_cast<uint8_t>(Opcode::OP_16)) {
            return false;
        }
    }
    return true;
}

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
            // Data push — show hex
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

CScript CScript::subscript(size_t start) const {
    if (start >= size()) return CScript{};
    CScript result;
    result.assign(begin() + static_cast<ptrdiff_t>(start), end());
    return result;
}

CScript& CScript::find_and_delete(const CScript& needle) {
    if (needle.empty()) return *this;

    auto it = std::search(begin(), end(), needle.begin(), needle.end());
    while (it != end()) {
        it = erase(it, it + static_cast<ptrdiff_t>(needle.size()));
        it = std::search(it, end(), needle.begin(), needle.end());
    }
    return *this;
}

// ── ScriptIterator ──────────────────────────────────────────────────

ScriptIterator::ScriptIterator(const CScript& script)
    : script_(script) {}

bool ScriptIterator::next(Opcode& op, std::vector<uint8_t>& data) {
    data.clear();
    if (pos_ >= script_.size()) return false;

    uint8_t opcode = script_[pos_++];
    op = static_cast<Opcode>(opcode);

    if (opcode == 0x00) {
        // OP_0: push empty
        return true;
    }

    if (opcode >= 0x01 && opcode <= 0x4b) {
        // Direct push: next opcode bytes
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

    // Non-push opcode
    return true;
}

}  // namespace rnet::script
