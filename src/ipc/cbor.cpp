#include "ipc/cbor.h"

#include <algorithm>
#include <cstring>
#include <format>

namespace rnet::ipc {

namespace {

constexpr uint8_t kMajorUint = 0;
constexpr uint8_t kMajorNint = 1;
constexpr uint8_t kMajorBytes = 2;
constexpr uint8_t kMajorText = 3;
constexpr uint8_t kMajorArray = 4;
constexpr uint8_t kMajorMap = 5;
constexpr uint8_t kMajorTag = 6;
constexpr uint8_t kMajorSimple = 7;

constexpr uint8_t kSimpleFalse = 20;
constexpr uint8_t kSimpleTrue = 21;

// The one legal head for a given value. Anything longer is a second spelling.
size_t MinimalHeadLength(uint64_t value) {
    if (value < 24) return 1;
    if (value <= 0xFFull) return 2;
    if (value <= 0xFFFFull) return 3;
    if (value <= 0xFFFFFFFFull) return 5;
    return 9;
}

// The encoded form of a text-string key, which is what the ordering compares.
std::vector<uint8_t> EncodeTextKey(std::string_view key) {
    std::vector<uint8_t> out;
    const uint64_t len = key.size();
    if (len < 24) {
        out.push_back(static_cast<uint8_t>((kMajorText << 5) | len));
    } else if (len <= 0xFF) {
        out.push_back(static_cast<uint8_t>((kMajorText << 5) | 24));
        out.push_back(static_cast<uint8_t>(len));
    } else {
        out.push_back(static_cast<uint8_t>((kMajorText << 5) | 25));
        out.push_back(static_cast<uint8_t>(len >> 8));
        out.push_back(static_cast<uint8_t>(len & 0xFF));
    }
    out.insert(out.end(), key.begin(), key.end());
    return out;
}

}  // namespace

namespace detail {

class CborParserImpl {
public:
    explicit CborParserImpl(std::span<const uint8_t> raw) : data_(raw) {}

    Result<CborValue> ParseValue(uint32_t depth) {
        if (depth > kMaxCborDepth) {
            return Err(std::format("cbor: nesting deeper than {}", kMaxCborDepth));
        }
        auto head = ReadHead();
        if (!head) return Err(head.error());
        const uint8_t major = head.value().major;
        const uint64_t argument = head.value().argument;

        CborValue value;
        switch (major) {
            case kMajorUint:
                value.type_ = CborType::Uint;
                value.uint_ = argument;
                return value;

            case kMajorNint: {
                // Encoded as -1 - n. The most negative representable value would
                // overflow int64_t, and a decoder that wraps silently turns a
                // rejected size into an accepted one.
                if (argument > static_cast<uint64_t>(INT64_MAX)) {
                    return Err("cbor: negative integer out of range");
                }
                value.type_ = CborType::Nint;
                value.nint_ = -1 - static_cast<int64_t>(argument);
                return value;
            }

            case kMajorBytes:
            case kMajorText: {
                // The length is a claim by the sender. Checked against the bytes
                // actually present before anything is allocated.
                if (argument > remaining()) {
                    return Err(std::format("cbor: string claims {} bytes, {} remain", argument,
                                           remaining()));
                }
                const auto* start = data_.data() + pos_;
                pos_ += static_cast<size_t>(argument);
                value.type_ = major == kMajorBytes ? CborType::Bytes : CborType::Text;
                value.bytes_.assign(start, start + argument);
                if (major == kMajorText) {
                    if (auto st = ValidateUtf8(value.bytes_); !st) return Err(st.error());
                }
                return value;
            }

            case kMajorArray: {
                if (argument > kMaxCborItems) {
                    return Err(std::format("cbor: array of {} items exceeds the limit", argument));
                }
                // A container cannot hold more items than there are bytes left to
                // encode them, so this refuses a huge count before reserving.
                if (argument > remaining()) {
                    return Err("cbor: array declares more items than the input can hold");
                }
                value.type_ = CborType::Array;
                value.items_.reserve(static_cast<size_t>(argument));
                for (uint64_t i = 0; i < argument; ++i) {
                    auto item = ParseValue(depth + 1);
                    if (!item) return Err(item.error());
                    value.items_.push_back(std::move(item.value()));
                }
                return value;
            }

            case kMajorMap: {
                if (argument > kMaxCborItems) {
                    return Err(std::format("cbor: map of {} pairs exceeds the limit", argument));
                }
                if (argument * 2 > remaining()) {
                    return Err("cbor: map declares more pairs than the input can hold");
                }
                value.type_ = CborType::Map;
                std::vector<uint8_t> previous_key;
                for (uint64_t i = 0; i < argument; ++i) {
                    // Keys are text only. Integer keys would be a second way to
                    // name the same field, and mixed key types make the ordering
                    // rule depend on which types happen to be present.
                    const size_t key_start = pos_;
                    auto key = ParseValue(depth + 1);
                    if (!key) return Err(key.error());
                    if (key.value().type_ != CborType::Text) {
                        return Err("cbor: map keys must be text strings");
                    }
                    const std::vector<uint8_t> encoded_key(data_.begin() + key_start,
                                                           data_.begin() + pos_);
                    if (i > 0) {
                        if (encoded_key == previous_key) {
                            // "Last wins" and "first wins" are both defensible,
                            // which is exactly why two implementations disagree.
                            return Err("cbor: duplicate map key");
                        }
                        if (!std::lexicographical_compare(previous_key.begin(), previous_key.end(),
                                                          encoded_key.begin(), encoded_key.end())) {
                            return Err("cbor: map keys are not in canonical order");
                        }
                    }
                    previous_key = encoded_key;

                    auto item = ParseValue(depth + 1);
                    if (!item) return Err(item.error());
                    value.entries_.emplace(
                        std::string(reinterpret_cast<const char*>(key.value().bytes_.data()),
                                    key.value().bytes_.size()),
                        std::move(item.value()));
                }
                return value;
            }

            case kMajorSimple: {
                if (argument == kSimpleFalse || argument == kSimpleTrue) {
                    value.type_ = CborType::Bool;
                    value.bool_ = argument == kSimpleTrue;
                    return value;
                }
                // Floats, null and undefined all land here and are all refused:
                // half, single and double can each encode 1.0, NaN carries a
                // payload, and -0.0 is a distinct bit pattern that compares equal
                // to 0.0. Absence is expressed by leaving a key out.
                return Err("cbor: only true and false are permitted in major type 7");
            }

            case kMajorTag:
                // A tag changes what a value means without changing its bytes.
                return Err("cbor: tags are not permitted");

            default:
                return Err("cbor: unknown major type");
        }
    }

    size_t position() const { return pos_; }
    size_t remaining() const { return data_.size() - pos_; }

private:
    struct Head {
        uint8_t major{0};
        uint64_t argument{0};
    };

    Result<Head> ReadHead() {
        if (remaining() < 1) return Err("cbor: truncated at a head byte");
        const uint8_t initial = data_[pos_++];
        const uint8_t major = static_cast<uint8_t>(initial >> 5);
        const uint8_t info = static_cast<uint8_t>(initial & 0x1F);

        uint64_t argument = 0;
        size_t extra = 0;
        if (info < 24) {
            argument = info;
        } else if (info == 24) {
            extra = 1;
        } else if (info == 25) {
            extra = 2;
        } else if (info == 26) {
            extra = 4;
        } else if (info == 27) {
            extra = 8;
        } else if (info == 31) {
            // Indefinite length: a second way to write what a definite length
            // already says, and it lets a sender stream past any bound checked
            // on the head.
            return Err("cbor: indefinite-length items are not permitted");
        } else {
            return Err(std::format("cbor: reserved additional information {}", info));
        }

        if (extra > 0) {
            if (remaining() < extra) return Err("cbor: truncated argument");
            for (size_t i = 0; i < extra; ++i) argument = (argument << 8) | data_[pos_++];
            // 1 must be 0x01, never 0x1801. Accepting both means accepting two
            // byte strings that mean the same thing, and any check made on bytes
            // rather than on meaning can then be walked around.
            if (MinimalHeadLength(argument) != extra + 1) {
                return Err(std::format("cbor: {} is not encoded in its shortest form", argument));
            }
        }
        return Head{major, argument};
    }

    // Text strings must be valid UTF-8, or two implementations will disagree the
    // moment one of them decodes and the other compares bytes.
    static Status ValidateUtf8(const std::vector<uint8_t>& bytes) {
        size_t i = 0;
        while (i < bytes.size()) {
            const uint8_t c = bytes[i];
            size_t length = 0;
            uint32_t code = 0;
            if (c < 0x80) {
                ++i;
                continue;
            } else if ((c & 0xE0) == 0xC0) {
                length = 2;
                code = c & 0x1Fu;
            } else if ((c & 0xF0) == 0xE0) {
                length = 3;
                code = c & 0x0Fu;
            } else if ((c & 0xF8) == 0xF0) {
                length = 4;
                code = c & 0x07u;
            } else {
                return Err("cbor: invalid utf-8 lead byte");
            }
            if (i + length > bytes.size()) return Err("cbor: truncated utf-8 sequence");
            for (size_t k = 1; k < length; ++k) {
                const uint8_t cc = bytes[i + k];
                if ((cc & 0xC0) != 0x80) return Err("cbor: invalid utf-8 continuation");
                code = (code << 6) | (cc & 0x3Fu);
            }
            // Overlong forms are the utf-8 version of a non-minimal integer, and
            // surrogates are not valid scalar values.
            static constexpr uint32_t kMinimum[5] = {0, 0, 0x80, 0x800, 0x10000};
            if (code < kMinimum[length]) return Err("cbor: overlong utf-8 sequence");
            if (code >= 0xD800 && code <= 0xDFFF) return Err("cbor: utf-8 surrogate");
            if (code > 0x10FFFF) return Err("cbor: utf-8 code point out of range");
            i += length;
        }
        return Status::Ok();
    }

    std::span<const uint8_t> data_;
    size_t pos_{0};
};

}  // namespace detail

bool CborKeyLess(std::string_view a, std::string_view b) {
    const auto ea = EncodeTextKey(a);
    const auto eb = EncodeTextKey(b);
    return std::lexicographical_compare(ea.begin(), ea.end(), eb.begin(), eb.end());
}

Result<CborValue> CborParse(std::span<const uint8_t> raw) {
    detail::CborParserImpl parser(raw);
    auto value = parser.ParseValue(0);
    if (!value) return Err(value.error());
    if (parser.remaining() != 0) {
        // Trailing bytes are an error for the same reason they are in the canon
        // container format: an attacker who can append data that some parsers
        // ignore can make two readers see different messages in one frame.
        return Err(std::format("cbor: {} trailing bytes after the value", parser.remaining()));
    }
    return value;
}

// ---------------------------------------------------------------------------
// CborValue accessors
// ---------------------------------------------------------------------------

CborValue CborValue::MakeUint(uint64_t v) {
    CborValue value;
    value.type_ = CborType::Uint;
    value.uint_ = v;
    return value;
}

CborValue CborValue::MakeNint(int64_t v) {
    CborValue value;
    value.type_ = CborType::Nint;
    value.nint_ = v;
    return value;
}

CborValue CborValue::MakeBool(bool v) {
    CborValue value;
    value.type_ = CborType::Bool;
    value.bool_ = v;
    return value;
}

Result<uint64_t> CborValue::AsUint() const {
    if (type_ != CborType::Uint) return Err("cbor: value is not an unsigned integer");
    return uint_;
}

Result<int64_t> CborValue::AsInt() const {
    if (type_ == CborType::Nint) return nint_;
    if (type_ != CborType::Uint) return Err("cbor: value is not an integer");
    if (uint_ > static_cast<uint64_t>(INT64_MAX)) return Err("cbor: integer out of range");
    return static_cast<int64_t>(uint_);
}

Result<std::span<const uint8_t>> CborValue::AsBytes() const {
    if (type_ != CborType::Bytes) return Err("cbor: value is not a byte string");
    return std::span<const uint8_t>(bytes_);
}

Result<std::string_view> CborValue::AsText() const {
    if (type_ != CborType::Text) return Err("cbor: value is not a text string");
    return std::string_view(reinterpret_cast<const char*>(bytes_.data()), bytes_.size());
}

Result<bool> CborValue::AsBool() const {
    if (type_ != CborType::Bool) return Err("cbor: value is not a boolean");
    return bool_;
}

const CborValue* CborValue::Find(std::string_view key) const {
    if (type_ != CborType::Map) return nullptr;
    const auto it = entries_.find(std::string(key));
    return it == entries_.end() ? nullptr : &it->second;
}

Result<const CborValue*> CborValue::Get(std::string_view key) const {
    if (type_ != CborType::Map) return Err("cbor: value is not a map");
    const auto* found = Find(key);
    if (found == nullptr) return Err(std::format("cbor: required field '{}' is missing", key));
    return found;
}

Result<uint64_t> CborValue::GetUint(std::string_view key) const {
    auto field = Get(key);
    if (!field) return Err(field.error());
    auto value = field.value()->AsUint();
    if (!value) return Err(std::format("cbor: field '{}': {}", key, value.error()));
    return value.value();
}

Result<int64_t> CborValue::GetInt(std::string_view key) const {
    auto field = Get(key);
    if (!field) return Err(field.error());
    auto value = field.value()->AsInt();
    if (!value) return Err(std::format("cbor: field '{}': {}", key, value.error()));
    return value.value();
}

Result<std::vector<uint8_t>> CborValue::GetBytes(std::string_view key) const {
    auto field = Get(key);
    if (!field) return Err(field.error());
    auto value = field.value()->AsBytes();
    if (!value) return Err(std::format("cbor: field '{}': {}", key, value.error()));
    return std::vector<uint8_t>(value.value().begin(), value.value().end());
}

Result<std::array<uint8_t, 32>> CborValue::GetHash(std::string_view key) const {
    auto raw = GetBytes(key);
    if (!raw) return Err(raw.error());
    if (raw.value().size() != 32) {
        // A hash of the wrong length is not a hash. Padding or truncating one
        // would turn a malformed message into a valid-looking identity.
        return Err(std::format("cbor: field '{}' is {} bytes, a hash is 32", key,
                               raw.value().size()));
    }
    std::array<uint8_t, 32> out{};
    std::copy(raw.value().begin(), raw.value().end(), out.begin());
    return out;
}

Result<std::string> CborValue::GetText(std::string_view key) const {
    auto field = Get(key);
    if (!field) return Err(field.error());
    auto value = field.value()->AsText();
    if (!value) return Err(std::format("cbor: field '{}': {}", key, value.error()));
    return std::string(value.value());
}

Result<bool> CborValue::GetBool(std::string_view key) const {
    auto field = Get(key);
    if (!field) return Err(field.error());
    auto value = field.value()->AsBool();
    if (!value) return Err(std::format("cbor: field '{}': {}", key, value.error()));
    return value.value();
}

// ---------------------------------------------------------------------------
// CborWriter
// ---------------------------------------------------------------------------

void CborWriter::Fail(std::string reason) {
    if (error_.empty()) error_ = std::move(reason);
}

void CborWriter::Head(uint8_t major, uint64_t value) {
    const uint8_t prefix = static_cast<uint8_t>(major << 5);
    if (value < 24) {
        buf_.push_back(static_cast<uint8_t>(prefix | value));
    } else if (value <= 0xFF) {
        buf_.push_back(static_cast<uint8_t>(prefix | 24));
        buf_.push_back(static_cast<uint8_t>(value));
    } else if (value <= 0xFFFF) {
        buf_.push_back(static_cast<uint8_t>(prefix | 25));
        for (int shift = 8; shift >= 0; shift -= 8) {
            buf_.push_back(static_cast<uint8_t>((value >> shift) & 0xFF));
        }
    } else if (value <= 0xFFFFFFFF) {
        buf_.push_back(static_cast<uint8_t>(prefix | 26));
        for (int shift = 24; shift >= 0; shift -= 8) {
            buf_.push_back(static_cast<uint8_t>((value >> shift) & 0xFF));
        }
    } else {
        buf_.push_back(static_cast<uint8_t>(prefix | 27));
        for (int shift = 56; shift >= 0; shift -= 8) {
            buf_.push_back(static_cast<uint8_t>((value >> shift) & 0xFF));
        }
    }
}

// Accounts for one written item against the enclosing container, so a declared
// count that does not match what follows is caught here rather than by whoever
// reads the frame.
void CborWriter::CountItem() {
    if (stack_.empty()) return;
    auto& frame = stack_.back();
    if (frame.is_map && frame.expecting_key) {
        Fail("cbor: a map key must be written with Key()");
        return;
    }
    if (frame.remaining == 0) {
        Fail("cbor: more items written than the container declared");
        return;
    }
    --frame.remaining;
    if (frame.is_map) frame.expecting_key = true;
    if (frame.remaining == 0) stack_.pop_back();
}

// A frame carries exactly one top-level value. Writing a second would produce two
// concatenated documents in one frame — and a reader that stops after the first,
// as ours does, would silently ignore whatever followed.
bool CborWriter::BeginItem() {
    if (!error_.empty()) return false;
    if (stack_.empty() && !buf_.empty()) {
        Fail("cbor: a frame holds one value; a second was written after it");
        return false;
    }
    return true;
}

CborWriter& CborWriter::Uint(uint64_t v) {
    if (!BeginItem()) return *this;
    Head(kMajorUint, v);
    CountItem();
    return *this;
}

CborWriter& CborWriter::Int(int64_t v) {
    if (!BeginItem()) return *this;
    if (v < 0) {
        Head(kMajorNint, static_cast<uint64_t>(-1 - v));
    } else {
        Head(kMajorUint, static_cast<uint64_t>(v));
    }
    CountItem();
    return *this;
}

CborWriter& CborWriter::Bytes(std::span<const uint8_t> data) {
    if (!BeginItem()) return *this;
    Head(kMajorBytes, data.size());
    buf_.insert(buf_.end(), data.begin(), data.end());
    CountItem();
    return *this;
}

CborWriter& CborWriter::Text(std::string_view text) {
    if (!BeginItem()) return *this;
    Head(kMajorText, text.size());
    buf_.insert(buf_.end(), text.begin(), text.end());
    CountItem();
    return *this;
}

CborWriter& CborWriter::Bool(bool v) {
    if (!BeginItem()) return *this;
    Head(kMajorSimple, v ? kSimpleTrue : kSimpleFalse);
    CountItem();
    return *this;
}

CborWriter& CborWriter::BeginArray(uint32_t count) {
    if (!BeginItem()) return *this;
    if (count > kMaxCborItems) {
        Fail("cbor: array exceeds the item limit");
        return *this;
    }
    Head(kMajorArray, count);
    CountItem();
    if (count > 0) stack_.push_back(Frame{false, count, false, {}});
    return *this;
}

CborWriter& CborWriter::BeginMap(uint32_t count) {
    if (!BeginItem()) return *this;
    if (count > kMaxCborItems) {
        Fail("cbor: map exceeds the item limit");
        return *this;
    }
    Head(kMajorMap, count);
    CountItem();
    // A map of n pairs is 2n items; the frame counts both halves so a missing
    // value is as loud as a missing key.
    if (count > 0) stack_.push_back(Frame{true, count * 2, true, {}});
    return *this;
}

CborWriter& CborWriter::Key(std::string_view key) {
    if (!error_.empty()) return *this;
    if (stack_.empty() || !stack_.back().is_map) {
        Fail("cbor: Key() outside a map");
        return *this;
    }
    auto& frame = stack_.back();
    if (!frame.expecting_key) {
        Fail(std::format("cbor: key '{}' where a value was expected", key));
        return *this;
    }

    // Sorted by encoded bytes, so the order is a property of the keys and not of
    // whoever happened to write them.
    auto encoded = EncodeTextKey(key);
    if (!frame.last_key.empty()) {
        if (encoded == frame.last_key) {
            Fail(std::format("cbor: duplicate map key '{}'", key));
            return *this;
        }
        if (!std::lexicographical_compare(frame.last_key.begin(), frame.last_key.end(),
                                          encoded.begin(), encoded.end())) {
            Fail(std::format("cbor: key '{}' is out of canonical order", key));
            return *this;
        }
    }
    frame.last_key = encoded;

    buf_.insert(buf_.end(), encoded.begin(), encoded.end());
    if (frame.remaining == 0) {
        Fail("cbor: more items written than the map declared");
        return *this;
    }
    --frame.remaining;
    frame.expecting_key = false;
    return *this;
}

Result<std::vector<uint8_t>> CborWriter::Take() {
    if (!error_.empty()) return Err(error_);
    if (!stack_.empty()) {
        return Err(std::format("cbor: {} container(s) left unfinished", stack_.size()));
    }
    if (buf_.empty()) return Err("cbor: nothing was written");
    return std::move(buf_);
}

}  // namespace rnet::ipc
