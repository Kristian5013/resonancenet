// Copyright (c) 2024-present ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "rpc/request.h"

#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <stdexcept>

namespace rnet::rpc {

// ===========================================================================
//  Static Sentinels
// ===========================================================================

// ---------------------------------------------------------------------------
// NULL_VALUE — default-constructed JsonValue returned for missing keys
//
// Design: operator[] on objects/arrays must return a const reference even when
// the key is absent.  A file-scope static null avoids dangling-reference UB
// and lets callers chain lookups without checking existence first.
// ---------------------------------------------------------------------------

static const JsonValue NULL_VALUE;

// ===========================================================================
//  JsonValue Constructors
// ===========================================================================

// ---------------------------------------------------------------------------
// JsonValue() — default constructor, produces JSON null
//
// Design: JSON's type system (RFC 8259) defines seven value kinds: null, bool,
// integer (stored as int64_t), double, string, array, and object.  Each
// constructor tags the discriminant and initialises exactly one payload field.
// ---------------------------------------------------------------------------

JsonValue::JsonValue() : type_(JSON_NULL) {}

// ---------------------------------------------------------------------------
// JsonValue(bool) — boolean true / false
//
// Design: explicit bool overload prevents implicit conversions from pointers
// or integers from silently creating a boolean value.
// ---------------------------------------------------------------------------

JsonValue::JsonValue(bool v) : type_(JSON_BOOL), bool_val_(v) {}

// ---------------------------------------------------------------------------
// JsonValue(int) — convenience for integer literals without suffix
//
// Design: widens to int64_t internally so the type system stays uniform.
// ---------------------------------------------------------------------------

JsonValue::JsonValue(int v) : type_(JSON_INT), int_val_(v) {}

// ---------------------------------------------------------------------------
// JsonValue(int64_t) — signed 64-bit integer
//
// Design: covers the full range of Bitcoin-style amounts and block heights.
// ---------------------------------------------------------------------------

JsonValue::JsonValue(int64_t v) : type_(JSON_INT), int_val_(v) {}

// ---------------------------------------------------------------------------
// JsonValue(uint64_t) — unsigned integer cast to int64_t
//
// Design: static_cast preserves bit pattern for values <= INT64_MAX.  Values
// above that wrap, but JSON has no unsigned type — callers must be aware.
// ---------------------------------------------------------------------------

JsonValue::JsonValue(uint64_t v)
    : type_(JSON_INT), int_val_(static_cast<int64_t>(v)) {}

// ---------------------------------------------------------------------------
// JsonValue(double) — IEEE 754 floating-point number
//
// Design: NaN and Inf are not valid JSON; serialisation replaces them with
// null (handled in to_string_impl).
// ---------------------------------------------------------------------------

JsonValue::JsonValue(double v) : type_(JSON_DOUBLE), double_val_(v) {}

// ---------------------------------------------------------------------------
// JsonValue(const char*) — string from C literal
//
// Design: implicit conversion from string literals avoids verbose wrapping at
// every RPC call site.
// ---------------------------------------------------------------------------

JsonValue::JsonValue(const char* v)
    : type_(JSON_STRING), string_val_(v) {}

// ---------------------------------------------------------------------------
// JsonValue(std::string) — string by value, moved into storage
//
// Design: move semantics avoid a copy when the caller already owns a
// temporary string (e.g. from serialisation helpers).
// ---------------------------------------------------------------------------

JsonValue::JsonValue(std::string v)
    : type_(JSON_STRING), string_val_(std::move(v)) {}

// ---------------------------------------------------------------------------
// JsonValue(std::string_view) — string from a non-owning view
//
// Design: copies the view contents into an owning std::string to guarantee
// lifetime independence from the source buffer.
// ---------------------------------------------------------------------------

JsonValue::JsonValue(std::string_view v)
    : type_(JSON_STRING), string_val_(v) {}

// ---------------------------------------------------------------------------
// JsonValue(Array) — array of JsonValues
//
// Design: Array is std::vector<JsonValue>.  Moved in to avoid deep copies of
// potentially large result sets (e.g. listtransactions output).
// ---------------------------------------------------------------------------

JsonValue::JsonValue(Array v)
    : type_(JSON_ARRAY), array_val_(std::move(v)) {}

// ---------------------------------------------------------------------------
// JsonValue(Object) — ordered map of string keys to JsonValues
//
// Design: Object is std::map<std::string, JsonValue>.  Ordered map ensures
// deterministic serialisation order, which simplifies test comparisons.
// ---------------------------------------------------------------------------

JsonValue::JsonValue(Object v)
    : type_(JSON_OBJECT), object_val_(std::move(v)) {}

// ===========================================================================
//  JsonValue Factory Methods
// ===========================================================================

// ---------------------------------------------------------------------------
// null_value / boolean / integer / number / string / array / object
//
// Design: named factories provide self-documenting call sites and avoid
// ambiguity when constructing from literals (e.g. JsonValue::integer(42) vs
// JsonValue(42) which could match int or bool on some compilers).
// ---------------------------------------------------------------------------

JsonValue JsonValue::null_value()       { return JsonValue(); }
JsonValue JsonValue::boolean(bool v)    { return JsonValue(v); }
JsonValue JsonValue::integer(int64_t v) { return JsonValue(v); }
JsonValue JsonValue::number(double v)   { return JsonValue(v); }
JsonValue JsonValue::string(std::string v) { return JsonValue(std::move(v)); }
JsonValue JsonValue::array(Array v)     { return JsonValue(std::move(v)); }
JsonValue JsonValue::object(Object v)   { return JsonValue(std::move(v)); }

// ===========================================================================
//  JsonValue Accessors
// ===========================================================================

// ---------------------------------------------------------------------------
// as_bool / as_int / as_double / as_string / as_array / as_object
//
// Design: thin accessors that return the stored payload directly.  No type
// checking — callers must verify type() first.  as_double() auto-promotes
// integers so callers can treat numeric fields uniformly.
// ---------------------------------------------------------------------------

bool        JsonValue::as_bool()   const { return bool_val_; }
int64_t     JsonValue::as_int()    const { return int_val_; }
double      JsonValue::as_double() const {
    if (type_ == JSON_INT) return static_cast<double>(int_val_);
    return double_val_;
}
const std::string&         JsonValue::as_string() const { return string_val_; }
const JsonValue::Array&    JsonValue::as_array()  const { return array_val_; }
const JsonValue::Object&   JsonValue::as_object() const { return object_val_; }
JsonValue::Array&          JsonValue::as_array()        { return array_val_; }
JsonValue::Object&         JsonValue::as_object()       { return object_val_; }

// ---------------------------------------------------------------------------
// operator[](const std::string&) — object key lookup
//
// Design: returns a const reference to the value for the given key, or
// NULL_VALUE if the key is absent or the value is not an object.  This lets
// callers chain lookups like json["result"]["hash"] without null checks.
// ---------------------------------------------------------------------------

const JsonValue& JsonValue::operator[](const std::string& key) const {
    // 1. Reject non-object types
    if (type_ != JSON_OBJECT) return NULL_VALUE;
    // 2. Search for key in the ordered map
    auto it = object_val_.find(key);
    if (it == object_val_.end()) return NULL_VALUE;
    // 3. Return the found value
    return it->second;
}

// ---------------------------------------------------------------------------
// operator[](size_t) — array index lookup
//
// Design: bounds-checked access returning NULL_VALUE on out-of-range rather
// than throwing, consistent with the object overload's fail-soft behaviour.
// ---------------------------------------------------------------------------

const JsonValue& JsonValue::operator[](size_t index) const {
    // 1. Reject non-array types and out-of-bounds indices
    if (type_ != JSON_ARRAY || index >= array_val_.size()) return NULL_VALUE;
    // 2. Return the element at the given index
    return array_val_[index];
}

// ---------------------------------------------------------------------------
// has_key — check whether an object contains the given key
//
// Design: returns false for non-object types rather than throwing, so callers
// can safely probe without first checking type().
// ---------------------------------------------------------------------------

bool JsonValue::has_key(const std::string& key) const {
    if (type_ != JSON_OBJECT) return false;
    return object_val_.find(key) != object_val_.end();
}

// ---------------------------------------------------------------------------
// set — insert or overwrite an object key
//
// Design: auto-promotes the value to JSON_OBJECT if it is currently a
// different type.  This simplifies builder patterns where the caller
// constructs an object incrementally without pre-declaring the type.
// ---------------------------------------------------------------------------

void JsonValue::set(const std::string& key, JsonValue val) {
    // 1. Auto-promote to object if necessary
    if (type_ != JSON_OBJECT) {
        type_ = JSON_OBJECT;
        object_val_.clear();
    }
    // 2. Insert or overwrite the key
    object_val_[key] = std::move(val);
}

// ---------------------------------------------------------------------------
// push_back — append an element to an array
//
// Design: auto-promotes to JSON_ARRAY, mirroring set()'s auto-promote for
// objects.  Enables incremental array construction.
// ---------------------------------------------------------------------------

void JsonValue::push_back(JsonValue val) {
    // 1. Auto-promote to array if necessary
    if (type_ != JSON_ARRAY) {
        type_ = JSON_ARRAY;
        array_val_.clear();
    }
    // 2. Append the element
    array_val_.push_back(std::move(val));
}

// ---------------------------------------------------------------------------
// size — element count for arrays and objects
//
// Design: returns 0 for scalar types, allowing generic size checks without
// first branching on type().
// ---------------------------------------------------------------------------

size_t JsonValue::size() const {
    if (type_ == JSON_ARRAY) return array_val_.size();
    if (type_ == JSON_OBJECT) return object_val_.size();
    return 0;
}

// ===========================================================================
//  JSON Serialization
// ===========================================================================

// ---------------------------------------------------------------------------
// escape_string — produce a quoted JSON string with RFC 8259 escapes
//
// Design: handles the seven mandatory escape sequences (\" \\ \b \f \n \r \t)
// plus \u00xx for control characters below 0x20.  Pre-reserves output to
// avoid repeated reallocation on typical short strings.
// ---------------------------------------------------------------------------

std::string JsonValue::escape_string(const std::string& s) {
    // 1. Reserve space for the string plus surrounding quotes
    std::string out;
    out.reserve(s.size() + 2);
    out.push_back('"');
    // 2. Translate each character through the escape table
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    // 3. Encode control characters as \u00xx
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x",
                                  static_cast<unsigned>(static_cast<unsigned char>(c)));
                    out += buf;
                } else {
                    out.push_back(c);
                }
                break;
        }
    }
    out.push_back('"');
    return out;
}

// ---------------------------------------------------------------------------
// to_string_impl — recursive compact serialisation into an output buffer
//
// Design: switch-dispatches on type_ to emit RFC 8259 JSON.  Integers use
// std::to_chars for locale-independent formatting.  Doubles use %.17g for
// full round-trip precision.  NaN/Inf are replaced with null per convention.
// Arrays and objects recurse into child elements with comma separators.
// ---------------------------------------------------------------------------

void JsonValue::to_string_impl(std::string& out) const {
    switch (type_) {
        case JSON_NULL:
            // 1. Null literal
            out += "null";
            break;
        case JSON_BOOL:
            // 2. Boolean literal
            out += bool_val_ ? "true" : "false";
            break;
        case JSON_INT: {
            // 3. Integer via to_chars (locale-independent)
            char buf[32];
            auto [ptr, ec] = std::to_chars(buf, buf + sizeof(buf), int_val_);
            out.append(buf, ptr);
            break;
        }
        case JSON_DOUBLE: {
            // 4. Double with NaN/Inf guard
            if (std::isnan(double_val_) || std::isinf(double_val_)) {
                out += "null";
            } else {
                char buf[64];
                std::snprintf(buf, sizeof(buf), "%.17g", double_val_);
                out += buf;
            }
            break;
        }
        case JSON_STRING:
            // 5. Escaped string
            out += escape_string(string_val_);
            break;
        case JSON_ARRAY:
            // 6. Array: comma-separated elements in brackets
            out.push_back('[');
            for (size_t i = 0; i < array_val_.size(); ++i) {
                if (i > 0) out.push_back(',');
                array_val_[i].to_string_impl(out);
            }
            out.push_back(']');
            break;
        case JSON_OBJECT:
            // 7. Object: comma-separated key:value pairs in braces
            out.push_back('{');
            {
                bool first = true;
                for (const auto& [k, v] : object_val_) {
                    if (!first) out.push_back(',');
                    first = false;
                    out += escape_string(k);
                    out.push_back(':');
                    v.to_string_impl(out);
                }
            }
            out.push_back('}');
            break;
    }
}

// ---------------------------------------------------------------------------
// to_string — produce a compact JSON string
//
// Design: pre-reserves 256 bytes to cover typical RPC responses in a single
// allocation, then delegates to to_string_impl for recursive output.
// ---------------------------------------------------------------------------

std::string JsonValue::to_string() const {
    std::string out;
    out.reserve(256);
    to_string_impl(out);
    return out;
}

// ---------------------------------------------------------------------------
// to_string_pretty_impl — recursive pretty-printed serialisation
//
// Design: adds newlines and indentation for human readability.  Only arrays
// and objects produce multi-line output; scalars delegate to the compact path.
// Empty containers emit [] or {} on a single line.
// ---------------------------------------------------------------------------

void JsonValue::to_string_pretty_impl(std::string& out, int indent, int depth) const {
    // 1. Build indentation strings for current and inner levels
    std::string pad(static_cast<size_t>(indent * depth), ' ');
    std::string pad_inner(static_cast<size_t>(indent * (depth + 1)), ' ');

    switch (type_) {
        case JSON_ARRAY:
            // 2. Empty array on one line
            if (array_val_.empty()) {
                out += "[]";
                return;
            }
            // 3. Multi-line array with indented elements
            out += "[\n";
            for (size_t i = 0; i < array_val_.size(); ++i) {
                out += pad_inner;
                array_val_[i].to_string_pretty_impl(out, indent, depth + 1);
                if (i + 1 < array_val_.size()) out += ",";
                out += "\n";
            }
            out += pad + "]";
            break;
        case JSON_OBJECT:
            // 4. Empty object on one line
            if (object_val_.empty()) {
                out += "{}";
                return;
            }
            // 5. Multi-line object with indented key-value pairs
            out += "{\n";
            {
                size_t count = 0;
                for (const auto& [k, v] : object_val_) {
                    out += pad_inner;
                    out += escape_string(k);
                    out += ": ";
                    v.to_string_pretty_impl(out, indent, depth + 1);
                    if (++count < object_val_.size()) out += ",";
                    out += "\n";
                }
            }
            out += pad + "}";
            break;
        default:
            // 6. Scalars use compact representation
            to_string_impl(out);
            break;
    }
}

// ---------------------------------------------------------------------------
// to_string_pretty — produce a human-readable JSON string
//
// Design: entry point for pretty printing, pre-reserves 512 bytes for typical
// RPC output.  The indent parameter controls spaces per nesting level.
// ---------------------------------------------------------------------------

std::string JsonValue::to_string_pretty(int indent) const {
    std::string out;
    out.reserve(512);
    to_string_pretty_impl(out, indent, 0);
    return out;
}

// ===========================================================================
//  JSON Parsing
// ===========================================================================

// ---------------------------------------------------------------------------
// JsonParser — recursive descent parser for RFC 8259 JSON
//
// Design: single-pass parser operating on a string_view (zero-copy input).
// Tracks position with a simple index.  Each parse_* method corresponds to
// one JSON grammar production: value, null, bool, number, string, array,
// object.  Returns false on malformed input rather than throwing.
//
// Number parsing: integers use std::from_chars; doubles fall back to strtod
// because Apple libc++ lacks from_chars<double>.
//
// String parsing: handles all RFC 8259 escape sequences including \uXXXX
// with UTF-8 encoding for BMP codepoints.
// ---------------------------------------------------------------------------

namespace {

class JsonParser {
public:
    explicit JsonParser(std::string_view input)
        : input_(input), pos_(0) {}

    bool parse(JsonValue& out) {
        // 1. Skip leading whitespace
        skip_ws();
        // 2. Parse the root value
        if (!parse_value(out)) return false;
        // 3. Skip trailing whitespace
        skip_ws();
        return true;
    }

private:
    std::string_view input_;
    size_t pos_;

    char peek() const {
        if (pos_ >= input_.size()) return '\0';
        return input_[pos_];
    }

    char next() {
        if (pos_ >= input_.size()) return '\0';
        return input_[pos_++];
    }

    bool eof() const { return pos_ >= input_.size(); }

    void skip_ws() {
        while (pos_ < input_.size()) {
            char c = input_[pos_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                ++pos_;
            } else {
                break;
            }
        }
    }

    bool expect(char c) {
        skip_ws();
        if (peek() == c) { ++pos_; return true; }
        return false;
    }

    bool match(std::string_view literal) {
        if (pos_ + literal.size() > input_.size()) return false;
        if (input_.substr(pos_, literal.size()) == literal) {
            pos_ += literal.size();
            return true;
        }
        return false;
    }

    // -----------------------------------------------------------------------
    // parse_value — dispatch to type-specific parser based on first character
    // -----------------------------------------------------------------------

    bool parse_value(JsonValue& out) {
        // 1. Skip whitespace before the value
        skip_ws();
        // 2. Dispatch based on the leading character
        char c = peek();
        if (c == '"') return parse_string_value(out);
        if (c == '{') return parse_object(out);
        if (c == '[') return parse_array(out);
        if (c == 't' || c == 'f') return parse_bool(out);
        if (c == 'n') return parse_null(out);
        if (c == '-' || (c >= '0' && c <= '9')) return parse_number(out);
        return false;
    }

    // -----------------------------------------------------------------------
    // parse_null — match the literal "null"
    // -----------------------------------------------------------------------

    bool parse_null(JsonValue& out) {
        if (match("null")) { out = JsonValue(); return true; }
        return false;
    }

    // -----------------------------------------------------------------------
    // parse_bool — match "true" or "false" literals
    // -----------------------------------------------------------------------

    bool parse_bool(JsonValue& out) {
        if (match("true"))  { out = JsonValue(true);  return true; }
        if (match("false")) { out = JsonValue(false); return true; }
        return false;
    }

    // -----------------------------------------------------------------------
    // parse_number — integer or floating-point number
    //
    // Design: scans the number token to determine if it contains a decimal
    // point or exponent (making it a float).  Integers use std::from_chars
    // for exact parsing.  Doubles use strtod for portability (Apple libc++
    // lacks from_chars for floating-point types).
    // -----------------------------------------------------------------------

    bool parse_number(JsonValue& out) {
        // 1. Record start position and track whether this is a float
        size_t start = pos_;
        bool is_float = false;

        // 2. Optional leading minus sign
        if (peek() == '-') ++pos_;

        // 3. Integer part: leading zero or 1-9 followed by digits
        if (peek() == '0') {
            ++pos_;
        } else if (peek() >= '1' && peek() <= '9') {
            while (peek() >= '0' && peek() <= '9') ++pos_;
        } else {
            return false;
        }

        // 4. Optional fractional part
        if (peek() == '.') {
            is_float = true;
            ++pos_;
            if (!(peek() >= '0' && peek() <= '9')) return false;
            while (peek() >= '0' && peek() <= '9') ++pos_;
        }

        // 5. Optional exponent part
        if (peek() == 'e' || peek() == 'E') {
            is_float = true;
            ++pos_;
            if (peek() == '+' || peek() == '-') ++pos_;
            if (!(peek() >= '0' && peek() <= '9')) return false;
            while (peek() >= '0' && peek() <= '9') ++pos_;
        }

        // 6. Extract the number substring
        std::string_view num_str = input_.substr(start, pos_ - start);

        // 7. Convert to the appropriate numeric type
        if (is_float) {
            std::string tmp(num_str);
            char* end = nullptr;
            double val = std::strtod(tmp.c_str(), &end);
            if (end != tmp.c_str() + tmp.size()) return false;
            out = JsonValue(val);
        } else {
            int64_t val = 0;
            auto [p, ec] = std::from_chars(num_str.data(),
                                           num_str.data() + num_str.size(),
                                           val);
            if (ec != std::errc{}) return false;
            out = JsonValue(val);
        }
        return true;
    }

    // -----------------------------------------------------------------------
    // parse_raw_string — parse a JSON string into a std::string
    //
    // Design: consumes the opening quote, then processes characters until the
    // closing quote.  Backslash escapes are translated per RFC 8259.  The
    // \uXXXX form reads four hex digits and encodes the codepoint as UTF-8
    // (1-3 bytes for BMP characters).
    // -----------------------------------------------------------------------

    bool parse_raw_string(std::string& result) {
        // 1. Consume opening quote
        if (next() != '"') return false;

        result.clear();
        while (!eof()) {
            char c = next();
            // 2. Closing quote ends the string
            if (c == '"') return true;
            // 3. Handle escape sequences
            if (c == '\\') {
                char esc = next();
                switch (esc) {
                    case '"':  result.push_back('"'); break;
                    case '\\': result.push_back('\\'); break;
                    case '/':  result.push_back('/'); break;
                    case 'b':  result.push_back('\b'); break;
                    case 'f':  result.push_back('\f'); break;
                    case 'n':  result.push_back('\n'); break;
                    case 'r':  result.push_back('\r'); break;
                    case 't':  result.push_back('\t'); break;
                    case 'u': {
                        // 4. Parse 4 hex digits for \uXXXX escape
                        if (pos_ + 4 > input_.size()) return false;
                        uint32_t cp = 0;
                        for (int i = 0; i < 4; ++i) {
                            char h = next();
                            cp <<= 4;
                            if (h >= '0' && h <= '9') cp |= (h - '0');
                            else if (h >= 'a' && h <= 'f') cp |= (h - 'a' + 10);
                            else if (h >= 'A' && h <= 'F') cp |= (h - 'A' + 10);
                            else return false;
                        }
                        // 5. Encode codepoint as UTF-8 (BMP only)
                        if (cp < 0x80) {
                            result.push_back(static_cast<char>(cp));
                        } else if (cp < 0x800) {
                            result.push_back(static_cast<char>(0xC0 | (cp >> 6)));
                            result.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                        } else {
                            result.push_back(static_cast<char>(0xE0 | (cp >> 12)));
                            result.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                            result.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                        }
                        break;
                    }
                    default:
                        return false;
                }
            } else {
                // 6. Regular character, append directly
                result.push_back(c);
            }
        }
        return false;  // unterminated string
    }

    // -----------------------------------------------------------------------
    // parse_string_value — parse a JSON string into a JsonValue
    // -----------------------------------------------------------------------

    bool parse_string_value(JsonValue& out) {
        std::string s;
        if (!parse_raw_string(s)) return false;
        out = JsonValue(std::move(s));
        return true;
    }

    // -----------------------------------------------------------------------
    // parse_array — parse a JSON array [elem, elem, ...]
    //
    // Design: handles the empty array special case, then loops consuming
    // comma-separated values until the closing bracket.  Returns false on
    // missing commas or brackets (strict parsing, no trailing comma).
    // -----------------------------------------------------------------------

    bool parse_array(JsonValue& out) {
        // 1. Consume opening bracket
        if (next() != '[') return false;
        JsonValue::Array arr;
        // 2. Handle empty array
        skip_ws();
        if (peek() == ']') { ++pos_; out = JsonValue(std::move(arr)); return true; }

        // 3. Parse comma-separated elements
        while (true) {
            JsonValue elem;
            if (!parse_value(elem)) return false;
            arr.push_back(std::move(elem));
            skip_ws();
            if (peek() == ',') { ++pos_; skip_ws(); continue; }
            if (peek() == ']') { ++pos_; break; }
            return false;
        }
        out = JsonValue(std::move(arr));
        return true;
    }

    // -----------------------------------------------------------------------
    // parse_object — parse a JSON object {"key": value, ...}
    //
    // Design: similar structure to parse_array but expects string keys
    // followed by colons.  Keys are parsed via parse_raw_string to reuse
    // the escape-handling logic.  Duplicate keys overwrite silently (last
    // value wins), consistent with most JSON implementations.
    // -----------------------------------------------------------------------

    bool parse_object(JsonValue& out) {
        // 1. Consume opening brace
        if (next() != '{') return false;
        JsonValue::Object obj;
        // 2. Handle empty object
        skip_ws();
        if (peek() == '}') { ++pos_; out = JsonValue(std::move(obj)); return true; }

        // 3. Parse comma-separated key-value pairs
        while (true) {
            skip_ws();
            // 4. Parse the string key
            std::string key;
            if (!parse_raw_string(key)) return false;
            // 5. Expect colon separator
            skip_ws();
            if (next() != ':') return false;
            // 6. Parse the value
            skip_ws();
            JsonValue val;
            if (!parse_value(val)) return false;
            obj[std::move(key)] = std::move(val);
            // 7. Expect comma or closing brace
            skip_ws();
            if (peek() == ',') { ++pos_; continue; }
            if (peek() == '}') { ++pos_; break; }
            return false;
        }
        out = JsonValue(std::move(obj));
        return true;
    }
};

} // anonymous namespace

// ---------------------------------------------------------------------------
// parse_json(string_view) — parse JSON text, return null on failure
//
// Design: convenience overload that returns a default-constructed (null)
// JsonValue on parse failure, allowing one-liner usage without error checks
// when the caller can tolerate null as a sentinel.
// ---------------------------------------------------------------------------

JsonValue parse_json(std::string_view input) {
    // 1. Attempt to parse the input
    JsonValue result;
    JsonParser parser(input);
    // 2. Return null on failure
    if (!parser.parse(result)) return JsonValue();
    return result;
}

// ---------------------------------------------------------------------------
// parse_json(string_view, JsonValue&) — parse JSON text with error indication
//
// Design: returns a bool so callers can distinguish "parsed null" from
// "parse failed".  The output parameter is only valid when true is returned.
// ---------------------------------------------------------------------------

bool parse_json(std::string_view input, JsonValue& out) {
    JsonParser parser(input);
    return parser.parse(out);
}

// ===========================================================================
//  RPCRequest Handling
// ===========================================================================

// ---------------------------------------------------------------------------
// RPCRequest::from_json — parse a JSON-RPC 2.0 request object
//
// Design: extracts "method" (required string), "params" (optional, defaults
// to empty array), and "id" (optional, used for response correlation).
// Returns false if the input is not an object or lacks a string "method"
// field.  Does not enforce "jsonrpc":"2.0" to stay compatible with JSON-RPC
// 1.0 clients.
// ---------------------------------------------------------------------------

bool RPCRequest::from_json(const JsonValue& json, RPCRequest& out) {
    // 1. Verify the input is a JSON object
    if (!json.is_object()) return false;

    // 2. Extract the required "method" string
    const auto& method = json["method"];
    if (!method.is_string()) return false;
    out.method = method.as_string();

    // 3. Extract optional "params" (default to empty array)
    out.params = json["params"];
    if (out.params.is_null()) {
        out.params = JsonValue::array();
    }

    // 4. Extract optional "id" for response correlation
    out.id = json["id"];

    return true;
}

// ---------------------------------------------------------------------------
// RPCResponse::to_json — serialise a JSON-RPC 2.0 response
//
// Design: builds a {"result":..., "error":..., "id":...} object.  Both
// result and error are always present (one is null) per JSON-RPC convention.
// ---------------------------------------------------------------------------

std::string RPCResponse::to_json() const {
    // 1. Build the response object
    JsonValue obj = JsonValue::object();
    obj.set("result", result);
    obj.set("error", error);
    obj.set("id", id);
    // 2. Serialise to compact JSON
    return obj.to_string();
}

// ---------------------------------------------------------------------------
// RPCResponse::success — construct a successful response
//
// Design: sets error to null and attaches the given result and id.  The id
// must match the request's id for the client to correlate the response.
// ---------------------------------------------------------------------------

RPCResponse RPCResponse::success(JsonValue result_val, JsonValue id_val) {
    RPCResponse resp;
    resp.result = std::move(result_val);
    resp.error = JsonValue();  // null
    resp.id = std::move(id_val);
    return resp;
}

// ---------------------------------------------------------------------------
// RPCResponse::make_error — construct an error response
//
// Design: builds a {"code":N, "message":"..."} error object per JSON-RPC 2.0
// spec.  Standard error codes: -32700 (parse error), -32600 (invalid request),
// -32601 (method not found), -32602 (invalid params), -32603 (internal error).
// ---------------------------------------------------------------------------

RPCResponse RPCResponse::make_error(int code, const std::string& message,
                                    JsonValue id_val) {
    RPCResponse resp;
    resp.result = JsonValue();  // null

    // 1. Build the error object with code and message
    JsonValue err_obj = JsonValue::object();
    err_obj.set("code", JsonValue(static_cast<int64_t>(code)));
    err_obj.set("message", JsonValue(message));
    resp.error = std::move(err_obj);
    resp.id = std::move(id_val);
    return resp;
}

} // namespace rnet::rpc
