#include "rpc/request.h"

#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <stdexcept>

namespace rnet::rpc {

// ── Static sentinel for missing keys ────────────────────────────────

static const JsonValue NULL_VALUE;

// ── JsonValue constructors ──────────────────────────────────────────

JsonValue::JsonValue() : type_(JSON_NULL) {}

JsonValue::JsonValue(bool v) : type_(JSON_BOOL), bool_val_(v) {}

JsonValue::JsonValue(int v) : type_(JSON_INT), int_val_(v) {}

JsonValue::JsonValue(int64_t v) : type_(JSON_INT), int_val_(v) {}

JsonValue::JsonValue(uint64_t v)
    : type_(JSON_INT), int_val_(static_cast<int64_t>(v)) {}

JsonValue::JsonValue(double v) : type_(JSON_DOUBLE), double_val_(v) {}

JsonValue::JsonValue(const char* v)
    : type_(JSON_STRING), string_val_(v) {}

JsonValue::JsonValue(std::string v)
    : type_(JSON_STRING), string_val_(std::move(v)) {}

JsonValue::JsonValue(std::string_view v)
    : type_(JSON_STRING), string_val_(v) {}

JsonValue::JsonValue(Array v)
    : type_(JSON_ARRAY), array_val_(std::move(v)) {}

JsonValue::JsonValue(Object v)
    : type_(JSON_OBJECT), object_val_(std::move(v)) {}

JsonValue JsonValue::null_value()       { return JsonValue(); }
JsonValue JsonValue::boolean(bool v)    { return JsonValue(v); }
JsonValue JsonValue::integer(int64_t v) { return JsonValue(v); }
JsonValue JsonValue::number(double v)   { return JsonValue(v); }
JsonValue JsonValue::string(std::string v) { return JsonValue(std::move(v)); }
JsonValue JsonValue::array(Array v)     { return JsonValue(std::move(v)); }
JsonValue JsonValue::object(Object v)   { return JsonValue(std::move(v)); }

// ── Value accessors ─────────────────────────────────────────────────

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

const JsonValue& JsonValue::operator[](const std::string& key) const {
    if (type_ != JSON_OBJECT) return NULL_VALUE;
    auto it = object_val_.find(key);
    if (it == object_val_.end()) return NULL_VALUE;
    return it->second;
}

const JsonValue& JsonValue::operator[](size_t index) const {
    if (type_ != JSON_ARRAY || index >= array_val_.size()) return NULL_VALUE;
    return array_val_[index];
}

bool JsonValue::has_key(const std::string& key) const {
    if (type_ != JSON_OBJECT) return false;
    return object_val_.find(key) != object_val_.end();
}

void JsonValue::set(const std::string& key, JsonValue val) {
    if (type_ != JSON_OBJECT) {
        type_ = JSON_OBJECT;
        object_val_.clear();
    }
    object_val_[key] = std::move(val);
}

void JsonValue::push_back(JsonValue val) {
    if (type_ != JSON_ARRAY) {
        type_ = JSON_ARRAY;
        array_val_.clear();
    }
    array_val_.push_back(std::move(val));
}

size_t JsonValue::size() const {
    if (type_ == JSON_ARRAY) return array_val_.size();
    if (type_ == JSON_OBJECT) return object_val_.size();
    return 0;
}

// ── Serialization ───────────────────────────────────────────────────

std::string JsonValue::escape_string(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    out.push_back('"');
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

void JsonValue::to_string_impl(std::string& out) const {
    switch (type_) {
        case JSON_NULL:
            out += "null";
            break;
        case JSON_BOOL:
            out += bool_val_ ? "true" : "false";
            break;
        case JSON_INT: {
            char buf[32];
            auto [ptr, ec] = std::to_chars(buf, buf + sizeof(buf), int_val_);
            out.append(buf, ptr);
            break;
        }
        case JSON_DOUBLE: {
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
            out += escape_string(string_val_);
            break;
        case JSON_ARRAY:
            out.push_back('[');
            for (size_t i = 0; i < array_val_.size(); ++i) {
                if (i > 0) out.push_back(',');
                array_val_[i].to_string_impl(out);
            }
            out.push_back(']');
            break;
        case JSON_OBJECT:
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

std::string JsonValue::to_string() const {
    std::string out;
    out.reserve(256);
    to_string_impl(out);
    return out;
}

void JsonValue::to_string_pretty_impl(std::string& out, int indent, int depth) const {
    std::string pad(static_cast<size_t>(indent * depth), ' ');
    std::string pad_inner(static_cast<size_t>(indent * (depth + 1)), ' ');

    switch (type_) {
        case JSON_ARRAY:
            if (array_val_.empty()) {
                out += "[]";
                return;
            }
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
            if (object_val_.empty()) {
                out += "{}";
                return;
            }
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
            to_string_impl(out);
            break;
    }
}

std::string JsonValue::to_string_pretty(int indent) const {
    std::string out;
    out.reserve(512);
    to_string_pretty_impl(out, indent, 0);
    return out;
}

// ── JSON Parser ─────────────────────────────────────────────────────

namespace {

class JsonParser {
public:
    explicit JsonParser(std::string_view input)
        : input_(input), pos_(0) {}

    bool parse(JsonValue& out) {
        skip_ws();
        if (!parse_value(out)) return false;
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

    bool parse_value(JsonValue& out) {
        skip_ws();
        char c = peek();
        if (c == '"') return parse_string_value(out);
        if (c == '{') return parse_object(out);
        if (c == '[') return parse_array(out);
        if (c == 't' || c == 'f') return parse_bool(out);
        if (c == 'n') return parse_null(out);
        if (c == '-' || (c >= '0' && c <= '9')) return parse_number(out);
        return false;
    }

    bool parse_null(JsonValue& out) {
        if (match("null")) { out = JsonValue(); return true; }
        return false;
    }

    bool parse_bool(JsonValue& out) {
        if (match("true"))  { out = JsonValue(true);  return true; }
        if (match("false")) { out = JsonValue(false); return true; }
        return false;
    }

    bool parse_number(JsonValue& out) {
        size_t start = pos_;
        bool is_float = false;

        if (peek() == '-') ++pos_;

        if (peek() == '0') {
            ++pos_;
        } else if (peek() >= '1' && peek() <= '9') {
            while (peek() >= '0' && peek() <= '9') ++pos_;
        } else {
            return false;
        }

        if (peek() == '.') {
            is_float = true;
            ++pos_;
            if (!(peek() >= '0' && peek() <= '9')) return false;
            while (peek() >= '0' && peek() <= '9') ++pos_;
        }

        if (peek() == 'e' || peek() == 'E') {
            is_float = true;
            ++pos_;
            if (peek() == '+' || peek() == '-') ++pos_;
            if (!(peek() >= '0' && peek() <= '9')) return false;
            while (peek() >= '0' && peek() <= '9') ++pos_;
        }

        std::string_view num_str = input_.substr(start, pos_ - start);

        if (is_float) {
            // Apple libc++ doesn't support std::from_chars for doubles
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

    bool parse_raw_string(std::string& result) {
        if (next() != '"') return false;

        result.clear();
        while (!eof()) {
            char c = next();
            if (c == '"') return true;
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
                        // Parse 4 hex digits
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
                        // Simple UTF-8 encoding for BMP characters
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
                result.push_back(c);
            }
        }
        return false;  // unterminated string
    }

    bool parse_string_value(JsonValue& out) {
        std::string s;
        if (!parse_raw_string(s)) return false;
        out = JsonValue(std::move(s));
        return true;
    }

    bool parse_array(JsonValue& out) {
        if (next() != '[') return false;
        JsonValue::Array arr;
        skip_ws();
        if (peek() == ']') { ++pos_; out = JsonValue(std::move(arr)); return true; }

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

    bool parse_object(JsonValue& out) {
        if (next() != '{') return false;
        JsonValue::Object obj;
        skip_ws();
        if (peek() == '}') { ++pos_; out = JsonValue(std::move(obj)); return true; }

        while (true) {
            skip_ws();
            std::string key;
            if (!parse_raw_string(key)) return false;
            skip_ws();
            if (next() != ':') return false;
            skip_ws();
            JsonValue val;
            if (!parse_value(val)) return false;
            obj[std::move(key)] = std::move(val);
            skip_ws();
            if (peek() == ',') { ++pos_; continue; }
            if (peek() == '}') { ++pos_; break; }
            return false;
        }
        out = JsonValue(std::move(obj));
        return true;
    }
};

}  // anonymous namespace

JsonValue parse_json(std::string_view input) {
    JsonValue result;
    JsonParser parser(input);
    if (!parser.parse(result)) return JsonValue();
    return result;
}

bool parse_json(std::string_view input, JsonValue& out) {
    JsonParser parser(input);
    return parser.parse(out);
}

// ── RPCRequest ──────────────────────────────────────────────────────

bool RPCRequest::from_json(const JsonValue& json, RPCRequest& out) {
    if (!json.is_object()) return false;

    const auto& method = json["method"];
    if (!method.is_string()) return false;
    out.method = method.as_string();

    out.params = json["params"];
    if (out.params.is_null()) {
        out.params = JsonValue::array();
    }

    out.id = json["id"];

    return true;
}

// ── RPCResponse ─────────────────────────────────────────────────────

std::string RPCResponse::to_json() const {
    JsonValue obj = JsonValue::object();
    obj.set("result", result);
    obj.set("error", error);
    obj.set("id", id);
    return obj.to_string();
}

RPCResponse RPCResponse::success(JsonValue result_val, JsonValue id_val) {
    RPCResponse resp;
    resp.result = std::move(result_val);
    resp.error = JsonValue();  // null
    resp.id = std::move(id_val);
    return resp;
}

RPCResponse RPCResponse::make_error(int code, const std::string& message,
                                    JsonValue id_val) {
    RPCResponse resp;
    resp.result = JsonValue();  // null

    JsonValue err_obj = JsonValue::object();
    err_obj.set("code", JsonValue(static_cast<int64_t>(code)));
    err_obj.set("message", JsonValue(message));
    resp.error = std::move(err_obj);
    resp.id = std::move(id_val);
    return resp;
}

}  // namespace rnet::rpc
