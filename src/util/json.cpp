#include "util/json.h"

#include <charconv>
#include <cmath>
#include <cstdio>

namespace rnet::util {

namespace {

void EscapeInto(std::string& out, std::string_view s) {
    out.push_back('"');
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[7];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
                    out += buf;
                } else {
                    out.push_back(c);
                }
        }
    }
    out.push_back('"');
}

class Parser {
public:
    explicit Parser(std::string_view text) : s_(text) {}

    Result<Json> ParseDocument() {
        SkipWs();
        auto v = ParseValue(0);
        if (!v) return v;
        SkipWs();
        if (pos_ != s_.size()) return Err("json: trailing data at offset " + std::to_string(pos_));
        return v;
    }

private:
    static constexpr int kMaxDepth = 64;  // bounds recursion on hostile input

    void SkipWs() {
        while (pos_ < s_.size() && (s_[pos_] == ' ' || s_[pos_] == '\t' || s_[pos_] == '\n' ||
                                    s_[pos_] == '\r')) {
            ++pos_;
        }
    }

    bool Consume(char c) {
        if (pos_ < s_.size() && s_[pos_] == c) { ++pos_; return true; }
        return false;
    }

    bool Literal(std::string_view lit) {
        if (s_.substr(pos_, lit.size()) != lit) return false;
        pos_ += lit.size();
        return true;
    }

    Result<Json> ParseValue(int depth) {
        if (depth > kMaxDepth) return Err("json: nesting too deep");
        if (pos_ >= s_.size()) return Err("json: unexpected end of input");
        switch (s_[pos_]) {
            case '{': return ParseObject(depth);
            case '[': return ParseArray(depth);
            case '"': {
                auto str = ParseString();
                if (!str) return Err(str.error());
                return Json(str.value());
            }
            case 't': if (Literal("true")) return Json(true); return Err("json: bad literal");
            case 'f': if (Literal("false")) return Json(false); return Err("json: bad literal");
            case 'n': if (Literal("null")) return Json(nullptr); return Err("json: bad literal");
            default: return ParseNumber();
        }
    }

    Result<Json> ParseObject(int depth) {
        if (!Consume('{')) return Err("json: expected '{'");
        Json obj = Json::Object();
        SkipWs();
        if (Consume('}')) return obj;
        while (true) {
            SkipWs();
            auto key = ParseString();
            if (!key) return Err(key.error());
            SkipWs();
            if (!Consume(':')) return Err("json: expected ':'");
            SkipWs();
            auto val = ParseValue(depth + 1);
            if (!val) return val;
            obj.Set(key.value(), val.value());
            SkipWs();
            if (Consume(',')) continue;
            if (Consume('}')) return obj;
            return Err("json: expected ',' or '}'");
        }
    }

    Result<Json> ParseArray(int depth) {
        if (!Consume('[')) return Err("json: expected '['");
        Json arr = Json::Array();
        SkipWs();
        if (Consume(']')) return arr;
        while (true) {
            SkipWs();
            auto val = ParseValue(depth + 1);
            if (!val) return val;
            arr.Push(val.value());
            SkipWs();
            if (Consume(',')) continue;
            if (Consume(']')) return arr;
            return Err("json: expected ',' or ']'");
        }
    }

    Result<std::string> ParseString() {
        if (!Consume('"')) return Err("json: expected string");
        std::string out;
        while (true) {
            if (pos_ >= s_.size()) return Err("json: unterminated string");
            const char c = s_[pos_++];
            if (c == '"') return out;
            if (static_cast<unsigned char>(c) < 0x20) return Err("json: control char in string");
            if (c != '\\') { out.push_back(c); continue; }
            if (pos_ >= s_.size()) return Err("json: unterminated escape");
            const char e = s_[pos_++];
            switch (e) {
                case '"':  out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/':  out.push_back('/'); break;
                case 'n':  out.push_back('\n'); break;
                case 'r':  out.push_back('\r'); break;
                case 't':  out.push_back('\t'); break;
                case 'b':  out.push_back('\b'); break;
                case 'f':  out.push_back('\f'); break;
                case 'u': {
                    if (pos_ + 4 > s_.size()) return Err("json: truncated \\u escape");
                    unsigned code = 0;
                    for (int i = 0; i < 4; ++i) {
                        const char h = s_[pos_++];
                        unsigned d;
                        if (h >= '0' && h <= '9') d = static_cast<unsigned>(h - '0');
                        else if (h >= 'a' && h <= 'f') d = static_cast<unsigned>(h - 'a' + 10);
                        else if (h >= 'A' && h <= 'F') d = static_cast<unsigned>(h - 'A' + 10);
                        else return Err("json: bad \\u escape");
                        code = (code << 4) | d;
                    }
                    // Encode as UTF-8 (surrogate pairs are passed through as-is;
                    // callers that need strict Unicode should validate separately).
                    if (code < 0x80) {
                        out.push_back(static_cast<char>(code));
                    } else if (code < 0x800) {
                        out.push_back(static_cast<char>(0xC0 | (code >> 6)));
                        out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
                    } else {
                        out.push_back(static_cast<char>(0xE0 | (code >> 12)));
                        out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
                        out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
                    }
                    break;
                }
                default: return Err("json: unknown escape");
            }
        }
    }

    Result<Json> ParseNumber() {
        const size_t start = pos_;
        if (pos_ < s_.size() && (s_[pos_] == '-' || s_[pos_] == '+')) ++pos_;
        bool is_double = false;
        while (pos_ < s_.size()) {
            const char c = s_[pos_];
            if (c >= '0' && c <= '9') { ++pos_; continue; }
            if (c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') {
                is_double = true;
                ++pos_;
                continue;
            }
            break;
        }
        const std::string_view tok = s_.substr(start, pos_ - start);
        if (tok.empty()) return Err("json: expected value");
        if (!is_double) {
            int64_t v = 0;
            const auto res = std::from_chars(tok.data(), tok.data() + tok.size(), v);
            if (res.ec == std::errc() && res.ptr == tok.data() + tok.size()) return Json(v);
            // Integer literal that does not fit int64: keep it as a double rather
            // than silently wrapping.
        }
        double d = 0.0;
        const auto res = std::from_chars(tok.data(), tok.data() + tok.size(), d);
        if (res.ec != std::errc() || res.ptr != tok.data() + tok.size()) {
            return Err("json: malformed number");
        }
        return Json(d);
    }

    std::string_view s_;
    size_t pos_{0};
};

}  // namespace

Json Json::Array(std::vector<Json> items) {
    Json j;
    j.type_ = Type::Array;
    j.arr_ = std::move(items);
    return j;
}

Json Json::Object() {
    Json j;
    j.type_ = Type::Object;
    return j;
}

Result<bool> Json::AsBool() const {
    if (type_ != Type::Bool) return Err("json: not a bool");
    return bool_;
}

Result<int64_t> Json::AsInt() const {
    if (type_ != Type::Int) return Err("json: not an integer");
    return int_;
}

Result<double> Json::AsDouble() const {
    if (type_ == Type::Double) return double_;
    if (type_ == Type::Int) return static_cast<double>(int_);
    return Err("json: not a number");
}

Result<std::string> Json::AsString() const {
    if (type_ != Type::String) return Err("json: not a string");
    return str_;
}

void Json::Set(std::string key, Json value) {
    type_ = Type::Object;
    obj_[std::move(key)] = std::move(value);
}

void Json::Push(Json value) {
    type_ = Type::Array;
    arr_.push_back(std::move(value));
}

bool Json::Has(std::string_view key) const {
    return obj_.find(std::string(key)) != obj_.end();
}

Result<Json> Json::At(std::string_view key) const {
    if (type_ != Type::Object) return Err("json: not an object");
    const auto it = obj_.find(std::string(key));
    if (it == obj_.end()) return Err("json: missing key '" + std::string(key) + "'");
    return it->second;
}

std::string Json::Dump(int indent) const {
    std::string out;
    DumpTo(out, indent, 0);
    return out;
}

void Json::DumpTo(std::string& out, int indent, int depth) const {
    const std::string pad = indent > 0 ? std::string(static_cast<size_t>(indent * (depth + 1)), ' ') : "";
    const std::string pad_close = indent > 0 ? std::string(static_cast<size_t>(indent * depth), ' ') : "";
    const char* nl = indent > 0 ? "\n" : "";

    switch (type_) {
        case Type::Null: out += "null"; return;
        case Type::Bool: out += bool_ ? "true" : "false"; return;
        case Type::Int: out += std::to_string(int_); return;
        case Type::Double: {
            char buf[40];
            std::snprintf(buf, sizeof(buf), "%.17g", double_);
            out += buf;
            return;
        }
        case Type::String: EscapeInto(out, str_); return;
        case Type::Array: {
            if (arr_.empty()) { out += "[]"; return; }
            out += "[";
            out += nl;
            for (size_t i = 0; i < arr_.size(); ++i) {
                out += pad;
                arr_[i].DumpTo(out, indent, depth + 1);
                if (i + 1 < arr_.size()) out += ",";
                out += nl;
            }
            out += pad_close;
            out += "]";
            return;
        }
        case Type::Object: {
            if (obj_.empty()) { out += "{}"; return; }
            out += "{";
            out += nl;
            size_t i = 0;
            for (const auto& [k, v] : obj_) {
                out += pad;
                EscapeInto(out, k);
                out += indent > 0 ? ": " : ":";
                v.DumpTo(out, indent, depth + 1);
                if (++i < obj_.size()) out += ",";
                out += nl;
            }
            out += pad_close;
            out += "}";
            return;
        }
    }
}

Result<Json> Json::Parse(std::string_view text) { return Parser(text).ParseDocument(); }

}  // namespace rnet::util
