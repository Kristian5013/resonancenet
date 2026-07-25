// A small, strict JSON value type — enough for RPC and human-readable manifests,
// with no external dependency.
//
// Strictness matters: parsing is used on untrusted RPC input, so trailing data,
// unterminated strings, bad escapes, control characters in strings and malformed
// numbers are all errors. Integers are kept exact (int64) and never routed
// through double — money and token counts must not lose precision above 2^53.
#pragma once

#include <cstdint>
#include <initializer_list>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "util/result.h"

namespace rnet::util {

class Json {
public:
    enum class Type { Null, Bool, Int, Double, String, Array, Object };

    Json() = default;
    Json(std::nullptr_t) {}                                       // NOLINT
    Json(bool v) : type_(Type::Bool), bool_(v) {}                 // NOLINT
    Json(int64_t v) : type_(Type::Int), int_(v) {}                // NOLINT
    Json(int v) : type_(Type::Int), int_(v) {}                    // NOLINT
    Json(uint64_t v) : type_(Type::Int), int_(static_cast<int64_t>(v)) {}  // NOLINT
    Json(double v) : type_(Type::Double), double_(v) {}           // NOLINT
    Json(std::string v) : type_(Type::String), str_(std::move(v)) {}       // NOLINT
    Json(const char* v) : type_(Type::String), str_(v) {}         // NOLINT

    static Json Array(std::vector<Json> items = {});
    static Json Object();

    Type type() const { return type_; }
    bool is_null() const { return type_ == Type::Null; }
    bool is_object() const { return type_ == Type::Object; }
    bool is_array() const { return type_ == Type::Array; }

    // Typed accessors: return an error rather than a default when the type or key
    // is wrong, so callers cannot silently proceed on malformed input.
    Result<bool> AsBool() const;
    Result<int64_t> AsInt() const;
    Result<double> AsDouble() const;
    Result<std::string> AsString() const;

    const std::vector<Json>& items() const { return arr_; }
    const std::map<std::string, Json>& entries() const { return obj_; }

    void Set(std::string key, Json value);
    void Push(Json value);
    bool Has(std::string_view key) const;
    Result<Json> At(std::string_view key) const;

    std::string Dump(int indent = 0) const;

    static Result<Json> Parse(std::string_view text);

private:
    void DumpTo(std::string& out, int indent, int depth) const;

    Type type_{Type::Null};
    bool bool_{false};
    int64_t int_{0};
    double double_{0.0};
    std::string str_;
    std::vector<Json> arr_;
    std::map<std::string, Json> obj_;
};

}  // namespace rnet::util
