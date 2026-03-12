#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace rnet::rpc {

// ── JSON Value ──────────────────────────────────────────────────────

/// Minimal JSON value type — recursive variant supporting null, bool,
/// int64, double, string, array, object.
class JsonValue {
public:
    enum Type {
        JSON_NULL,
        JSON_BOOL,
        JSON_INT,
        JSON_DOUBLE,
        JSON_STRING,
        JSON_ARRAY,
        JSON_OBJECT,
    };

    using Array  = std::vector<JsonValue>;
    using Object = std::map<std::string, JsonValue>;

    /// Constructors
    JsonValue();                                       // null
    explicit JsonValue(bool v);
    explicit JsonValue(int v);
    explicit JsonValue(int64_t v);
    explicit JsonValue(uint64_t v);
    explicit JsonValue(double v);
    explicit JsonValue(const char* v);
    explicit JsonValue(std::string v);
    explicit JsonValue(std::string_view v);
    explicit JsonValue(Array v);
    explicit JsonValue(Object v);

    /// Named constructors
    static JsonValue null_value();
    static JsonValue boolean(bool v);
    static JsonValue integer(int64_t v);
    static JsonValue number(double v);
    static JsonValue string(std::string v);
    static JsonValue array(Array v = {});
    static JsonValue object(Object v = {});

    /// Type queries
    Type type() const { return type_; }
    bool is_null()   const { return type_ == JSON_NULL; }
    bool is_bool()   const { return type_ == JSON_BOOL; }
    bool is_int()    const { return type_ == JSON_INT; }
    bool is_double() const { return type_ == JSON_DOUBLE; }
    bool is_string() const { return type_ == JSON_STRING; }
    bool is_array()  const { return type_ == JSON_ARRAY; }
    bool is_object() const { return type_ == JSON_OBJECT; }
    bool is_number() const { return type_ == JSON_INT || type_ == JSON_DOUBLE; }

    /// Value accessors (undefined behaviour if wrong type)
    bool        as_bool()   const;
    int64_t     as_int()    const;
    double      as_double() const;
    const std::string& as_string() const;
    const Array&       as_array()  const;
    const Object&      as_object() const;
    Array&             as_array();
    Object&            as_object();

    /// Object access by key. Returns JSON_NULL if key not found.
    const JsonValue& operator[](const std::string& key) const;

    /// Array access by index. Returns JSON_NULL if out of range.
    const JsonValue& operator[](size_t index) const;

    /// Object helpers
    bool has_key(const std::string& key) const;
    void set(const std::string& key, JsonValue val);

    /// Array helpers
    void push_back(JsonValue val);
    size_t size() const;

    /// Serialize to JSON string
    std::string to_string() const;

    /// Pretty-print with indentation
    std::string to_string_pretty(int indent = 2) const;

private:
    void to_string_impl(std::string& out) const;
    void to_string_pretty_impl(std::string& out, int indent, int depth) const;

    static std::string escape_string(const std::string& s);

    Type type_ = JSON_NULL;

    // Storage
    bool        bool_val_ = false;
    int64_t     int_val_ = 0;
    double      double_val_ = 0.0;
    std::string string_val_;
    Array       array_val_;
    Object      object_val_;
};

// ── JSON Parser ─────────────────────────────────────────────────────

/// Parse a JSON string into a JsonValue. Returns null on error.
JsonValue parse_json(std::string_view input);

/// Parse JSON, returning false on failure.
bool parse_json(std::string_view input, JsonValue& out);

// ── RPC Request / Response ──────────────────────────────────────────

/// JSON-RPC 2.0 request
struct RPCRequest {
    std::string method;
    JsonValue   params;    // array or object
    JsonValue   id;        // string, int, or null

    /// Parse from JSON body
    static bool from_json(const JsonValue& json, RPCRequest& out);
};

/// JSON-RPC 2.0 response
struct RPCResponse {
    JsonValue result;
    JsonValue error;
    JsonValue id;

    /// Serialize to JSON string
    std::string to_json() const;

    /// Convenience constructors
    static RPCResponse success(JsonValue result, JsonValue id);
    static RPCResponse make_error(int code, const std::string& message, JsonValue id);
};

}  // namespace rnet::rpc
