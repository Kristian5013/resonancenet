// Copyright (c) 2024-present ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "core/error.h"

#include <sstream>

namespace rnet::core {

// ===========================================================================
//  Error formatting utilities
// ===========================================================================

// ---------------------------------------------------------------------------
// format_error / format_error_chain
//
// format_error:       "[CODE_NAME] message"
// format_error_chain: "err1 -> err2 -> err3" (causal chain, root first)
// ---------------------------------------------------------------------------
std::string format_error(ErrorCode code, const std::string& msg) {
    std::ostringstream oss;
    oss << "[" << error_code_name(code) << "] " << msg;
    return oss.str();
}

std::string format_error_chain(const std::vector<Error>& errors) {
    if (errors.empty()) return "no error";

    std::ostringstream oss;
    for (size_t i = 0; i < errors.size(); ++i) {
        if (i > 0) oss << " -> ";
        oss << errors[i].to_string();
    }
    return oss.str();
}

// ---------------------------------------------------------------------------
// Factory helpers — one per common ErrorCode.
// ---------------------------------------------------------------------------
Error make_error(ErrorCode code, const std::string& msg) {
    return Error{code, msg};
}

Error make_invalid_arg(const std::string& msg) {
    return Error{ErrorCode::INVALID_ARGUMENT, msg};
}

Error make_not_found(const std::string& msg) {
    return Error{ErrorCode::NOT_FOUND, msg};
}

Error make_io_error(const std::string& msg) {
    return Error{ErrorCode::IO_FAILURE, msg};
}

Error make_parse_error(const std::string& msg) {
    return Error{ErrorCode::PARSE_FAILURE, msg};
}

Error make_timeout(const std::string& msg) {
    return Error{ErrorCode::TIMEOUT, msg};
}

Error make_internal(const std::string& msg) {
    return Error{ErrorCode::INTERNAL, msg};
}

} // namespace rnet::core
