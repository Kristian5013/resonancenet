#pragma once

#include <string>

namespace rnet::consensus {

/// Validation outcome for blocks and transactions.
enum class ValidationResult {
    VALID,
    INVALID,
    INCONCLUSIVE,
};

/// Tracks validation state during block/tx checking.
struct ValidationState {
    ValidationResult result = ValidationResult::VALID;
    std::string reject_reason;

    bool is_valid() const {
        return result == ValidationResult::VALID;
    }

    void invalid(std::string reason) {
        result = ValidationResult::INVALID;
        reject_reason = std::move(reason);
    }

    void inconclusive(std::string reason) {
        result = ValidationResult::INCONCLUSIVE;
        reject_reason = std::move(reason);
    }
};

}  // namespace rnet::consensus
