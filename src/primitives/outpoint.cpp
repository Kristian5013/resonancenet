#include "primitives/outpoint.h"

namespace rnet::primitives {

std::string COutPoint::to_string() const {
    return hash.to_hex_rev() + ":" + std::to_string(n);
}

}  // namespace rnet::primitives
