#include "primitives/txin.h"

namespace rnet::primitives {

std::string CTxIn::to_string() const {
    std::string result = "CTxIn(";
    result += prevout.to_string();
    result += ", scriptSig.size=" + std::to_string(script_sig.size());
    result += ", seq=" + std::to_string(sequence);
    if (!witness.is_null()) {
        result += ", witness=" + witness.to_string();
    }
    result += ")";
    return result;
}

}  // namespace rnet::primitives
