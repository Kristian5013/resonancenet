#include "primitives/txout.h"

#include <cstring>

namespace rnet::primitives {

std::string CTxOut::to_string() const {
    return "CTxOut(value=" + FormatMoney(value) +
           ", scriptPubKey.size=" + std::to_string(script_pub_key.size()) + ")";
}

std::vector<uint8_t> make_p2wpkh_script(const uint8_t* hash160) {
    // OP_0 OP_PUSHBYTES_20 <20-byte-hash>
    std::vector<uint8_t> script(22);
    script[0] = 0x00;  // OP_0 (witness version 0)
    script[1] = 0x14;  // Push 20 bytes
    std::memcpy(script.data() + 2, hash160, 20);
    return script;
}

std::vector<uint8_t> make_p2wsh_script(const uint8_t* hash256) {
    // OP_0 OP_PUSHBYTES_32 <32-byte-hash>
    std::vector<uint8_t> script(34);
    script[0] = 0x00;  // OP_0 (witness version 0)
    script[1] = 0x20;  // Push 32 bytes
    std::memcpy(script.data() + 2, hash256, 32);
    return script;
}

}  // namespace rnet::primitives
