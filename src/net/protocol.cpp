#include "net/protocol.h"

#include <algorithm>
#include <cstring>
#include <sstream>

namespace rnet::net {

std::string CInv::to_string() const {
    return inv_type_string(type) + " " + hash.to_hex().substr(0, 16) + "...";
}

void CNetAddr::set_ipv4(uint8_t a, uint8_t b, uint8_t c, uint8_t d) {
    ip.fill(0);
    // IPv4-mapped IPv6: ::ffff:x.x.x.x
    ip[10] = 0xFF;
    ip[11] = 0xFF;
    ip[12] = a;
    ip[13] = b;
    ip[14] = c;
    ip[15] = d;
}

std::string CNetAddr::to_string() const {
    if (is_ipv4()) {
        std::ostringstream oss;
        oss << static_cast<int>(ip[12]) << "."
            << static_cast<int>(ip[13]) << "."
            << static_cast<int>(ip[14]) << "."
            << static_cast<int>(ip[15])
            << ":" << port;
        return oss.str();
    }
    // Full IPv6
    std::ostringstream oss;
    oss << "[";
    for (int i = 0; i < 16; i += 2) {
        if (i > 0) oss << ":";
        oss << std::hex
            << (static_cast<int>(ip[static_cast<size_t>(i)]) << 8 |
                static_cast<int>(ip[static_cast<size_t>(i + 1)]));
    }
    oss << "]:" << std::dec << port;
    return oss.str();
}

bool CNetAddr::is_ipv4() const {
    // Check for IPv4-mapped IPv6 address
    for (int i = 0; i < 10; ++i) {
        if (ip[static_cast<size_t>(i)] != 0) return false;
    }
    return ip[10] == 0xFF && ip[11] == 0xFF;
}

bool CNetAddr::is_routable() const {
    return !is_local() && !ip[12] == 0;
}

bool CNetAddr::is_local() const {
    if (is_ipv4()) {
        // 127.0.0.0/8
        return ip[12] == 127;
    }
    // ::1
    for (int i = 0; i < 15; ++i) {
        if (ip[static_cast<size_t>(i)] != 0) return false;
    }
    return ip[15] == 1;
}

void MessageHeader::set_command(std::string_view cmd) {
    command.fill(0);
    size_t len = std::min(cmd.size(), COMMAND_SIZE);
    std::memcpy(command.data(), cmd.data(), len);
}

std::string MessageHeader::get_command() const {
    // Find the null terminator or end of array
    size_t len = 0;
    while (len < COMMAND_SIZE && command[len] != 0) {
        ++len;
    }
    return std::string(command.data(), len);
}

bool MessageHeader::valid_magic() const {
    return magic == NETWORK_MAGIC_REF();
}

std::string inv_type_string(InvType type) {
    switch (type) {
        case InvType::INV_ERROR:          return "ERROR";
        case InvType::INV_TX:             return "TX";
        case InvType::INV_BLOCK:          return "BLOCK";
        case InvType::INV_FILTERED_BLOCK: return "FILTERED_BLOCK";
        case InvType::INV_WITNESS_TX:     return "WITNESS_TX";
        case InvType::INV_WITNESS_BLOCK:  return "WITNESS_BLOCK";
        case InvType::INV_CHECKPOINT:     return "CHECKPOINT";
        default:                          return "UNKNOWN";
    }
}

}  // namespace rnet::net
