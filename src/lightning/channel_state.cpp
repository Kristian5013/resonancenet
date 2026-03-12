#include "lightning/channel_state.h"

#include "crypto/keccak.h"

namespace rnet::lightning {

std::string_view channel_state_name(ChannelState state) {
    switch (state) {
        case ChannelState::PREOPENING:        return "PREOPENING";
        case ChannelState::FUNDING_CREATED:   return "FUNDING_CREATED";
        case ChannelState::FUNDING_BROADCAST: return "FUNDING_BROADCAST";
        case ChannelState::FUNDING_LOCKED:    return "FUNDING_LOCKED";
        case ChannelState::NORMAL:            return "NORMAL";
        case ChannelState::SHUTDOWN:          return "SHUTDOWN";
        case ChannelState::FORCE_CLOSING:     return "FORCE_CLOSING";
        case ChannelState::CLOSED:            return "CLOSED";
        default:                              return "UNKNOWN";
    }
}

ChannelId make_channel_id(const primitives::COutPoint& funding_outpoint) {
    ChannelId id = funding_outpoint.hash;
    // XOR the output index into the last 4 bytes
    uint32_t n = funding_outpoint.n;
    id[28] ^= static_cast<uint8_t>((n >> 24) & 0xFF);
    id[29] ^= static_cast<uint8_t>((n >> 16) & 0xFF);
    id[30] ^= static_cast<uint8_t>((n >> 8) & 0xFF);
    id[31] ^= static_cast<uint8_t>(n & 0xFF);
    return id;
}

uint64_t CommitmentNumber::obscured(const uint256& obscure_factor) const {
    // Take last 6 bytes of the obscure factor
    uint64_t mask = 0;
    for (int i = 0; i < 6; ++i) {
        mask |= static_cast<uint64_t>(obscure_factor[26 + i]) << (i * 8);
    }
    return number ^ mask;
}

}  // namespace rnet::lightning
