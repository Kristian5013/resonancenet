#include "net/peer.h"

#include <sstream>

#include "core/logging.h"
#include "core/time.h"

namespace rnet::net {

CPeer::CPeer(uint64_t peer_id, const CNetAddr& address, bool inbound)
    : id(peer_id)
    , addr(address)
    , is_inbound(inbound)
    , connect_time(core::get_time())
{}

void CPeer::complete_handshake() {
    handshake_complete = true;
}

bool CPeer::has_service(ServiceFlags flag) const {
    return (services & static_cast<uint64_t>(flag)) != 0;
}

bool CPeer::misbehaving(int points, const std::string& reason) {
    int new_score = ban_score.fetch_add(points) + points;
    if (new_score >= BAN_THRESHOLD) {
        LogPrint(NET, "Peer %llu misbehaving (score=%d): %s",
                 static_cast<unsigned long long>(id), new_score,
                 reason.c_str());
        disconnect_requested.store(true);
        return true;
    }
    return false;
}

bool CPeer::should_disconnect() const {
    return disconnect_requested.load();
}

int64_t CPeer::get_ping_time() const {
    return ping_wait;
}

std::string CPeer::to_string() const {
    std::ostringstream oss;
    oss << "CPeer(id=" << id
        << " addr=" << addr.to_string()
        << " version=" << version
        << " inbound=" << (is_inbound ? "yes" : "no")
        << " height=" << start_height
        << ")";
    return oss.str();
}

}  // namespace rnet::net
