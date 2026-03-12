#pragma once

#include <map>
#include <string>
#include <vector>

#include "core/error.h"
#include "core/sync.h"
#include "core/types.h"
#include "primitives/address.h"

namespace rnet::wallet {

/// Tracked address entry.
struct AddressEntry {
    std::string address;                           ///< Encoded address string
    primitives::AddressType type = primitives::AddressType::P2WPKH;
    uint160 pubkey_hash;                           ///< Hash160 of the pubkey
    std::string label;
    int64_t creation_time = 0;
    bool is_change = false;
    bool is_used = false;                          ///< Has received funds
};

/// AddressManager: tracks generated addresses and their metadata.
class AddressManager {
public:
    AddressManager() = default;

    /// Generate a P2WPKH address from a pubkey hash.
    Result<std::string> create_address(
        const uint160& pubkey_hash,
        const std::string& label = "",
        bool is_change = false,
        primitives::NetworkType net = primitives::NetworkType::MAINNET);

    /// Add a pre-built address entry.
    Result<void> add_entry(const AddressEntry& entry);

    /// Look up an address entry by address string.
    Result<AddressEntry> get_by_address(const std::string& addr) const;

    /// Look up an address entry by pubkey hash.
    Result<AddressEntry> get_by_pubkey_hash(const uint160& hash) const;

    /// Check if an address is ours.
    bool is_mine(const std::string& addr) const;

    /// Check if a pubkey hash belongs to us.
    bool is_mine_hash(const uint160& hash) const;

    /// Mark an address as used.
    void mark_used(const std::string& addr);

    /// Get all addresses (optionally filter by change/receive).
    std::vector<AddressEntry> get_all(bool include_change = true) const;

    /// Get all receive (non-change) addresses.
    std::vector<AddressEntry> get_receive_addresses() const;

    /// Get all change addresses.
    std::vector<AddressEntry> get_change_addresses() const;

    /// Number of tracked addresses.
    size_t size() const;

private:
    mutable core::Mutex mutex_;
    std::map<std::string, AddressEntry> by_address_;
    std::map<uint160, std::string> hash_to_address_;
};

}  // namespace rnet::wallet
