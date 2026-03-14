// Copyright (c) 2025-2026 The ResonanceNet Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "wallet/addresses.h"

#include "core/logging.h"

#include <chrono>

namespace rnet::wallet {

// ===========================================================================
//  AddressManager -- dual-index address book (by string and pubkey hash)
// ===========================================================================
//
//  Every address is P2WPKH: [0x00][0x14][20-byte Hash160(pubkey)].
//  Two maps are kept in sync:
//    by_address_       : bech32-string  -> AddressEntry
//    hash_to_address_  : uint160 hash   -> bech32-string
//
//  Callers distinguish receive vs. change via the is_change flag.

// ---------------------------------------------------------------------------
// create_address -- encode a pubkey hash as bech32 and store it
// ---------------------------------------------------------------------------

Result<std::string> AddressManager::create_address(
    const uint160& pubkey_hash,
    const std::string& label,
    bool is_change,
    primitives::NetworkType net) {

    // 1. Encode the 20-byte hash as a bech32 P2WPKH address.
    std::string addr = primitives::encode_p2wpkh_address(pubkey_hash.data(), net);
    if (addr.empty()) {
        return Result<std::string>::err("failed to encode address");
    }

    // 2. Build the entry and insert into both maps.
    AddressEntry entry;
    entry.address = addr;
    entry.type = primitives::AddressType::P2WPKH;
    entry.pubkey_hash = pubkey_hash;
    entry.label = label;
    entry.creation_time = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    entry.is_change = is_change;
    entry.is_used = false;

    LOCK(mutex_);
    by_address_[addr] = entry;
    hash_to_address_[pubkey_hash] = addr;

    return Result<std::string>::ok(addr);
}

// ---------------------------------------------------------------------------
// add_entry -- insert a pre-built AddressEntry (e.g. loaded from DB)
// ---------------------------------------------------------------------------

Result<void> AddressManager::add_entry(const AddressEntry& entry) {
    LOCK(mutex_);
    by_address_[entry.address] = entry;
    hash_to_address_[entry.pubkey_hash] = entry.address;
    return Result<void>::ok();
}

// ---------------------------------------------------------------------------
// Lookups
// ---------------------------------------------------------------------------

Result<AddressEntry> AddressManager::get_by_address(const std::string& addr) const {
    LOCK(mutex_);
    auto it = by_address_.find(addr);
    if (it == by_address_.end()) {
        return Result<AddressEntry>::err("address not found");
    }
    return Result<AddressEntry>::ok(it->second);
}

Result<AddressEntry> AddressManager::get_by_pubkey_hash(const uint160& hash) const {
    LOCK(mutex_);
    auto it = hash_to_address_.find(hash);
    if (it == hash_to_address_.end()) {
        return Result<AddressEntry>::err("pubkey hash not found");
    }
    auto addr_it = by_address_.find(it->second);
    if (addr_it == by_address_.end()) {
        return Result<AddressEntry>::err("internal inconsistency");
    }
    return Result<AddressEntry>::ok(addr_it->second);
}

// ---------------------------------------------------------------------------
// Ownership checks
// ---------------------------------------------------------------------------

bool AddressManager::is_mine(const std::string& addr) const {
    LOCK(mutex_);
    return by_address_.count(addr) > 0;
}

bool AddressManager::is_mine_hash(const uint160& hash) const {
    LOCK(mutex_);
    return hash_to_address_.count(hash) > 0;
}

// ---------------------------------------------------------------------------
// mark_used -- flag an address as having appeared in a transaction
// ---------------------------------------------------------------------------

void AddressManager::mark_used(const std::string& addr) {
    LOCK(mutex_);
    auto it = by_address_.find(addr);
    if (it != by_address_.end()) {
        it->second.is_used = true;
    }
}

// ---------------------------------------------------------------------------
// Bulk retrieval
// ---------------------------------------------------------------------------

std::vector<AddressEntry> AddressManager::get_all(bool include_change) const {
    LOCK(mutex_);
    std::vector<AddressEntry> result;
    for (const auto& [_, entry] : by_address_) {
        if (include_change || !entry.is_change) {
            result.push_back(entry);
        }
    }
    return result;
}

std::vector<AddressEntry> AddressManager::get_receive_addresses() const {
    return get_all(false);
}

std::vector<AddressEntry> AddressManager::get_change_addresses() const {
    LOCK(mutex_);
    std::vector<AddressEntry> result;
    for (const auto& [_, entry] : by_address_) {
        if (entry.is_change) {
            result.push_back(entry);
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// size
// ---------------------------------------------------------------------------

size_t AddressManager::size() const {
    LOCK(mutex_);
    return by_address_.size();
}

} // namespace rnet::wallet
