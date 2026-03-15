// Copyright (c) 2024-2026 The ResonanceNet Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "chain/chainstate.h"

#include "consensus/block_reward.h"
#include "consensus/block_verify.h"
#include "consensus/genesis.h"
#include "consensus/tx_verify.h"
#include "core/logging.h"

#include <unordered_map>
#include <vector>

namespace rnet::chain {

// -- File-local undo storage ------------------------------------------------

static std::unordered_map<rnet::uint256, std::vector<Coin>> block_undo_data_;

// -- Construction / destruction ---------------------------------------------

// ---------------------------------------------------------------------------
// CChainState (constructor)
// ---------------------------------------------------------------------------
CChainState::CChainState(const consensus::ConsensusParams& params,
                         std::unique_ptr<CCoinsView> coins_view,
                         std::unique_ptr<BlockStorage> storage)
    : params_(params)
    , coins_view_(std::move(coins_view))
    , coins_cache_(coins_view_.get())
    , storage_(std::move(storage))
{}

// ---------------------------------------------------------------------------
// ~CChainState
// ---------------------------------------------------------------------------
CChainState::~CChainState() = default;

// -- Genesis ----------------------------------------------------------------

// ---------------------------------------------------------------------------
// load_genesis
// ---------------------------------------------------------------------------
// Creates the genesis block, writes it to disk, and sets it as the active
// chain tip with its coinbase outputs added to the UTXO set.
//
// Steps:
//   1. Check if genesis already loaded
//   2. Build block index entry
//   3. Write genesis to disk
//   4. Set as tip
//   5. Add genesis coinbase outputs to UTXO set
// ---------------------------------------------------------------------------
Result<void> CChainState::load_genesis() {
    LOCK(cs_main_);

    const auto genesis = consensus::create_genesis_block(params_);
    const auto hash = genesis.hash();

    // 1. Check if genesis already loaded
    if (block_index_.count(hash)) {
        return Result<void>::ok();
    }

    // 2. Build block index entry
    auto idx = std::make_unique<CBlockIndex>(
        static_cast<const primitives::CBlockHeader&>(genesis));

    idx->status = CBlockIndex::FULLY_VALIDATED;
    idx->chain_tx = static_cast<int64_t>(genesis.vtx.size());

    auto* raw = idx.get();
    block_index_[hash] = std::move(idx);

    // 3. Write genesis to disk
    auto pos_result = storage_->write_block(genesis);
    if (!pos_result) {
        return Result<void>::err("Failed to write genesis: " + pos_result.error());
    }
    raw->file_number = pos_result.value().file_number;
    raw->data_pos = pos_result.value().pos;

    // 4. Set as tip
    tip_ = raw;
    active_chain_.clear();
    active_chain_.push_back(raw);

    coins_cache_.set_best_block(hash);

    // 5. Add genesis coinbase outputs to UTXO set
    if (!genesis.vtx.empty()) {
        const auto& coinbase = genesis.vtx[0];
        for (uint32_t i = 0; i < coinbase->vout().size(); ++i) {
            primitives::COutPoint outpoint(coinbase->txid(), i);
            Coin coin(coinbase->vout()[i], 0, true, genesis.val_loss);
            coins_cache_.add_coin(outpoint, std::move(coin));
        }
    }

    LogPrintf("Loaded genesis block: %s", hash.to_hex().c_str());

    return Result<void>::ok();
}

// -- Block index loading ----------------------------------------------------

// ---------------------------------------------------------------------------
// load_block_index
// ---------------------------------------------------------------------------
// Scans all block files on disk and rebuilds the in-memory block index so
// that the chain state survives across rnetd restarts.  Must be called
// after load_genesis() so the genesis block is already in the index.
//
// Steps:
//   1. Scan all stored blocks from disk
//   2. Insert each block into the block index with parent linkage
//   3. Find the best tip using the PoT fork-choice rule
//   4. Rebuild the active chain from genesis to best tip
//   5. Log the loaded chain height
// ---------------------------------------------------------------------------
Result<void> CChainState::load_block_index() {
    LOCK(cs_main_);

    // 1. Scan all stored blocks from disk
    auto scan_result = storage_->scan_block_files();
    if (!scan_result) {
        return Result<void>::err(
            "Failed to scan block files: " + scan_result.error());
    }

    const auto& stored_blocks = scan_result.value();
    if (stored_blocks.empty()) {
        LogPrintf("No blocks found on disk; starting from genesis");
        return Result<void>::ok();
    }

    // 2. Insert each block into the block index with parent linkage
    int loaded = 0;
    for (const auto& sb : stored_blocks) {
        const auto hash = sb.block.hash();

        // Skip if already in the index (e.g. genesis)
        if (block_index_.count(hash)) {
            // Update disk position for genesis too.
            auto* existing = block_index_[hash].get();
            existing->file_number = sb.pos.file_number;
            existing->data_pos = sb.pos.pos;
            continue;
        }

        // Find parent in the index
        auto parent_it = block_index_.find(sb.block.prev_hash);
        if (parent_it == block_index_.end()) {
            LogPrintf("load_block_index: skipping orphan block %s (no parent)",
                     hash.to_hex().c_str());
            continue;
        }
        auto* parent = parent_it->second.get();

        // Build block index entry with disk position
        auto idx = std::make_unique<CBlockIndex>(
            static_cast<const primitives::CBlockHeader&>(sb.block));
        idx->prev = parent;
        idx->status = CBlockIndex::FULLY_VALIDATED;
        idx->chain_tx = parent->chain_tx
                        + static_cast<int64_t>(sb.block.vtx.size());
        idx->file_number = sb.pos.file_number;
        idx->data_pos = sb.pos.pos;

        auto* raw = idx.get();
        block_index_[hash] = std::move(idx);
        ++loaded;
    }

    // 3. Find the best tip using the PoT fork-choice rule
    CBlockIndex* best = nullptr;
    for (const auto& [hash, idx] : block_index_) {
        if (idx->status >= CBlockIndex::FULLY_VALIDATED) {
            if (!best || is_better_tip(idx.get(), best)) {
                best = idx.get();
            }
        }
    }

    if (!best) {
        return Result<void>::ok();
    }

    // 4. Rebuild the active chain from genesis to best tip
    tip_ = best;
    active_chain_.clear();
    active_chain_.resize(static_cast<size_t>(best->height) + 1, nullptr);

    CBlockIndex* walk = best;
    while (walk) {
        active_chain_[static_cast<size_t>(walk->height)] = walk;
        walk = walk->prev;
    }

    coins_cache_.set_best_block(best->block_hash);

    // 5. Log the loaded chain height
    LogPrintf("Loaded block index: %d blocks, tip height=%d hash=%s",
             loaded, best->height, best->block_hash.to_hex().c_str());

    return Result<void>::ok();
}

// -- Header acceptance ------------------------------------------------------

// ---------------------------------------------------------------------------
// accept_block_header
// ---------------------------------------------------------------------------
// Validates a block header against its parent and inserts it into the block
// index if not already known.
//
// Steps:
//   1. Return early if header already known
//   2. Find parent in block index
//   3. Validate header against parent
//   4. Insert into block index
// ---------------------------------------------------------------------------
Result<CBlockIndex*> CChainState::accept_block_header(
    const primitives::CBlockHeader& header)
{
    LOCK(cs_main_);

    const auto hash = header.hash();

    // 1. Already known?
    auto it = block_index_.find(hash);
    if (it != block_index_.end()) {
        return Result<CBlockIndex*>::ok(it->second.get());
    }

    // 2. Find parent
    auto parent_it = block_index_.find(header.prev_hash);
    if (parent_it == block_index_.end()) {
        return Result<CBlockIndex*>::err(
            "Unknown parent block: " + header.prev_hash.to_hex());
    }
    auto* parent = parent_it->second.get();

    // 3. Validate header against parent
    consensus::ValidationState state;
    if (!consensus::check_block_header(header, parent->header, state, params_)) {
        return Result<CBlockIndex*>::err(
            "Invalid block header: " + state.reject_reason);
    }

    // 4. Insert into block index
    auto idx = std::make_unique<CBlockIndex>(header);
    idx->prev = parent;
    idx->status = CBlockIndex::HEADER_VALID;

    auto* raw = idx.get();
    block_index_[hash] = std::move(idx);

    return Result<CBlockIndex*>::ok(raw);
}

// -- Block acceptance -------------------------------------------------------

// ---------------------------------------------------------------------------
// accept_block
// ---------------------------------------------------------------------------
// Accepts a full block: validates the header, performs context-free block
// validation, writes to disk, and attempts to activate the best chain.
//
// Steps:
//   1. Accept header first
//   2. Return early if already fully validated
//   3. Context-free block validation
//   4. Write to disk
//   5. Try to activate best chain (may connect this block or trigger reorg)
// ---------------------------------------------------------------------------
Result<CBlockIndex*> CChainState::accept_block(
    const primitives::CBlock& block)
{
    // 1. Accept header first
    auto header_result = accept_block_header(
        static_cast<const primitives::CBlockHeader&>(block));
    if (!header_result) {
        return header_result;
    }

    CBlockIndex* pindex = header_result.value();

    LOCK(cs_main_);

    // 2. Already fully validated?
    if (pindex->status >= CBlockIndex::FULLY_VALIDATED) {
        return Result<CBlockIndex*>::ok(pindex);
    }

    // 3. Context-free block validation
    consensus::ValidationState state;
    if (!consensus::check_block(block, state, params_)) {
        return Result<CBlockIndex*>::err(
            "Invalid block: " + state.reject_reason);
    }

    // 4. Write to disk
    auto pos_result = storage_->write_block(block);
    if (!pos_result) {
        return Result<CBlockIndex*>::err(
            "Failed to write block: " + pos_result.error());
    }
    pindex->file_number = pos_result.value().file_number;
    pindex->data_pos = pos_result.value().pos;
    pindex->status = CBlockIndex::TREE_VALID;
    pindex->chain_tx = (pindex->prev ? pindex->prev->chain_tx : 0)
                       + static_cast<int64_t>(block.vtx.size());

    // 5. Try to activate best chain (may connect this block or trigger reorg)
    auto activate_result = activate_best_chain(&block);
    if (!activate_result) {
        return Result<CBlockIndex*>::err(activate_result.error());
    }

    return Result<CBlockIndex*>::ok(pindex);
}

// -- Chain accessors --------------------------------------------------------

// ---------------------------------------------------------------------------
// tip
// ---------------------------------------------------------------------------
CBlockIndex* CChainState::tip() const {
    return tip_;
}

// ---------------------------------------------------------------------------
// lookup_block_index
// ---------------------------------------------------------------------------
CBlockIndex* CChainState::lookup_block_index(
    const rnet::uint256& hash) const
{
    const auto it = block_index_.find(hash);
    if (it == block_index_.end()) return nullptr;
    return it->second.get();
}

// ---------------------------------------------------------------------------
// height
// ---------------------------------------------------------------------------
int CChainState::height() const {
    return tip_ ? tip_->height : -1;
}

// ---------------------------------------------------------------------------
// get_active_chain
// ---------------------------------------------------------------------------
std::vector<CBlockIndex*> CChainState::get_active_chain() const {
    return active_chain_;
}

// ---------------------------------------------------------------------------
// get_block_by_height
// ---------------------------------------------------------------------------
CBlockIndex* CChainState::get_block_by_height(int h) const {
    if (h < 0 || h >= static_cast<int>(active_chain_.size())) {
        return nullptr;
    }
    return active_chain_[static_cast<size_t>(h)];
}

// ---------------------------------------------------------------------------
// is_on_active_chain
// ---------------------------------------------------------------------------
bool CChainState::is_on_active_chain(const rnet::uint256& hash) const {
    const auto* idx = lookup_block_index(hash);
    if (!idx) return false;
    if (idx->height < 0 || idx->height >= static_cast<int>(active_chain_.size())) {
        return false;
    }
    return active_chain_[static_cast<size_t>(idx->height)] == idx;
}

// ---------------------------------------------------------------------------
// insert_block_index
// ---------------------------------------------------------------------------
CBlockIndex* CChainState::insert_block_index(const rnet::uint256& hash) {
    const auto it = block_index_.find(hash);
    if (it != block_index_.end()) return it->second.get();

    auto idx = std::make_unique<CBlockIndex>();
    idx->block_hash = hash;
    auto* raw = idx.get();
    block_index_[hash] = std::move(idx);
    return raw;
}

// -- Chain activation / reorg -----------------------------------------------

// ---------------------------------------------------------------------------
// activate_best_chain
// ---------------------------------------------------------------------------
// Switches the active chain to the best candidate tip. Handles reorgs by
// disconnecting blocks back to the fork point, then connecting blocks
// along the new best chain.
//
// Steps:
//   1. Find the fork point between current tip and new best tip
//   2. Disconnect blocks from old tip to fork point
//   3. Build the list of blocks to connect
//   4. Connect blocks from fork point to new best tip
// ---------------------------------------------------------------------------
Result<void> CChainState::activate_best_chain(
    const primitives::CBlock* new_block)
{
    CBlockIndex* best = find_best_tip();
    if (!best || best == tip_) {
        return Result<void>::ok();
    }

    // 1. Find the fork point
    CBlockIndex* fork = tip_;
    CBlockIndex* walk = best;

    while (fork && walk && fork->height > walk->height) fork = fork->prev;
    while (fork && walk && walk->height > fork->height) walk = walk->prev;
    while (fork && walk && fork != walk) { fork = fork->prev; walk = walk->prev; }
    CBlockIndex* fork_point = fork;

    // 2. Disconnect blocks from old tip to fork point
    if (tip_ && fork_point) {
        CBlockIndex* disconnect = tip_;
        while (disconnect != fork_point) {
            auto result = disconnect_block(disconnect);
            if (!result) return result;
            disconnect = disconnect->prev;
        }
    }

    // 3. Build the list of blocks to connect
    std::vector<CBlockIndex*> to_connect;
    for (CBlockIndex* idx = best; idx != fork_point; idx = idx->prev) {
        to_connect.push_back(idx);
    }
    std::reverse(to_connect.begin(), to_connect.end());

    // 4. Connect blocks from fork point to new best tip
    for (auto* idx : to_connect) {
        const bool use_memory = (new_block && idx->block_hash == new_block->hash());

        if (use_memory) {
            auto connect_result = connect_block(*new_block, idx);
            if (!connect_result) {
                return connect_result;
            }
        } else {
            DiskBlockPos dpos;
            dpos.file_number = idx->file_number;
            dpos.pos = idx->data_pos;

            auto block_result = storage_->read_block(dpos);
            if (!block_result) {
                return Result<void>::err(
                    "Failed to read block for connect: " + block_result.error());
            }

            auto connect_result = connect_block(block_result.value(), idx);
            if (!connect_result) {
                return connect_result;
            }
        }
    }

    return Result<void>::ok();
}

// ---------------------------------------------------------------------------
// connect_block
// ---------------------------------------------------------------------------
// Applies a block to the UTXO set and updates chain state.
//
// Steps:
//   1. Sum input values for fee computation
//   2. Compute allowed subsidy via adaptive emission
//   3. Verify coinbase output <= (subsidy + fees)
//   4. Verify transaction inputs (signatures, scripts)
//   5. Build undo data, spend inputs, add outputs
//   6. Store undo data keyed by block hash
//   7. Update chain tip and active chain vector
//
// Reward formula:
//   allowed_coinbase = subsidy + fees
//
// Fork choice rule (PoT):
//   height > tip.height            => candidate wins (longer chain)
//   height == tip.height           => lower val_loss wins (tiebreak)
// ---------------------------------------------------------------------------
Result<void> CChainState::connect_block(
    const primitives::CBlock& block, CBlockIndex* pindex)
{
    // 1. Sum fees from non-coinbase transactions
    int64_t total_fees = 0;
    for (size_t i = 1; i < block.vtx.size(); ++i) {
        const auto& tx = block.vtx[i];
        int64_t tx_in_value = 0;
        for (const auto& txin : tx->vin()) {
            Coin coin;
            if (!coins_cache_.get_coin(txin.prevout, coin)) {
                return Result<void>::err("bad-txns-inputs-missingorspent");
            }
            tx_in_value += coin.out.value;
        }
        const int64_t tx_out_value = tx->get_value_out();
        total_fees += tx_in_value - tx_out_value;
    }

    // 2. Compute allowed subsidy.
    consensus::EmissionState emission{};  // TODO: track cumulative emission
    const auto allowed = consensus::compute_block_reward(
        pindex->header.height, emission, params_);
    const int64_t max_coinbase = allowed.total() + total_fees;

    // 3. Verify coinbase amount.
    //    allowed_coinbase = subsidy + fees
    if (!block.vtx.empty() && block.vtx[0]->is_coinbase()) {
        if (block.vtx[0]->get_value_out() > max_coinbase) {
            return Result<void>::err("bad-cb-amount");
        }
    }

    // 4. Verify inputs for non-coinbase transactions
    for (size_t i = 1; i < block.vtx.size(); ++i) {
        consensus::ValidationState tx_state;
        if (!consensus::check_inputs(*block.vtx[i], coins_cache_, tx_state)) {
            return Result<void>::err(tx_state.reject_reason);
        }
    }

    // 5. Build undo data and update UTXO set
    std::vector<Coin> undo_coins;

    for (size_t i = 0; i < block.vtx.size(); ++i) {
        const auto& tx = block.vtx[i];

        // Spend inputs (skip coinbase), saving spent coins for undo
        if (!tx->is_coinbase()) {
            for (const auto& txin : tx->vin()) {
                Coin spent;
                coins_cache_.spend_coin(txin.prevout, &spent);
                undo_coins.push_back(std::move(spent));
            }
        }

        // Add outputs
        for (uint32_t j = 0; j < tx->vout().size(); ++j) {
            if (tx->vout()[j].value < 0) continue;
            primitives::COutPoint outpoint(tx->txid(), j);
            Coin coin(tx->vout()[j], pindex->height,
                      tx->is_coinbase(), pindex->val_loss);
            coins_cache_.add_coin(outpoint, std::move(coin));
        }
    }

    // 6. Store undo data keyed by block hash
    block_undo_data_[pindex->block_hash] = std::move(undo_coins);

    // 7. Update chain state
    pindex->status = CBlockIndex::FULLY_VALIDATED;
    tip_ = pindex;

    if (pindex->height >= static_cast<int>(active_chain_.size())) {
        active_chain_.resize(static_cast<size_t>(pindex->height) + 1);
    }
    active_chain_[static_cast<size_t>(pindex->height)] = pindex;

    // Truncate any stale entries beyond current height
    active_chain_.resize(static_cast<size_t>(pindex->height) + 1);

    coins_cache_.set_best_block(pindex->block_hash);

    on_block_connected.emit(pindex);
    on_new_tip.emit(pindex);

    return Result<void>::ok();
}

// ---------------------------------------------------------------------------
// disconnect_block
// ---------------------------------------------------------------------------
// Reverses a block's effects on the UTXO set using stored undo data.
//
// Steps:
//   1. Read block from disk
//   2. Look up undo data for this block
//   3. Remove outputs (reverse order)
//   4. Restore spent inputs from undo data
//   5. Update chain state
// ---------------------------------------------------------------------------
Result<void> CChainState::disconnect_block(CBlockIndex* pindex) {
    // 1. Read block from disk
    DiskBlockPos dpos;
    dpos.file_number = pindex->file_number;
    dpos.pos = pindex->data_pos;

    auto block_result = storage_->read_block(dpos);
    if (!block_result) {
        return Result<void>::err(
            "Failed to read block for disconnect: " + block_result.error());
    }
    const auto& block = block_result.value();

    // 2. Look up undo data for this block
    const auto undo_it = block_undo_data_.find(pindex->block_hash);
    const bool have_undo = (undo_it != block_undo_data_.end());

    // 3. Remove outputs (reverse order)
    for (auto it = block.vtx.rbegin(); it != block.vtx.rend(); ++it) {
        const auto& tx = *it;
        for (uint32_t j = 0; j < tx->vout().size(); ++j) {
            primitives::COutPoint outpoint(tx->txid(), j);
            coins_cache_.spend_coin(outpoint);
        }
    }

    // 4. Restore spent inputs from undo data
    if (have_undo) {
        size_t undo_idx = 0;
        for (size_t i = 0; i < block.vtx.size(); ++i) {
            const auto& tx = block.vtx[i];
            if (tx->is_coinbase()) continue;
            for (const auto& txin : tx->vin()) {
                if (undo_idx < undo_it->second.size()) {
                    coins_cache_.add_coin(txin.prevout,
                                          undo_it->second[undo_idx], true);
                    ++undo_idx;
                }
            }
        }
        block_undo_data_.erase(undo_it);
    }

    // 5. Update chain state
    tip_ = pindex->prev;
    if (tip_) {
        active_chain_.resize(static_cast<size_t>(tip_->height) + 1);
    } else {
        active_chain_.clear();
    }

    on_block_disconnected.emit(pindex);

    return Result<void>::ok();
}

// -- Fork choice ------------------------------------------------------------

// ---------------------------------------------------------------------------
// find_best_tip
// ---------------------------------------------------------------------------
// Scans the entire block index for the best candidate tip using the PoT
// fork-choice rule (longest chain, then lowest val_loss as tiebreak).
// ---------------------------------------------------------------------------
CBlockIndex* CChainState::find_best_tip() const {
    CBlockIndex* best = tip_;
    for (const auto& [hash, idx] : block_index_) {
        if (idx->status >= CBlockIndex::TREE_VALID) {
            if (!best || is_better_tip(idx.get(), best)) {
                best = idx.get();
            }
        }
    }
    return best;
}

// ---------------------------------------------------------------------------
// is_better_tip
// ---------------------------------------------------------------------------
// Compares two chain tips using the PoT fork-choice rule.
//
//   height > current.height   => candidate wins (longer chain)
//   height == current.height  => lower val_loss wins (tiebreak)
// ---------------------------------------------------------------------------
bool CChainState::is_better_tip(const CBlockIndex* candidate,
                                const CBlockIndex* current) const {
    if (!current) return true;

    // Primary: longer chain wins (more cumulative training work)
    if (candidate->height > current->height) return true;
    if (candidate->height < current->height) return false;

    // Tiebreak at same height: lower val_loss wins
    return candidate->val_loss < current->val_loss;
}

} // namespace rnet::chain
