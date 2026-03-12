#include "chain/chainstate.h"

#include "consensus/block_verify.h"
#include "consensus/genesis.h"
#include "consensus/tx_verify.h"
#include "core/logging.h"

namespace rnet::chain {

CChainState::CChainState(const consensus::ConsensusParams& params,
                         std::unique_ptr<CCoinsView> coins_view,
                         std::unique_ptr<BlockStorage> storage)
    : params_(params)
    , coins_view_(std::move(coins_view))
    , coins_cache_(coins_view_.get())
    , storage_(std::move(storage))
{}

CChainState::~CChainState() = default;

Result<void> CChainState::load_genesis() {
    LOCK(cs_main_);

    auto genesis = consensus::create_genesis_block(params_);
    auto hash = genesis.hash();

    // Check if genesis already loaded
    if (block_index_.count(hash)) {
        return Result<void>::ok();
    }

    auto idx = std::make_unique<CBlockIndex>(
        static_cast<const primitives::CBlockHeader&>(genesis));

    idx->status = CBlockIndex::FULLY_VALIDATED;
    idx->chain_tx = static_cast<int64_t>(genesis.vtx.size());

    auto* raw = idx.get();
    block_index_[hash] = std::move(idx);

    // Write genesis to disk
    auto pos_result = storage_->write_block(genesis);
    if (!pos_result) {
        return Result<void>::err("Failed to write genesis: " + pos_result.error());
    }
    raw->file_number = pos_result.value().file_number;
    raw->data_pos = pos_result.value().pos;

    // Set as tip
    tip_ = raw;
    active_chain_.clear();
    active_chain_.push_back(raw);

    coins_cache_.set_best_block(hash);

    // Add genesis coinbase outputs to UTXO set
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

Result<CBlockIndex*> CChainState::accept_block_header(
    const primitives::CBlockHeader& header)
{
    LOCK(cs_main_);

    auto hash = header.hash();

    // Already known?
    auto it = block_index_.find(hash);
    if (it != block_index_.end()) {
        return Result<CBlockIndex*>::ok(it->second.get());
    }

    // Find parent
    auto parent_it = block_index_.find(header.prev_hash);
    if (parent_it == block_index_.end()) {
        return Result<CBlockIndex*>::err(
            "Unknown parent block: " + header.prev_hash.to_hex());
    }
    auto* parent = parent_it->second.get();

    // Validate header against parent
    consensus::ValidationState state;
    if (!consensus::check_block_header(header, parent->header, state, params_)) {
        return Result<CBlockIndex*>::err(
            "Invalid block header: " + state.reject_reason);
    }

    // Insert into block index
    auto idx = std::make_unique<CBlockIndex>(header);
    idx->prev = parent;
    idx->status = CBlockIndex::HEADER_VALID;

    auto* raw = idx.get();
    block_index_[hash] = std::move(idx);

    return Result<CBlockIndex*>::ok(raw);
}

Result<CBlockIndex*> CChainState::accept_block(
    const primitives::CBlock& block)
{
    // Accept header first
    auto header_result = accept_block_header(
        static_cast<const primitives::CBlockHeader&>(block));
    if (!header_result) {
        return header_result;
    }

    CBlockIndex* pindex = header_result.value();

    LOCK(cs_main_);

    // Already fully validated?
    if (pindex->status >= CBlockIndex::FULLY_VALIDATED) {
        return Result<CBlockIndex*>::ok(pindex);
    }

    // Context-free block validation
    consensus::ValidationState state;
    if (!consensus::check_block(block, state, params_)) {
        return Result<CBlockIndex*>::err(
            "Invalid block: " + state.reject_reason);
    }

    // Write to disk
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

    // Try to activate best chain (may connect this block or trigger reorg)
    auto activate_result = activate_best_chain();
    if (!activate_result) {
        return Result<CBlockIndex*>::err(activate_result.error());
    }

    return Result<CBlockIndex*>::ok(pindex);
}

CBlockIndex* CChainState::tip() const {
    return tip_;
}

CBlockIndex* CChainState::lookup_block_index(
    const rnet::uint256& hash) const
{
    auto it = block_index_.find(hash);
    if (it == block_index_.end()) return nullptr;
    return it->second.get();
}

int CChainState::height() const {
    return tip_ ? tip_->height : -1;
}

std::vector<CBlockIndex*> CChainState::get_active_chain() const {
    return active_chain_;
}

CBlockIndex* CChainState::get_block_by_height(int h) const {
    if (h < 0 || h >= static_cast<int>(active_chain_.size())) {
        return nullptr;
    }
    return active_chain_[static_cast<size_t>(h)];
}

bool CChainState::is_on_active_chain(const rnet::uint256& hash) const {
    auto* idx = lookup_block_index(hash);
    if (!idx) return false;
    if (idx->height < 0 || idx->height >= static_cast<int>(active_chain_.size())) {
        return false;
    }
    return active_chain_[static_cast<size_t>(idx->height)] == idx;
}

CBlockIndex* CChainState::insert_block_index(const rnet::uint256& hash) {
    auto it = block_index_.find(hash);
    if (it != block_index_.end()) return it->second.get();

    auto idx = std::make_unique<CBlockIndex>();
    idx->block_hash = hash;
    auto* raw = idx.get();
    block_index_[hash] = std::move(idx);
    return raw;
}

Result<void> CChainState::activate_best_chain() {
    CBlockIndex* best = find_best_tip();
    if (!best || best == tip_) {
        return Result<void>::ok();
    }

    // Find the fork point
    CBlockIndex* fork = tip_;
    CBlockIndex* walk = best;

    // Walk both chains back to the same height
    while (fork && walk && fork->height > walk->height) {
        fork = fork->prev;
    }
    while (fork && walk && walk->height > fork->height) {
        walk = walk->prev;
    }
    while (fork && walk && fork != walk) {
        fork = fork->prev;
        walk = walk->prev;
    }
    CBlockIndex* fork_point = fork;

    // Disconnect blocks from old tip to fork point
    if (tip_ && fork_point) {
        CBlockIndex* disconnect = tip_;
        while (disconnect != fork_point) {
            auto result = disconnect_block(disconnect);
            if (!result) return result;
            disconnect = disconnect->prev;
        }
    }

    // Build the list of blocks to connect
    std::vector<CBlockIndex*> to_connect;
    for (CBlockIndex* idx = best; idx != fork_point; idx = idx->prev) {
        to_connect.push_back(idx);
    }
    std::reverse(to_connect.begin(), to_connect.end());

    // Connect blocks from fork point to new best tip
    for (auto* idx : to_connect) {
        // Read the block from disk
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

    return Result<void>::ok();
}

Result<void> CChainState::connect_block(
    const primitives::CBlock& block, CBlockIndex* pindex)
{
    // Update UTXO set: spend inputs, add outputs
    for (size_t i = 0; i < block.vtx.size(); ++i) {
        const auto& tx = block.vtx[i];

        // Spend inputs (skip coinbase)
        if (!tx->is_coinbase()) {
            for (const auto& txin : tx->vin()) {
                coins_cache_.spend_coin(txin.prevout);
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

    // Update chain state
    pindex->status = CBlockIndex::FULLY_VALIDATED;
    tip_ = pindex;

    // Update active_chain_ vector
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

Result<void> CChainState::disconnect_block(CBlockIndex* pindex) {
    // Read block from disk to undo its effects
    DiskBlockPos dpos;
    dpos.file_number = pindex->file_number;
    dpos.pos = pindex->data_pos;

    auto block_result = storage_->read_block(dpos);
    if (!block_result) {
        return Result<void>::err(
            "Failed to read block for disconnect: " + block_result.error());
    }
    const auto& block = block_result.value();

    // Reverse: remove outputs, then re-add spent inputs
    // (simplified — a full implementation would use undo data)
    for (auto it = block.vtx.rbegin(); it != block.vtx.rend(); ++it) {
        const auto& tx = *it;
        // Remove outputs
        for (uint32_t j = 0; j < tx->vout().size(); ++j) {
            primitives::COutPoint outpoint(tx->txid(), j);
            coins_cache_.spend_coin(outpoint);
        }
        // Re-adding inputs would require undo data (not implemented in stub)
    }

    tip_ = pindex->prev;
    if (tip_) {
        active_chain_.resize(static_cast<size_t>(tip_->height) + 1);
    } else {
        active_chain_.clear();
    }

    on_block_disconnected.emit(pindex);

    return Result<void>::ok();
}

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

bool CChainState::is_better_tip(const CBlockIndex* candidate,
                                const CBlockIndex* current) const {
    if (!current) return true;
    // PoT fork choice: lower val_loss wins
    if (candidate->val_loss < current->val_loss) return true;
    // Tiebreak: higher height (more work done)
    if (candidate->val_loss == current->val_loss &&
        candidate->height > current->height) return true;
    return false;
}

}  // namespace rnet::chain
