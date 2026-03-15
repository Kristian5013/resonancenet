#pragma once

#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "chain/block_index.h"
#include "chain/coins.h"
#include "chain/storage.h"
#include "consensus/params.h"
#include "consensus/validation.h"
#include "core/error.h"
#include "core/signal.h"
#include "core/sync.h"
#include "primitives/block.h"

namespace rnet::chain {

/// CChainState — manages the active chain, block index, and UTXO set.
/// Fork choice rule: lowest val_loss at the tip wins.
class CChainState {
public:
    explicit CChainState(const consensus::ConsensusParams& params,
                         std::unique_ptr<CCoinsView> coins_view,
                         std::unique_ptr<BlockStorage> storage);
    ~CChainState();

    // Non-copyable
    CChainState(const CChainState&) = delete;
    CChainState& operator=(const CChainState&) = delete;

    /// Initialize with genesis block
    Result<void> load_genesis();

    /// Reload block index from disk.
    /// Reads every stored block, rebuilds block_index_, and sets the tip
    /// to the highest fully-validated block on the best chain.
    Result<void> load_block_index();

    /// Accept a new block header into the block index.
    /// Validates header against parent. Does not validate transactions.
    Result<CBlockIndex*> accept_block_header(
        const primitives::CBlockHeader& header);

    /// Accept a full block (header + transactions).
    /// Validates and connects if it extends the best chain.
    Result<CBlockIndex*> accept_block(const primitives::CBlock& block);

    /// Get the current chain tip
    CBlockIndex* tip() const;

    /// Get the block index for a given hash
    CBlockIndex* lookup_block_index(const rnet::uint256& hash) const;

    /// Get the active chain height
    int height() const;

    /// Get the UTXO view (read-only)
    const CCoinsView& coins_view() const { return *coins_view_; }

    /// Get a mutable reference to the coins cache
    CCoinsViewCache& coins_cache() { return coins_cache_; }

    /// Get the block storage
    BlockStorage& storage() { return *storage_; }

    /// Get the active chain (tip down to genesis)
    std::vector<CBlockIndex*> get_active_chain() const;

    /// Get block index by height (on the active chain)
    CBlockIndex* get_block_by_height(int h) const;

    /// Check if a block hash is on the active chain
    bool is_on_active_chain(const rnet::uint256& hash) const;

    /// Get the consensus params
    const consensus::ConsensusParams& params() const { return params_; }

    /// Signals
    core::Signal<const CBlockIndex*> on_new_tip;       ///< New best tip
    core::Signal<const CBlockIndex*> on_block_connected;
    core::Signal<const CBlockIndex*> on_block_disconnected;

    /// Get the entire block index map (for sync)
    const std::unordered_map<rnet::uint256, std::unique_ptr<CBlockIndex>>&
    block_index() const { return block_index_; }

    /// Number of known block headers
    size_t block_index_size() const { return block_index_.size(); }

private:
    const consensus::ConsensusParams& params_;
    std::unique_ptr<CCoinsView> coins_view_;
    CCoinsViewCache coins_cache_;
    std::unique_ptr<BlockStorage> storage_;

    mutable core::Mutex cs_main_;

    /// Block index: hash -> CBlockIndex
    std::unordered_map<rnet::uint256, std::unique_ptr<CBlockIndex>> block_index_;

    /// Active chain tip
    CBlockIndex* tip_ = nullptr;

    /// Active chain (indexed by height for O(1) access)
    std::vector<CBlockIndex*> active_chain_;

    /// Insert a block index entry
    CBlockIndex* insert_block_index(const rnet::uint256& hash);

    /// Activate the best chain (may trigger reorg)
    /// @param new_block If provided, used instead of reading from disk
    ///                  when the block to connect matches this hash.
    Result<void> activate_best_chain(
        const primitives::CBlock* new_block = nullptr);

    /// Connect a block to the active chain (update UTXO set)
    Result<void> connect_block(const primitives::CBlock& block,
                               CBlockIndex* pindex);

    /// Disconnect a block from the active chain
    Result<void> disconnect_block(CBlockIndex* pindex);

    /// Find the best tip candidate by PoT fork-choice rule
    CBlockIndex* find_best_tip() const;

    /// Compare two chain tips: returns true if candidate is better
    bool is_better_tip(const CBlockIndex* candidate,
                       const CBlockIndex* current) const;
};

}  // namespace rnet::chain
