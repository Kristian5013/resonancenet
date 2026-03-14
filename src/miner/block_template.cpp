// Copyright (c) 2024-present ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

// Own header.
#include "miner/block_template.h"

// Project headers.
#include "core/logging.h"
#include "miner/block_assembler.h"
#include "miner/difficulty.h"

namespace rnet::miner {

// ---------------------------------------------------------------------------
// create_block_template
// ---------------------------------------------------------------------------
// Assembles a candidate block from the parent header, pending transactions,
// and miner identity.  Growth and reward are estimated conservatively since
// the actual validation loss improvement is unknown at template time.
// ---------------------------------------------------------------------------
BlockTemplate create_block_template(
    const primitives::CBlockHeader& parent_header,
    const std::vector<primitives::CTransactionRef>& txs,
    const std::vector<int64_t>& tx_fees,
    const crypto::Ed25519PublicKey& miner_pubkey,
    const consensus::EmissionState& emission,
    const consensus::ConsensusParams& params)
{
    BlockTemplate tmpl;

    // 1. Compute expected growth for this block.
    tmpl.growth = consensus::GrowthPolicy::expected_growth(parent_header);

    // 2. Compute block reward (base subsidy + recovered).
    tmpl.reward = consensus::compute_block_reward(
        parent_header.height + 1,
        emission,
        params);

    // 3. Build transaction entries with fee metadata.
    std::vector<TxEntry> entries;
    entries.reserve(txs.size());
    for (size_t i = 0; i < txs.size(); ++i) {
        TxEntry entry;
        entry.tx = txs[i];
        entry.fee = (i < tx_fees.size()) ? tx_fees[i] : 0;
        entry.weight = txs[i] ? txs[i]->get_weight() : 0;
        entry.fee_rate = (entry.weight > 0)
            ? (entry.fee * 1000 / static_cast<int64_t>(entry.weight))
            : 0;
        entries.push_back(std::move(entry));
    }

    // 4. Assemble the block.
    BlockAssembler assembler(params);
    assembler.add_transactions(entries);
    tmpl.block = assembler.assemble(parent_header, miner_pubkey,
                                     tmpl.reward, tmpl.growth);
    tmpl.total_fees = assembler.total_fees();

    LogPrint(MINING, "BlockTemplate: height=%llu, txs=%zu, fees=%lld, "
             "d_model=%u, n_layers=%u, growth_delta=%u",
             static_cast<unsigned long long>(parent_header.height + 1),
             tmpl.tx_count(),
             static_cast<long long>(tmpl.total_fees),
             tmpl.growth.new_d_model,
             tmpl.growth.new_n_layers,
             tmpl.growth.delta_d_model);

    return tmpl;
}

} // namespace rnet::miner
