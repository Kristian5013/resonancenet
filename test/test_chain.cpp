// Tests for chain module: block index, chainstate basics

#include "test_framework.h"

#include "chain/block_index.h"
#include "consensus/params.h"
#include "core/types.h"
#include "primitives/block_header.h"

using namespace rnet;
using namespace rnet::chain;
using namespace rnet::primitives;

// ─── CBlockIndex tests ─────────────────────────────────────────────

TEST(block_index_default) {
    CBlockIndex idx;
    ASSERT_EQ(idx.height, 0);
    ASSERT_EQ(idx.prev, nullptr);
    ASSERT_EQ(idx.status, CBlockIndex::HEADER_VALID);
}

TEST(block_index_from_header) {
    CBlockHeader header;
    header.height = 100;
    header.version = 1;
    header.d_model = 400;
    header.n_layers = 7;
    header.val_loss = 3.5f;
    header.timestamp = 1700000000;

    CBlockIndex idx(header);
    ASSERT_EQ(idx.height, 100);
    ASSERT_EQ(idx.val_loss, 3.5f);
    ASSERT_EQ(idx.d_model, uint32_t(400));
    ASSERT_EQ(idx.n_layers, uint32_t(7));
    ASSERT_EQ(idx.timestamp, uint64_t(1700000000));
}

TEST(block_index_get_block_hash) {
    CBlockHeader header;
    header.height = 1;
    header.timestamp = 12345;

    CBlockIndex idx(header);
    auto hash = idx.get_block_hash();
    ASSERT_FALSE(hash.is_zero());

    // Should be cached — same result
    auto hash2 = idx.get_block_hash();
    ASSERT_EQ(hash, hash2);
}

TEST(block_index_chain_links) {
    CBlockIndex genesis;
    genesis.height = 0;
    genesis.prev = nullptr;

    CBlockHeader h1;
    h1.height = 1;
    CBlockIndex idx1(h1);
    idx1.prev = &genesis;

    CBlockHeader h2;
    h2.height = 2;
    CBlockIndex idx2(h2);
    idx2.prev = &idx1;

    ASSERT_EQ(idx2.prev->height, 1);
    ASSERT_EQ(idx2.prev->prev->height, 0);
    ASSERT_EQ(idx2.prev->prev->prev, nullptr);
}

TEST(block_index_get_ancestor) {
    // Build a chain of 5 blocks
    CBlockIndex blocks[5];
    for (int i = 0; i < 5; ++i) {
        blocks[i].height = i;
        blocks[i].prev = (i > 0) ? &blocks[i - 1] : nullptr;
    }

    // Get ancestor at various heights
    auto* ancestor = blocks[4].get_ancestor(0);
    ASSERT_TRUE(ancestor != nullptr);
    ASSERT_EQ(ancestor->height, 0);

    ancestor = blocks[4].get_ancestor(2);
    ASSERT_TRUE(ancestor != nullptr);
    ASSERT_EQ(ancestor->height, 2);

    ancestor = blocks[4].get_ancestor(4);
    ASSERT_TRUE(ancestor != nullptr);
    ASSERT_EQ(ancestor->height, 4);

    // Height beyond chain tip should return nullptr
    ancestor = blocks[4].get_ancestor(5);
    ASSERT_TRUE(ancestor == nullptr);
}

TEST(block_index_is_ancestor_of) {
    CBlockIndex blocks[3];
    for (int i = 0; i < 3; ++i) {
        blocks[i].height = i;
        blocks[i].prev = (i > 0) ? &blocks[i - 1] : nullptr;
    }

    ASSERT_TRUE(blocks[0].is_ancestor_of(&blocks[2]));
    ASSERT_TRUE(blocks[1].is_ancestor_of(&blocks[2]));
    ASSERT_FALSE(blocks[2].is_ancestor_of(&blocks[0]));
}

TEST(block_index_get_locator) {
    // Build a chain of 20 blocks
    CBlockIndex blocks[20];
    for (int i = 0; i < 20; ++i) {
        CBlockHeader h;
        h.height = static_cast<uint64_t>(i);
        h.timestamp = static_cast<uint64_t>(1000 + i);
        blocks[i] = CBlockIndex(h);
        blocks[i].height = i;
        blocks[i].prev = (i > 0) ? &blocks[i - 1] : nullptr;
    }

    auto locator = blocks[19].get_locator();
    // Locator should have entries (exponentially spaced)
    ASSERT_FALSE(locator.empty());
    // First entry should be the tip's hash
    ASSERT_EQ(locator[0], blocks[19].get_block_hash());
}

TEST(block_index_status_progression) {
    CBlockIndex idx;
    ASSERT_EQ(idx.status, CBlockIndex::HEADER_VALID);

    idx.status = CBlockIndex::TREE_VALID;
    ASSERT_EQ(idx.status, CBlockIndex::TREE_VALID);

    idx.status = CBlockIndex::FULLY_VALIDATED;
    ASSERT_EQ(idx.status, CBlockIndex::FULLY_VALIDATED);
}
