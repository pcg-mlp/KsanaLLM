/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "configure/model_config.h"
#include "utils/indexer_skip_config.h"
#include "test.h"

namespace ksana_llm {

class IndexerSkipConfigTest : public testing::Test {
 protected:
  DsaConfig MakeConfig(uint32_t skip_offset, uint32_t topk_freq, bool share_for_mtp = false) {
    DsaConfig cfg;
    cfg.index_skip_topk_offset = skip_offset;
    cfg.index_topk_freq = topk_freq;
    cfg.index_share_for_mtp_iteration = share_for_mtp;
    return cfg;
  }
};

// Layers before index_skip_topk_offset are kept (not skipped).
TEST_F(IndexerSkipConfigTest, KeepEarlyMainLayers) {
  const DsaConfig cfg = MakeConfig(/*skip_offset=*/3, /*topk_freq=*/2);
  const int num_layer = 10;
  EXPECT_FALSE(ShouldSkip(0, num_layer, cfg));
  EXPECT_FALSE(ShouldSkip(1, num_layer, cfg));
  EXPECT_FALSE(ShouldSkip(2, num_layer, cfg));
}

// Within main layers (>= offset, < num_layer), every index_topk_freq-th layer is kept.
TEST_F(IndexerSkipConfigTest, MainLayerStridePattern) {
  const DsaConfig cfg = MakeConfig(/*skip_offset=*/2, /*topk_freq=*/3);
  const int num_layer = 10;
  // (layer + 1 - 2) % 3 == 0 means kept: layer = 4, 7
  EXPECT_TRUE(ShouldSkip(2, num_layer, cfg));
  EXPECT_TRUE(ShouldSkip(3, num_layer, cfg));
  EXPECT_FALSE(ShouldSkip(4, num_layer, cfg));
  EXPECT_TRUE(ShouldSkip(5, num_layer, cfg));
  EXPECT_TRUE(ShouldSkip(6, num_layer, cfg));
  EXPECT_FALSE(ShouldSkip(7, num_layer, cfg));
}

// First mtp layer (layer == num_layer) is always kept.
TEST_F(IndexerSkipConfigTest, KeepFirstMtpLayer) {
  const DsaConfig cfg = MakeConfig(/*skip_offset=*/1, /*topk_freq=*/1, /*share_for_mtp=*/true);
  EXPECT_FALSE(ShouldSkip(/*layer=*/8, /*num_layer=*/8, cfg));
}

// Subsequent mtp layers follow index_share_for_mtp_iteration.
TEST_F(IndexerSkipConfigTest, MtpFollowupLayersFollowShareFlag) {
  const int num_layer = 8;
  const DsaConfig share = MakeConfig(/*skip_offset=*/1, /*topk_freq=*/1, /*share_for_mtp=*/true);
  const DsaConfig no_share = MakeConfig(/*skip_offset=*/1, /*topk_freq=*/1, /*share_for_mtp=*/false);
  EXPECT_TRUE(ShouldSkip(num_layer + 1, num_layer, share));
  EXPECT_TRUE(ShouldSkip(num_layer + 2, num_layer, share));
  EXPECT_FALSE(ShouldSkip(num_layer + 1, num_layer, no_share));
  EXPECT_FALSE(ShouldSkip(num_layer + 2, num_layer, no_share));
}

// GetIndexerSkipLayers returns exactly the layers ShouldSkip flags as true.
TEST_F(IndexerSkipConfigTest, GetIndexerSkipLayersMatchesShouldSkip) {
  const DsaConfig cfg = MakeConfig(/*skip_offset=*/2, /*topk_freq=*/3, /*share_for_mtp=*/true);
  const int num_layer = 10;
  const int nextn_num_layer = 3;
  // Main: kept = {0,1,4,7}; skipped = {2,3,5,6,8,9}
  // Mtp: layer 10 kept; layers 11,12 skipped (share_for_mtp=true)
  const std::unordered_set<int> expected = {2, 3, 5, 6, 8, 9, 11, 12};
  EXPECT_EQ(GetIndexerSkipLayers(num_layer, nextn_num_layer, cfg), expected);
}

// No mtp layers and topk_freq=1 with skip_offset=0 means every main layer is kept.
TEST_F(IndexerSkipConfigTest, GetIndexerSkipLayersEmptyWhenAllKept) {
  const DsaConfig cfg = MakeConfig(/*skip_offset=*/0, /*topk_freq=*/1);
  EXPECT_TRUE(GetIndexerSkipLayers(/*num_layer=*/5, /*nextn_num_layer=*/0, cfg).empty());
}

}  // namespace ksana_llm
