/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "cache_manager/direct_cache_manager.h"

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <vector>

#include "cache_manager/cache_manager_factory.h"
#include "test.h"
#include "utils/logger.h"
#include "utils/memory_allocator.h"
#include "utils/status.h"

#include "cache_manager/prefix_cache_manager_test_helper.h"

using namespace ksana_llm;

// ---------------------------------------------------------------------------
// Test fixture shared by all DirectCacheManager tests.
// ---------------------------------------------------------------------------
class DirectCacheManagerTest : public testing::Test {
 protected:
  // Build a DirectCacheManager without SWA (sliding_window_size == 0).
  void SetUpNoSwa(size_t device_block_num = 64, size_t block_token_num = 16, size_t mamba_block_num = 0) {
    device_block_num_ = device_block_num;
    block_token_num_ = block_token_num;

    BlockAllocatorGroupConfig group_config;
    group_config.devices = {0};
    group_config.device_block_num = device_block_num_;
    group_config.host_block_num = 0;
    group_config.block_size = block_token_num_ * sizeof(float);

    block_allocator_creation_fn_ = MakeFakedCreationFn();
    context_ = std::make_shared<Context>();
    memory_allocator_ = std::make_shared<FakedMemoryAllocator>();

    block_allocator_group_ = std::make_shared<FakedBlockAllocatorGroup>(group_config, memory_allocator_, context_,
                                                                        block_allocator_creation_fn_);

    if (mamba_block_num > 0) {
      BlockAllocatorGroupConfig mamba_group_config;
      mamba_group_config.devices = {0};
      mamba_group_config.device_block_num = mamba_block_num;
      mamba_group_config.host_block_num = 0;
      mamba_group_config.block_size = sizeof(float);

      auto mamba_creation_fn = [](MemoryLocation location, size_t block_num, size_t block_size, int rank,
                                  std::shared_ptr<MemoryAllocatorInterface> memory_allocator,
                                  std::shared_ptr<Context> context) -> std::shared_ptr<BlockAllocatorInterface> {
        return std::make_shared<FakedBlockAllocator>(location, block_num, block_size, rank,
                                                     memory_allocator, context);
      };
      mamba_block_allocator_group_ = std::make_shared<FakedBlockAllocatorGroup>(mamba_group_config, memory_allocator_,
                                                                                context_, mamba_creation_fn);
    }

    CacheManagerConfig cfg;
    cfg.block_token_num = block_token_num_;
    cfg.tensor_para_size = 1;
    cfg.sliding_window_size = 0;  // SWA disabled

    cache_manager_ =
        std::make_unique<DirectCacheManager>(cfg, block_allocator_group_, nullptr, mamba_block_allocator_group_);
    cache_manager_->InitializeCachedBlocks();
  }

  // Build a DirectCacheManager with SWA enabled.
  // sliding_window_size is expressed in tokens; swa_block_num is the number of
  // SWA blocks pre-allocated in the faked SWA allocator group.
  void SetUpWithSwa(size_t sliding_window_size, size_t swa_block_num = 32, size_t device_block_num = 64,
                    size_t block_token_num = 16) {
    device_block_num_ = device_block_num;
    block_token_num_ = block_token_num;
    sliding_window_size_ = sliding_window_size;

    BlockAllocatorGroupConfig kv_group_config;
    kv_group_config.devices = {0};
    kv_group_config.device_block_num = device_block_num_;
    kv_group_config.host_block_num = 0;
    kv_group_config.block_size = block_token_num_ * sizeof(float);

    BlockAllocatorGroupConfig swa_group_config;
    swa_group_config.devices = {0};
    swa_group_config.device_block_num = swa_block_num;
    swa_group_config.host_block_num = 0;
    swa_group_config.block_size = block_token_num_ * sizeof(float);

    block_allocator_creation_fn_ = MakeFakedCreationFn();
    context_ = std::make_shared<Context>();
    memory_allocator_ = std::make_shared<FakedMemoryAllocator>();

    block_allocator_group_ = std::make_shared<FakedBlockAllocatorGroup>(kv_group_config, memory_allocator_, context_,
                                                                        block_allocator_creation_fn_);
    swa_block_allocator_group_ = std::make_shared<FakedBlockAllocatorGroup>(swa_group_config, memory_allocator_,
                                                                            context_, block_allocator_creation_fn_);

    CacheManagerConfig cfg;
    cfg.block_token_num = block_token_num_;
    cfg.tensor_para_size = 1;
    cfg.sliding_window_size = sliding_window_size_;

    cache_manager_ = std::make_unique<DirectCacheManager>(cfg, block_allocator_group_, swa_block_allocator_group_);
    cache_manager_->InitializeCachedBlocks();
  }

  // Helper: register a request and return its id.
  int64_t RegisterRequest(int64_t req_id) {
    size_t shared_block_num = 0, unique_block_num = 0, shared_token_num = 0;
    std::vector<int> dummy_tokens;
    cache_manager_->GetRequestPrefixBlockNumber(req_id, dummy_tokens, block_token_num_, shared_block_num,
                                                unique_block_num, shared_token_num);
    return req_id;
  }

 private:
  static BlockAllocatorCreationFunc MakeFakedCreationFn() {
    return [](MemoryLocation location, size_t block_num, size_t block_size, int rank,
              std::shared_ptr<MemoryAllocatorInterface> memory_allocator,
              std::shared_ptr<Context> context) -> std::shared_ptr<BlockAllocatorInterface> {
      return std::make_shared<FakedBlockAllocator>(location, block_num, block_size, rank, memory_allocator, context);
    };
  }

 protected:
  size_t device_block_num_ = 64;
  size_t block_token_num_ = 16;
  size_t sliding_window_size_ = 0;

  std::shared_ptr<Context> context_;
  std::shared_ptr<MemoryAllocatorInterface> memory_allocator_;
  BlockAllocatorCreationFunc block_allocator_creation_fn_;
  std::shared_ptr<BlockAllocatorGroupInterface> block_allocator_group_;
  std::shared_ptr<BlockAllocatorGroupInterface> swa_block_allocator_group_;
  std::shared_ptr<BlockAllocatorGroupInterface> mamba_block_allocator_group_;

  std::unique_ptr<DirectCacheManager> cache_manager_;
};

// ===========================================================================
// Tests without SWA (sliding_window_size == 0)
// ===========================================================================

// AllocateRequestBlocks: basic allocation returns correct block ids and leaves
// swa outputs empty when SWA is disabled.
TEST_F(DirectCacheManagerTest, NoSwa_AllocateRequestBlocks_Basic) {
  SetUpNoSwa(/*device_block_num=*/16, /*block_token_num=*/16);

  const int64_t req_id = 1;
  RegisterRequest(req_id);

  std::vector<int> req_block_ids;
  std::vector<int> swa_req_block_ids;
  std::vector<size_t> swa_block_idx_offsets;  // sentinel: stays empty when SWA disabled

  int mamba_kv_cache_block_id = -1;
  Status s = cache_manager_->AllocateRequestBlocks(req_id, /*block_num=*/3, req_block_ids, swa_req_block_ids,
                                                   swa_block_idx_offsets, mamba_kv_cache_block_id);
  ASSERT_TRUE(s.OK()) << s.GetMessage();

  // Three KV blocks allocated.
  EXPECT_EQ(req_block_ids.size(), 3u);

  // SWA outputs must be empty when SWA is disabled.
  EXPECT_TRUE(swa_req_block_ids.empty());
  EXPECT_TRUE(swa_block_idx_offsets.empty());  // not written by the function

  // Free block count decreased by 3.
  EXPECT_EQ(cache_manager_->GetUsableBlockNumber(), 16u - 3u);
}

TEST_F(DirectCacheManagerTest, NoSwa_AllocateRequestBlocks_AllocatesMambaBlock) {
  SetUpNoSwa(/*device_block_num=*/16, /*block_token_num=*/16, /*mamba_block_num=*/2);

  const int64_t req_id = RegisterRequest(/*req_id=*/1);
  std::vector<int> req_block_ids;
  std::vector<int> swa_req_block_ids;
  std::vector<size_t> swa_block_idx_offsets;

  int mamba_block_id = -1;
  Status status = cache_manager_->AllocateRequestBlocks(req_id, /*block_num=*/1, req_block_ids, swa_req_block_ids,
                                                        swa_block_idx_offsets, mamba_block_id);

  ASSERT_TRUE(status.OK()) << status.GetMessage();
  ASSERT_EQ(req_block_ids.size(), 1u);
  EXPECT_GT(mamba_block_id, 0);
  EXPECT_EQ(mamba_block_allocator_group_->GetDeviceBlockAllocator(0)->GetFreeBlockNumber(), 1u);
}

TEST_F(DirectCacheManagerTest, NoSwa_AllocateRequestBlocks_ReturnsExistingMambaBlock) {
  SetUpNoSwa(/*device_block_num=*/16, /*block_token_num=*/16, /*mamba_block_num=*/2);

  const int64_t req_id = RegisterRequest(/*req_id=*/1);
  std::vector<int> req_block_ids;
  std::vector<int> swa_req_block_ids;
  std::vector<size_t> swa_block_idx_offsets;
  int mamba_block_id = -1;
  ASSERT_TRUE(cache_manager_->AllocateRequestBlocks(req_id, /*block_num=*/1, req_block_ids, swa_req_block_ids,
                                                    swa_block_idx_offsets, mamba_block_id)
                  .OK());
  const int first_mamba_kv_cache_block_id = mamba_block_id;

  int second_mamba_block_id = -1;
  Status status = cache_manager_->AllocateRequestBlocks(req_id, /*block_num=*/0, req_block_ids, swa_req_block_ids,
                                                        swa_block_idx_offsets, second_mamba_block_id);

  ASSERT_TRUE(status.OK()) << status.GetMessage();
  EXPECT_EQ(second_mamba_block_id, first_mamba_kv_cache_block_id);
  EXPECT_EQ(mamba_block_allocator_group_->GetDeviceBlockAllocator(0)->GetFreeBlockNumber(), 1u);
}

TEST_F(DirectCacheManagerTest, NoSwa_AllocateRequestBlocks_FaPrecheckFailureDoesNotConsumeMambaBlock) {
  SetUpNoSwa(/*device_block_num=*/1, /*block_token_num=*/16, /*mamba_block_num=*/1);

  const int64_t req_id = RegisterRequest(/*req_id=*/1);
  std::vector<int> req_block_ids;
  std::vector<int> swa_req_block_ids;
  std::vector<size_t> swa_block_idx_offsets;

  int mamba_block_id = -1;
  Status status = cache_manager_->AllocateRequestBlocks(req_id, /*block_num=*/2, req_block_ids, swa_req_block_ids,
                                                        swa_block_idx_offsets, mamba_block_id);

  EXPECT_FALSE(status.OK());
  EXPECT_TRUE(req_block_ids.empty());
  EXPECT_EQ(mamba_block_id, -1);
  EXPECT_EQ(mamba_block_allocator_group_->GetDeviceBlockAllocator(0)->GetFreeBlockNumber(), 1u);
}

TEST_F(DirectCacheManagerTest, NoSwa_AllocateRequestBlocks_MambaFailureDoesNotConsumeFaBlock) {
  SetUpNoSwa(/*device_block_num=*/2, /*block_token_num=*/16, /*mamba_block_num=*/1);

  const int64_t first_req_id = RegisterRequest(/*req_id=*/1);
  std::vector<int> first_req_block_ids;
  std::vector<int> first_swa_req_block_ids;
  std::vector<size_t> first_swa_block_idx_offsets;
  int first_mamba_kv_cache_block_id = -1;
  ASSERT_TRUE(cache_manager_
                  ->AllocateRequestBlocks(first_req_id, /*block_num=*/1, first_req_block_ids, first_swa_req_block_ids,
                                          first_swa_block_idx_offsets, first_mamba_kv_cache_block_id)
                  .OK());

  const int64_t second_req_id = RegisterRequest(/*req_id=*/2);
  std::vector<int> second_req_block_ids;
  std::vector<int> second_swa_req_block_ids;
  std::vector<size_t> second_swa_block_idx_offsets;
  const size_t usable_fa_blocks_before = cache_manager_->GetUsableBlockNumber();

  int second_mamba_block_id = -1;
  Status status = cache_manager_->AllocateRequestBlocks(second_req_id, /*block_num=*/1, second_req_block_ids,
                                                        second_swa_req_block_ids, second_swa_block_idx_offsets,
                                                        second_mamba_block_id);

  EXPECT_FALSE(status.OK());
  EXPECT_TRUE(second_req_block_ids.empty());
  EXPECT_EQ(second_mamba_block_id, -1);
  EXPECT_EQ(cache_manager_->GetUsableBlockNumber(), usable_fa_blocks_before);
}

TEST_F(DirectCacheManagerTest, NoSwa_DestroyFinishedRequest_FreesMambaBlock) {
  SetUpNoSwa(/*device_block_num=*/2, /*block_token_num=*/16, /*mamba_block_num=*/1);

  const int64_t req_id = RegisterRequest(/*req_id=*/1);
  std::vector<int> req_block_ids;
  std::vector<int> swa_req_block_ids;
  std::vector<size_t> swa_block_idx_offsets;
  int mamba_kv_cache_block_id = -1;
  ASSERT_TRUE(cache_manager_->AllocateRequestBlocks(req_id, /*block_num=*/1, req_block_ids, swa_req_block_ids,
                                                    swa_block_idx_offsets, mamba_kv_cache_block_id)
                  .OK());
  ASSERT_EQ(mamba_block_allocator_group_->GetDeviceBlockAllocator(0)->GetFreeBlockNumber(), 0u);

  cache_manager_->DestroyFinishedRequest(req_id);

  EXPECT_EQ(mamba_block_allocator_group_->GetDeviceBlockAllocator(0)->GetFreeBlockNumber(), 1u);
}

TEST_F(DirectCacheManagerTest, Factory_InjectsMambaAllocatorIntoDirectCacheManager) {
  SetUpNoSwa();

  BlockAllocatorGroupConfig fa_group_config;
  fa_group_config.devices = {0};
  fa_group_config.device_block_num = 2;
  fa_group_config.host_block_num = 0;
  fa_group_config.block_size = 16 * sizeof(float);
  auto fa_allocator_group = std::make_shared<FakedBlockAllocatorGroup>(fa_group_config, memory_allocator_, context_,
                                                                       block_allocator_creation_fn_);

  BlockAllocatorGroupConfig mamba_group_config;
  mamba_group_config.devices = {0};
  mamba_group_config.device_block_num = 1;
  mamba_group_config.host_block_num = 0;
  mamba_group_config.block_size = sizeof(float);
  auto mamba_allocator_group = std::make_shared<FakedBlockAllocatorGroup>(mamba_group_config, memory_allocator_,
                                                                          context_, block_allocator_creation_fn_);

  CacheManagerConfig config;
  config.block_token_num = 16;
  config.tensor_para_size = 1;
  config.enable_prefix_caching = false;
  auto cache_manager =
      CacheManagerFactory::CreateCacheManager(config, fa_allocator_group, nullptr, mamba_allocator_group);
  cache_manager->InitializeCachedBlocks();

  size_t shared_block_num = 0;
  size_t unique_block_num = 0;
  size_t shared_token_num = 0;
  ASSERT_TRUE(cache_manager
                  ->GetRequestPrefixBlockNumber(/*req_id=*/1, /*input_token_ids=*/{}, /*check_token_num=*/16,
                                                shared_block_num, unique_block_num, shared_token_num)
                  .OK());

  std::vector<int> req_block_ids;
  std::vector<int> swa_req_block_ids;
  std::vector<size_t> swa_block_idx_offsets;
  int mamba_block_id = -1;
  Status status = cache_manager->AllocateRequestBlocks(/*req_id=*/1, /*block_num=*/1, req_block_ids, swa_req_block_ids,
                                                       swa_block_idx_offsets, mamba_block_id);

  ASSERT_TRUE(status.OK()) << status.GetMessage();
  EXPECT_EQ(mamba_block_id, 1);
}

TEST_F(DirectCacheManagerTest, Factory_RejectsMambaAllocatorWithPrefixCaching) {
  SetUpNoSwa(/*device_block_num=*/2, /*block_token_num=*/16, /*mamba_block_num=*/1);

  CacheManagerConfig config;
  config.block_token_num = 16;
  config.tensor_para_size = 1;
  config.enable_prefix_caching = true;

  EXPECT_THROW(
      CacheManagerFactory::CreateCacheManager(config, block_allocator_group_, nullptr, mamba_block_allocator_group_),
      std::runtime_error);
}

// AllocateRequestBlocks: requesting more blocks than available returns an error.
TEST_F(DirectCacheManagerTest, NoSwa_AllocateRequestBlocks_InsufficientBlocks) {
  SetUpNoSwa(/*device_block_num=*/4, /*block_token_num=*/16);

  const int64_t req_id = 1;
  RegisterRequest(req_id);

  std::vector<int> req_block_ids;
  std::vector<int> swa_req_block_ids;
  std::vector<size_t> swa_block_idx_offsets;

  int mamba_kv_cache_block_id = -1;
  Status s = cache_manager_->AllocateRequestBlocks(req_id, /*block_num=*/5, req_block_ids, swa_req_block_ids,
                                                   swa_block_idx_offsets, mamba_kv_cache_block_id);
  EXPECT_FALSE(s.OK());
}

// UpdateRequestTokens: without SWA the function succeeds and does not touch
// the swa output parameters.
TEST_F(DirectCacheManagerTest, NoSwa_UpdateRequestTokens_NoSideEffects) {
  SetUpNoSwa(/*device_block_num=*/16, /*block_token_num=*/16);

  const int64_t req_id = 1;
  RegisterRequest(req_id);

  // Allocate 2 blocks first.
  std::vector<int> req_block_ids;
  std::vector<int> swa_req_block_ids;
  std::vector<size_t> swa_block_idx_offsets;
  int mamba_kv_cache_block_id = -1;
  ASSERT_TRUE(cache_manager_->AllocateRequestBlocks(req_id, 2, req_block_ids, swa_req_block_ids, swa_block_idx_offsets,
                                                    mamba_kv_cache_block_id).OK());

  // Simulate 32 tokens cached (fills both blocks).
  std::vector<int> kvcached_token_ids(32);
  bool block_merged = false;
  std::vector<int> out_req_block_ids;
  std::vector<int> out_swa_block_ids;
  std::vector<size_t> out_swa_offsets;  // stays empty when SWA disabled

  Status s = cache_manager_->UpdateRequestTokens(req_id, kvcached_token_ids, /*shareable_kvcache_token_num=*/32,
                                                 out_req_block_ids, block_merged, out_swa_block_ids, out_swa_offsets);
  ASSERT_TRUE(s.OK()) << s.GetMessage();

  // SWA outputs must remain empty when SWA is disabled.
  EXPECT_TRUE(out_swa_block_ids.empty());
  EXPECT_TRUE(out_swa_offsets.empty());
}

// DestroyFinishedRequest: blocks are returned to the free pool.
TEST_F(DirectCacheManagerTest, NoSwa_DestroyFinishedRequest_FreesBlocks) {
  SetUpNoSwa(/*device_block_num=*/8, /*block_token_num=*/16);

  const int64_t req_id = 1;
  RegisterRequest(req_id);

  std::vector<int> req_block_ids;
  std::vector<int> swa_req_block_ids;
  std::vector<size_t> swa_block_idx_offsets;
  int mamba_kv_cache_block_id = -1;
  ASSERT_TRUE(cache_manager_->AllocateRequestBlocks(req_id, 4, req_block_ids, swa_req_block_ids, swa_block_idx_offsets,
                                                    mamba_kv_cache_block_id).OK());
  EXPECT_EQ(cache_manager_->GetUsableBlockNumber(), 4u);

  cache_manager_->DestroyFinishedRequest(req_id);
  EXPECT_EQ(cache_manager_->GetUsableBlockNumber(), 8u);
}

// ===========================================================================
// Tests with SWA enabled
// ===========================================================================

// AllocateRequestBlocks: SWA blocks are allocated alongside KV blocks.
// The number of SWA blocks equals block_num (one SWA block per KV block on
// initial allocation when kvcached_token_num == 0).
TEST_F(DirectCacheManagerTest, Swa_AllocateRequestBlocks_SwaBlocksAllocated) {
  // sliding_window_size = 32 tokens = 2 blocks (block_token_num=16).
  SetUpWithSwa(/*sliding_window_size=*/32, /*swa_block_num=*/16, /*device_block_num=*/16, /*block_token_num=*/16);

  const int64_t req_id = 1;
  RegisterRequest(req_id);

  std::vector<int> req_block_ids;
  std::vector<int> swa_req_block_ids;
  std::vector<size_t> swa_block_idx_offsets;

  int mamba_kv_cache_block_id = -1;
  Status s = cache_manager_->AllocateRequestBlocks(req_id, /*block_num=*/2, req_block_ids, swa_req_block_ids,
                                                   swa_block_idx_offsets, mamba_kv_cache_block_id);
  ASSERT_TRUE(s.OK()) << s.GetMessage();

  // Two KV blocks allocated.
  EXPECT_EQ(req_block_ids.size(), 2u);

  // Two SWA blocks allocated (one per KV block on initial allocation).
  EXPECT_EQ(swa_req_block_ids.size(), 2u);

  // Initial offsets are {0, 1}.
  EXPECT_EQ(swa_block_idx_offsets.size(), 2u);
  EXPECT_EQ(swa_block_idx_offsets[0], 0u);

  // SWA allocator consumed 2 blocks.
  EXPECT_EQ(swa_block_allocator_group_->GetDeviceBlockAllocator(0)->GetFreeBlockNumber(), 16u - 2u);
}

// AllocateRequestBlocks: SWA block ids are distinct from KV block ids.
TEST_F(DirectCacheManagerTest, Swa_AllocateRequestBlocks_SwaIdsDistinctFromKv) {
  SetUpWithSwa(/*sliding_window_size=*/32, /*swa_block_num=*/16, /*device_block_num=*/16, /*block_token_num=*/16);

  const int64_t req_id = 1;
  RegisterRequest(req_id);

  std::vector<int> req_block_ids;
  std::vector<int> swa_req_block_ids;
  std::vector<size_t> swa_block_idx_offsets;
  int mamba_kv_cache_block_id = -1;
  ASSERT_TRUE(cache_manager_->AllocateRequestBlocks(req_id, 2, req_block_ids, swa_req_block_ids, swa_block_idx_offsets,
                                                    mamba_kv_cache_block_id).OK());

  // KV and SWA block ids come from separate allocators; they should not overlap
  // because FakedBlockAllocator assigns ids starting from 0 independently.
  // The important invariant is that swa_req_block_ids is non-empty and has the
  // expected size.
  EXPECT_EQ(swa_req_block_ids.size(), 2u);
  EXPECT_EQ(req_block_ids.size(), 2u);
}

// UpdateRequestTokens: when the window has not yet slid, no SWA blocks are
// released and the offset stays at 0.
TEST_F(DirectCacheManagerTest, Swa_UpdateRequestTokens_NoSlideYet) {
  // sliding_window_size = 32 tokens = 2 blocks.
  SetUpWithSwa(/*sliding_window_size=*/32, /*swa_block_num=*/16, /*device_block_num=*/16, /*block_token_num=*/16);

  const int64_t req_id = 1;
  RegisterRequest(req_id);

  std::vector<int> req_block_ids;
  std::vector<int> swa_req_block_ids;
  std::vector<size_t> swa_block_idx_offsets;
  int mamba_kv_cache_block_id = -1;
  ASSERT_TRUE(cache_manager_->AllocateRequestBlocks(req_id, 2, req_block_ids, swa_req_block_ids, swa_block_idx_offsets,
                                                    mamba_kv_cache_block_id).OK());

  // 32 tokens cached — exactly fills the window; left boundary = token 1,
  // block 0 is still inside the window.
  std::vector<int> kvcached_token_ids(32);
  bool block_merged = false;
  std::vector<int> out_req_block_ids;
  std::vector<int> out_swa_block_ids;
  std::vector<size_t> out_swa_offsets;

  Status s = cache_manager_->UpdateRequestTokens(req_id, kvcached_token_ids, /*shareable_kvcache_token_num=*/32,
                                                 out_req_block_ids, block_merged, out_swa_block_ids, out_swa_offsets);
  ASSERT_TRUE(s.OK()) << s.GetMessage();

  // No blocks slid out yet: offset still 0, both SWA blocks still present.
  EXPECT_EQ(out_swa_offsets[0], 0u);
  EXPECT_EQ(out_swa_block_ids.size(), 2u);

  // SWA allocator still has 14 free blocks (16 - 2 allocated).
  EXPECT_EQ(swa_block_allocator_group_->GetDeviceBlockAllocator(0)->GetFreeBlockNumber(), 14u);
}

// UpdateRequestTokens: when the window slides past the first block, that block
// is freed and the offset advances to 1.
// Setup: sliding_window_size=32 (2 blocks), block_token_num=16.
// Allocate 3 KV blocks (covers tokens [0,47]) and 3 SWA blocks.
// After 47 tokens cached: swa_left_token = 47 - 32 + 1 = 16 → swa_left_block = 16/16 = 1.
// Block 0 (tokens [0,15]) falls out of the window; offset advances to 1.
TEST_F(DirectCacheManagerTest, Swa_UpdateRequestTokens_SlideReleasesOldestBlock) {
  // sliding_window_size = 32 tokens = 2 blocks (block_token_num=16).
  SetUpWithSwa(/*sliding_window_size=*/32, /*swa_block_num=*/16, /*device_block_num=*/16, /*block_token_num=*/16);

  const int64_t req_id = 1;
  RegisterRequest(req_id);

  // Allocate 3 KV blocks (tokens [0,47]) and 3 SWA blocks.
  std::vector<int> req_block_ids;
  std::vector<int> swa_req_block_ids;
  std::vector<size_t> swa_block_idx_offsets;
  int mamba_kv_cache_block_id = -1;
  ASSERT_TRUE(cache_manager_->AllocateRequestBlocks(req_id, 3, req_block_ids, swa_req_block_ids, swa_block_idx_offsets,
                                                    mamba_kv_cache_block_id).OK());
  EXPECT_EQ(swa_req_block_ids.size(), 3u);

  // 47 tokens cached: exactly fills 3 blocks.
  // swa_left_token = 47 - 32 + 1 = 16  →  swa_left_block = 16 / 16 = 1.
  // Block 0 (tokens [0,15]) falls out of the window.
  std::vector<int> kvcached_token_ids(47);
  bool block_merged = false;
  std::vector<int> out_req_block_ids;
  std::vector<int> out_swa_block_ids;
  std::vector<size_t> out_swa_offsets;

  Status s = cache_manager_->UpdateRequestTokens(req_id, kvcached_token_ids, /*shareable_kvcache_token_num=*/47,
                                                 out_req_block_ids, block_merged, out_swa_block_ids, out_swa_offsets);
  ASSERT_TRUE(s.OK()) << s.GetMessage();

  // Block 0 was released; first offset advanced to 1.
  EXPECT_EQ(out_swa_offsets[0], 1u);
  // Blocks 1 and 2 remain in the window.
  EXPECT_EQ(out_swa_block_ids.size(), 2u);

  // The freed SWA block is back in the allocator's free pool: 16 - 3 + 1 = 14.
  EXPECT_EQ(swa_block_allocator_group_->GetDeviceBlockAllocator(0)->GetFreeBlockNumber(), 14u);
}

// UpdateRequestTokens: multiple consecutive slides release multiple blocks, then a
// subsequent allocation + update at a non-boundary token count leaves two SWA blocks.
//
// Phase 1 — boundary case (shareable_kvcache_token_num is an exact multiple of block_token_num):
//   Allocate 4 KV blocks (tokens [0,63]) + 4 SWA blocks.
//   Update with shareable=64: swa_left_token = 64-16+1 = 49 → swa_left_block = 3.
//   Blocks 0,1,2 freed; offset=3; 1 SWA block (block 3) remains.
//
// Phase 2 — non-boundary case (shareable_kvcache_token_num is NOT a multiple):
//   Allocate 1 more KV block (block 4) + 1 more SWA block (block 4).
//   Update with shareable=65: swa_left_token = 65-16+1 = 50 → swa_left_block = 50/16 = 3.
//   swa_left_block (3) == current offset (3) → no additional slide.
//   2 SWA blocks remain (blocks 3 and 4).
TEST_F(DirectCacheManagerTest, Swa_UpdateRequestTokens_MultipleSlides) {
  // sliding_window_size = 16 tokens = 1 block (block_token_num=16).
  SetUpWithSwa(/*sliding_window_size=*/16, /*swa_block_num=*/16, /*device_block_num=*/16, /*block_token_num=*/16);

  const int64_t req_id = 1;
  RegisterRequest(req_id);

  // --- Phase 1: allocate 4 blocks, update at exact boundary ---

  // Allocate 4 KV blocks (tokens [0,63]); initial SWA allocation covers blocks [0,3].
  std::vector<int> req_block_ids;
  std::vector<int> swa_req_block_ids;
  std::vector<size_t> swa_block_idx_offsets;
  int mamba_kv_cache_block_id = -1;
  ASSERT_TRUE(cache_manager_->AllocateRequestBlocks(req_id, 4, req_block_ids, swa_req_block_ids, swa_block_idx_offsets,
                                                    mamba_kv_cache_block_id).OK());
  EXPECT_EQ(swa_req_block_ids.size(), 4u);

  // 64 tokens cached: exactly fills 4 blocks.
  // swa_left_token = 64 - 16 + 1 = 49 → swa_left_block = 49 / 16 = 3.
  // Blocks 0, 1, 2 fall out of the window.
  {
    std::vector<int> kvcached_token_ids(64);
    bool block_merged = false;
    std::vector<int> out_req_block_ids;
    std::vector<int> out_swa_block_ids;
    std::vector<size_t> out_swa_offsets;

    Status s = cache_manager_->UpdateRequestTokens(req_id, kvcached_token_ids, /*shareable_kvcache_token_num=*/64,
                                                   out_req_block_ids, block_merged, out_swa_block_ids, out_swa_offsets);
    ASSERT_TRUE(s.OK()) << s.GetMessage();

    EXPECT_EQ(out_swa_offsets[0], 3u);
    EXPECT_EQ(out_swa_block_ids.size(), 1u);  // only block 3 remains

    // 3 SWA blocks freed → 16 - 4 + 3 = 15 free.
    EXPECT_EQ(swa_block_allocator_group_->GetDeviceBlockAllocator(0)->GetFreeBlockNumber(), 15u);
  }

  // --- Phase 2: allocate 1 more block, update at non-boundary token count ---

  // Allocate 1 more KV block (total KV = 5).
  // swa_right_block = cached_blocks.size() = 5; current_swa_end = offset(3) + remaining(1) = 4.
  // Loop i=4: allocates 1 new SWA block (block 4).
  // After this allocation: SWA blocks are [3, 4], offset=3.
  std::vector<int> req_block_ids2;
  std::vector<int> swa_req_block_ids2;
  std::vector<size_t> swa_block_idx_offsets2;
  int mamba_kv_cache_block_id_2 = -1;
  ASSERT_TRUE(cache_manager_->AllocateRequestBlocks(req_id, 1, req_block_ids2, swa_req_block_ids2,
                                                    swa_block_idx_offsets2, mamba_kv_cache_block_id_2)
                  .OK());
  // After this allocation: SWA blocks are [3, 4], offsets={3, 4}.
  EXPECT_EQ(swa_block_idx_offsets2[0], 3u);
  EXPECT_EQ(swa_req_block_ids2.size(), 2u);

  // 65 tokens cached (non-boundary: 65 is not a multiple of 16).
  // swa_left_token = 65 - 16 + 1 = 50 → swa_left_block = 50 / 16 = 3.
  // swa_left_block (3) == current offset (3) → no additional slide.
  // Both SWA blocks 3 and 4 remain in the window.
  {
    std::vector<int> kvcached_token_ids(65);
    bool block_merged = false;
    std::vector<int> out_req_block_ids;
    std::vector<int> out_swa_block_ids;
    std::vector<size_t> out_swa_offsets;

    Status s = cache_manager_->UpdateRequestTokens(req_id, kvcached_token_ids, /*shareable_kvcache_token_num=*/65,
                                                   out_req_block_ids, block_merged, out_swa_block_ids, out_swa_offsets);
    ASSERT_TRUE(s.OK()) << s.GetMessage();

    // No additional slide: first offset stays at 3.
    EXPECT_EQ(out_swa_offsets[0], 3u);
    // Two SWA blocks remain (blocks 3 and 4).
    EXPECT_EQ(out_swa_block_ids.size(), 2u);

    // Phase 2 allocated 1 more SWA block → 15 - 1 = 14 free; no slide in this update.
    EXPECT_EQ(swa_block_allocator_group_->GetDeviceBlockAllocator(0)->GetFreeBlockNumber(), 14u);
  }
}

// DestroyFinishedRequest: all SWA blocks are freed when the request finishes.
TEST_F(DirectCacheManagerTest, Swa_DestroyFinishedRequest_FreesAllSwaBlocks) {
  SetUpWithSwa(/*sliding_window_size=*/32, /*swa_block_num=*/16, /*device_block_num=*/16, /*block_token_num=*/16);

  const int64_t req_id = 1;
  RegisterRequest(req_id);

  std::vector<int> req_block_ids;
  std::vector<int> swa_req_block_ids;
  std::vector<size_t> swa_block_idx_offsets;
  int mamba_kv_cache_block_id = -1;
  ASSERT_TRUE(cache_manager_->AllocateRequestBlocks(req_id, 3, req_block_ids, swa_req_block_ids, swa_block_idx_offsets,
                                                    mamba_kv_cache_block_id).OK());

  // 3 SWA blocks consumed.
  EXPECT_EQ(swa_block_allocator_group_->GetDeviceBlockAllocator(0)->GetFreeBlockNumber(), 13u);

  cache_manager_->DestroyFinishedRequest(req_id);

  // All 3 SWA blocks returned.
  EXPECT_EQ(swa_block_allocator_group_->GetDeviceBlockAllocator(0)->GetFreeBlockNumber(), 16u);

  // KV blocks also returned.
  EXPECT_EQ(cache_manager_->GetUsableBlockNumber(), 16u);
}

// Two concurrent requests each get independent SWA block sets.
TEST_F(DirectCacheManagerTest, Swa_TwoRequests_IndependentSwaBlocks) {
  SetUpWithSwa(/*sliding_window_size=*/32, /*swa_block_num=*/16, /*device_block_num=*/16, /*block_token_num=*/16);

  const int64_t req_a = 1, req_b = 2;
  RegisterRequest(req_a);
  RegisterRequest(req_b);

  std::vector<int> kv_a, swa_a;
  std::vector<size_t> off_a;
  int mamba_kv_cache_block_id_a = -1;
  ASSERT_TRUE(cache_manager_->AllocateRequestBlocks(req_a, 2, kv_a, swa_a, off_a, mamba_kv_cache_block_id_a).OK());

  std::vector<int> kv_b, swa_b;
  std::vector<size_t> off_b;
  int mamba_kv_cache_block_id_b = -1;
  ASSERT_TRUE(cache_manager_->AllocateRequestBlocks(req_b, 2, kv_b, swa_b, off_b, mamba_kv_cache_block_id_b).OK());

  // Each request got 2 SWA blocks; 4 total consumed.
  EXPECT_EQ(swa_a.size(), 2u);
  EXPECT_EQ(swa_b.size(), 2u);
  EXPECT_EQ(swa_block_allocator_group_->GetDeviceBlockAllocator(0)->GetFreeBlockNumber(), 12u);

  // Destroy req_a; its SWA blocks are freed.
  cache_manager_->DestroyFinishedRequest(req_a);
  EXPECT_EQ(swa_block_allocator_group_->GetDeviceBlockAllocator(0)->GetFreeBlockNumber(), 14u);

  // Destroy req_b; all SWA blocks freed.
  cache_manager_->DestroyFinishedRequest(req_b);
  EXPECT_EQ(swa_block_allocator_group_->GetDeviceBlockAllocator(0)->GetFreeBlockNumber(), 16u);
}

// AllocateRequestBlocks on an unknown req_id returns an error.
TEST_F(DirectCacheManagerTest, NoSwa_AllocateRequestBlocks_UnknownReqId) {
  SetUpNoSwa();

  std::vector<int> req_block_ids;
  std::vector<int> swa_req_block_ids;
  std::vector<size_t> swa_block_idx_offsets;

  // req_id 999 was never registered via GetRequestPrefixBlockNumber.
  int mamba_kv_cache_block_id = -1;
  Status s =
      cache_manager_->AllocateRequestBlocks(999, 1, req_block_ids, swa_req_block_ids, swa_block_idx_offsets,
                                            mamba_kv_cache_block_id);
  EXPECT_FALSE(s.OK());
}

// UpdateRequestTokens on an unknown req_id returns an error.
TEST_F(DirectCacheManagerTest, NoSwa_UpdateRequestTokens_UnknownReqId) {
  SetUpNoSwa();

  std::vector<int> kvcached_token_ids(16);
  bool block_merged = false;
  std::vector<int> out_req_block_ids;
  std::vector<int> out_swa_block_ids;
  std::vector<size_t> out_swa_offsets;

  Status s = cache_manager_->UpdateRequestTokens(999, kvcached_token_ids, 16, out_req_block_ids, block_merged,
                                                 out_swa_block_ids, out_swa_offsets);
  EXPECT_FALSE(s.OK());
}

// ---------------------------------------------------------------------------
// DropUnbackedSwaBlocks
// ---------------------------------------------------------------------------

TEST_F(DirectCacheManagerTest, DropUnbackedSwaBlocks_NoSwaIsNoop) {
  SetUpNoSwa();
  std::vector<int> swa_req_block_ids = {1};
  std::vector<size_t> swa_block_idx_offsets = {0};
  EXPECT_TRUE(
      cache_manager_
          ->DropUnbackedSwaBlocks(/*req_id=*/1, /*keep_start_block_idx=*/0, swa_req_block_ids, swa_block_idx_offsets)
          .OK());
  EXPECT_EQ(swa_req_block_ids.size(), 1u);
}

TEST_F(DirectCacheManagerTest, DropUnbackedSwaBlocks_UnknownReqFails) {
  SetUpWithSwa(/*sliding_window_size=*/32, /*swa_block_num=*/16, /*device_block_num=*/16, /*block_token_num=*/16);
  std::vector<int> swa_req_block_ids;
  std::vector<size_t> swa_block_idx_offsets;
  EXPECT_FALSE(cache_manager_
                   ->DropUnbackedSwaBlocks(/*req_id=*/12345, /*keep_start_block_idx=*/0, swa_req_block_ids,
                                           swa_block_idx_offsets)
                   .OK());
}

TEST_F(DirectCacheManagerTest, DropUnbackedSwaBlocks_FreesBlocksBeforeKeepStart) {
  SetUpWithSwa(/*sliding_window_size=*/32, /*swa_block_num=*/16, /*device_block_num=*/16, /*block_token_num=*/16);

  const int64_t req_id = RegisterRequest(1);
  std::vector<int> req_block_ids;
  std::vector<int> swa_req_block_ids;
  std::vector<size_t> swa_block_idx_offsets;
  int mamba_kv_cache_block_id = -1;
  ASSERT_TRUE(cache_manager_
                  ->AllocateRequestBlocks(req_id, /*block_num=*/4, req_block_ids, swa_req_block_ids,
                                          swa_block_idx_offsets, mamba_kv_cache_block_id)
                  .OK());
  ASSERT_EQ(swa_req_block_ids.size(), 4u);
  EXPECT_EQ(swa_block_allocator_group_->GetDeviceBlockAllocator(0)->GetFreeBlockNumber(), 12u);

  // 只保留 idx >= 2 的 SWA block，前两块应释放回池。
  ASSERT_TRUE(cache_manager_
                  ->DropUnbackedSwaBlocks(req_id, /*keep_start_block_idx=*/2, swa_req_block_ids, swa_block_idx_offsets)
                  .OK());

  ASSERT_EQ(swa_req_block_ids.size(), 2u);
  ASSERT_EQ(swa_block_idx_offsets.size(), 2u);
  EXPECT_EQ(swa_block_idx_offsets[0], 2u);
  EXPECT_EQ(swa_block_idx_offsets[1], 3u);
  EXPECT_EQ(swa_block_allocator_group_->GetDeviceBlockAllocator(0)->GetFreeBlockNumber(), 14u);
}

TEST_F(DirectCacheManagerTest, DropUnbackedSwaBlocks_KeepAllWhenStartIsZero) {
  SetUpWithSwa(/*sliding_window_size=*/32, /*swa_block_num=*/16, /*device_block_num=*/16, /*block_token_num=*/16);

  const int64_t req_id = RegisterRequest(2);
  std::vector<int> req_block_ids;
  std::vector<int> swa_req_block_ids;
  std::vector<size_t> swa_block_idx_offsets;
  int mamba_kv_cache_block_id = -1;
  ASSERT_TRUE(cache_manager_
                  ->AllocateRequestBlocks(req_id, /*block_num=*/3, req_block_ids, swa_req_block_ids,
                                          swa_block_idx_offsets, mamba_kv_cache_block_id)
                  .OK());
  ASSERT_EQ(swa_req_block_ids.size(), 3u);
  const size_t free_before = swa_block_allocator_group_->GetDeviceBlockAllocator(0)->GetFreeBlockNumber();

  ASSERT_TRUE(cache_manager_
                  ->DropUnbackedSwaBlocks(req_id, /*keep_start_block_idx=*/0, swa_req_block_ids, swa_block_idx_offsets)
                  .OK());

  EXPECT_EQ(swa_req_block_ids.size(), 3u);
  EXPECT_EQ(swa_block_allocator_group_->GetDeviceBlockAllocator(0)->GetFreeBlockNumber(), free_before);
}
