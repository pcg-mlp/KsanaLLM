/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "cache_manager/prefix_cache_manager.h"

#include <gtest/gtest.h>

#include <memory>
#include <utility>

#include "runtime/request_state.h"
#include "test.h"
#include "utils/logger.h"
#include "utils/memory_allocator.h"
#include "utils/memory_utils.h"
#include "utils/singleton.h"
#include "utils/status.h"
#include "utils/string_utils.h"

#include "cache_manager/prefix_cache_manager_test_helper.h"

using namespace ksana_llm;

class PrefixCacheManagerTest : public testing::Test {
 protected:
  void SetUp() override {
    BlockAllocatorGroupConfig group_1_config;
    group_1_config.devices = {0, 1};
    group_1_config.device_block_num = device_block_num;
    group_1_config.host_block_num = host_block_num;
    group_1_config.block_size = block_token_num * 1024 * 1024;

    block_allocator_creation_fn_ = [](MemoryLocation location, size_t block_num, size_t block_size, int rank,
                                      std::shared_ptr<MemoryAllocatorInterface> memory_allocator,
                                      std::shared_ptr<Context> context) {
      return std::make_shared<FakedBlockAllocator>(location, block_num, block_size, rank, memory_allocator, context);
    };

    context_ = std::make_shared<Context>();
    memory_allocator_ = std::make_shared<FakedMemoryAllocator>();

    cache_manager_config.block_token_num = 16;
    cache_manager_config.tensor_para_size = 2;
    cache_manager_config.enable_prefix_caching = true;

    block_allocator_group_ = std::make_shared<FakedBlockAllocatorGroup>(group_1_config, memory_allocator_, context_,
                                                                        block_allocator_creation_fn_);
    cache_manager = new PrefixCacheManager(cache_manager_config, block_allocator_group_);
    faked_token_generator = new FakedTokenGenerator();
  }

  void TearDown() override {
    delete faked_token_generator;
    delete cache_manager;
  }

  // All blocks except last should be on tree, check state of every block.
  void CheckCachedRequetBlocks(int64_t req_id) {
    size_t req_block_num = cache_manager->cached_requests_[req_id]->cached_blocks.size();
    for (size_t i = 0; i < req_block_num; ++i) {
      PrefixCachedBlock* cb = cache_manager->cached_requests_[req_id]->cached_blocks[i];
      PrefixCachedBlock* cb_prev =
          (i == 0) ? cache_manager->root_cached_block_ : cache_manager->cached_requests_[req_id]->cached_blocks[i - 1];

      if (i != req_block_num - 1) {
        EXPECT_EQ(cb->parent, cb_prev);
        EXPECT_EQ(cb->parent->children[cb->token_ids], cb);
        EXPECT_EQ(cb->active_requests.size(), 1);
        EXPECT_EQ(cb->inactive_requests.size(), 0);
        EXPECT_EQ(cb->is_shareable, true);
        EXPECT_EQ(cb->is_device_location, true);
      } else {
        EXPECT_EQ(cb->parent, nullptr);
        EXPECT_EQ(cb->active_requests.size(), 0);
        EXPECT_EQ(cb->inactive_requests.size(), 0);
        EXPECT_EQ(cb->is_shareable, false);
        EXPECT_EQ(cb->is_device_location, true);
      }
    }
  }

 protected:
  BlockManagerConfig block_manager_config;

  CacheManagerConfig cache_manager_config;

  PrefixCacheManager* cache_manager = nullptr;

  FakedTokenGenerator* faked_token_generator = nullptr;

  std::shared_ptr<Context> context_ = nullptr;
  std::shared_ptr<MemoryAllocatorInterface> memory_allocator_ = nullptr;
  BlockAllocatorCreationFunc block_allocator_creation_fn_ = nullptr;
  std::shared_ptr<BlockAllocatorGroupInterface> block_allocator_group_ = nullptr;

  size_t device_block_num = 100;
  size_t host_block_num = 100;

  size_t block_token_num = 16;
  size_t tensor_para_size = 2;
  size_t attn_data_parallel_size = 1;
};

TEST_F(PrefixCacheManagerTest, SingleRequestTest) {
  cache_manager->InitializeCachedBlocks();

  // All blocks should be used.
  EXPECT_EQ(block_allocator_group_->GetDeviceBlockAllocator()->GetFreeBlockNumber(), 0);
  EXPECT_EQ(cache_manager->GetUsableBlockNumber(), device_block_num);
  EXPECT_EQ(cache_manager->GetHostFreeBlockNumber(), host_block_num);

  // Create a faked request.
  std::vector<int> output_token_ids;
  faked_token_generator->GeneratePromptTokens({std::make_pair(1, 82)}, output_token_ids);

  // Check needed block num.
  int64_t req_id = 1;
  size_t shared_token_num;
  size_t shared_block_num;
  size_t unique_block_num;
  cache_manager->GetRequestPrefixBlockNumber(req_id, output_token_ids, output_token_ids.size(), shared_block_num,
                                             unique_block_num, shared_token_num);

  // Check request state.
  EXPECT_EQ(cache_manager->cached_requests_.size(), 1);
  EXPECT_EQ(cache_manager->cached_requests_[req_id]->shared_block_num, shared_block_num);
  EXPECT_EQ(cache_manager->cached_requests_[req_id]->req_state, RequestState::kWaiting);
  EXPECT_EQ(cache_manager->cached_requests_[req_id]->cached_blocks.size(), shared_block_num);

  // No shared block, all unqiue block.
  EXPECT_EQ(shared_block_num, 0);
  EXPECT_EQ(unique_block_num, (82 + block_token_num - 1) / block_token_num);

  // No block on tree.
  EXPECT_EQ(cache_manager->root_cached_block_->children.size(), 0);

  // Allocate request block.
  std::vector<int> req_block_ids;
  std::vector<int> swa_req_block_ids;
  std::vector<size_t> swa_block_idx_offset;
  Status status = cache_manager->AllocateRequestBlocks(req_id, unique_block_num, req_block_ids, swa_req_block_ids,
                                                       swa_block_idx_offset);
  EXPECT_TRUE(status.OK());

  // Check memory number.
  EXPECT_EQ(req_block_ids.size(), unique_block_num);

  // Recheck usable block num.
  EXPECT_EQ(cache_manager->GetUsableBlockNumber(), device_block_num - unique_block_num);

  // Generate new token and update request.
  faked_token_generator->GenerateOneToken(1, output_token_ids);
  bool block_merged = false;
  status = cache_manager->UpdateRequestTokens(req_id, output_token_ids, output_token_ids.size() - 1, req_block_ids,
                                              block_merged, swa_req_block_ids, swa_block_idx_offset);
  EXPECT_TRUE(status.OK());

  // All blocks except last should be on tree, check state of every block.
  CheckCachedRequetBlocks(req_id);

  // Finish request
  cache_manager->DestroyFinishedRequest(req_id);

  // All block should be usable.
  EXPECT_EQ(cache_manager->GetUsableBlockNumber(), device_block_num);
  EXPECT_EQ(cache_manager->GetHostFreeBlockNumber(), host_block_num);

  // All filled block should be reusable now.
  EXPECT_EQ(cache_manager->reusable_cached_blocks_.size(), unique_block_num - 1);

  // Forcde to free some block.
  size_t real_free_block_num;
  cache_manager->FreeCachedBlocks(2, real_free_block_num);

  // Now the reusable block should decrease by 2.
  EXPECT_EQ(cache_manager->reusable_cached_blocks_.size(), unique_block_num - 3);
  EXPECT_EQ(cache_manager->GetUsableBlockNumber(), device_block_num);
  EXPECT_EQ(cache_manager->GetHostFreeBlockNumber(), host_block_num);

  // ///////////////////////////////////////////////////////////////////////////////
  // create 2 new request, with 36 tokens same as before.
  std::vector<int> output_token_ids_2;
  faked_token_generator->GeneratePromptTokens({std::make_pair(1, 36), std::make_pair(2, 18)}, output_token_ids_2);

  // Check needed block num.
  int64_t req_id_2 = 2;
  size_t shared_token_num_2 = 0;
  size_t shared_block_num_2 = 0;
  size_t unique_block_num_2 = 0;
  cache_manager->GetRequestPrefixBlockNumber(req_id_2, output_token_ids_2, output_token_ids_2.size(),
                                             shared_block_num_2, unique_block_num_2, shared_token_num_2);

  // Should be 2 matched blocks and 2 unique block.
  EXPECT_EQ(shared_block_num_2, 2);
  EXPECT_EQ(unique_block_num_2, ((54 + block_token_num - 1) / block_token_num) - 2);

  std::vector<int> req_block_ids_2;
  std::vector<int> swa_req_block_ids_2;
  std::vector<size_t> swa_block_idx_offset_2;
  status = cache_manager->AllocateRequestBlocks(req_id_2, unique_block_num_2, req_block_ids_2, swa_req_block_ids_2,
                                                swa_block_idx_offset_2);
  EXPECT_TRUE(status.OK());

  // Create request 3.
  std::vector<int> output_token_ids_3;
  faked_token_generator->GeneratePromptTokens({std::make_pair(1, 36), std::make_pair(2, 18)}, output_token_ids_3);

  // Check needed block num.
  int64_t req_id_3 = 3;
  size_t shared_token_num_3 = 0;
  size_t shared_block_num_3 = 0;
  size_t unique_block_num_3 = 0;
  cache_manager->GetRequestPrefixBlockNumber(req_id_3, output_token_ids_3, output_token_ids_3.size(),
                                             shared_block_num_3, unique_block_num_3, shared_token_num_3);

  // Should be 2 matched blocks and 2 unique block.
  EXPECT_EQ(shared_block_num_3, 2);
  EXPECT_EQ(unique_block_num_3, ((54 + block_token_num - 1) / block_token_num) - 2);

  // Create request 5,
  std::vector<int> output_token_ids_5;
  faked_token_generator->GeneratePromptTokens({std::make_pair(1, 36), std::make_pair(2, 18)}, output_token_ids_5);

  // Check needed block num.
  int64_t req_id_5 = 5;
  size_t shared_token_num_5 = 0;
  size_t shared_block_num_5 = 0;
  size_t unique_block_num_5 = 0;
  cache_manager->GetRequestPrefixBlockNumber(req_id_5, output_token_ids_5, output_token_ids_5.size(),
                                             shared_block_num_5, unique_block_num_5, shared_token_num_5);

  // Should be 2 matched blocks and 2 unique block, used to check recalc of prefix.
  EXPECT_EQ(shared_block_num_5, 2);
  EXPECT_EQ(unique_block_num_5, ((54 + block_token_num - 1) / block_token_num) - 2);

  std::vector<int> req_block_ids_3;
  std::vector<int> swa_req_block_ids_3;
  std::vector<size_t> swa_block_idx_offset_3;
  status = cache_manager->AllocateRequestBlocks(req_id_3, unique_block_num_3, req_block_ids_3, swa_req_block_ids_3,
                                                swa_block_idx_offset_3);
  EXPECT_TRUE(status.OK());

  // Now the matched block should have 2 active requests.
  EXPECT_EQ(cache_manager->cached_requests_[req_id_2]->cached_blocks[0]->active_requests.size(), 2);
  EXPECT_EQ(cache_manager->cached_requests_[req_id_2]->cached_blocks[1]->active_requests.size(), 2);
  EXPECT_EQ(cache_manager->cached_requests_[req_id_2]->cached_blocks[2]->active_requests.size(), 0);

  EXPECT_EQ(cache_manager->cached_requests_[req_id_3]->cached_blocks[0]->active_requests.size(), 2);
  EXPECT_EQ(cache_manager->cached_requests_[req_id_3]->cached_blocks[1]->active_requests.size(), 2);
  EXPECT_EQ(cache_manager->cached_requests_[req_id_3]->cached_blocks[2]->active_requests.size(), 0);

  // block 1st and 2nd must qual
  EXPECT_EQ(cache_manager->cached_requests_[req_id_2]->cached_blocks[0],
            cache_manager->cached_requests_[req_id_3]->cached_blocks[0]);
  EXPECT_EQ(cache_manager->cached_requests_[req_id_2]->cached_blocks[1],
            cache_manager->cached_requests_[req_id_3]->cached_blocks[1]);
  // block 3rd must not equal
  EXPECT_NE(cache_manager->cached_requests_[req_id_2]->cached_blocks[2],
            cache_manager->cached_requests_[req_id_3]->cached_blocks[2]);

  // Generate new token.
  faked_token_generator->GenerateOneToken(1, output_token_ids_2);
  faked_token_generator->GenerateOneToken(1, output_token_ids_3);

  // update first request token.
  block_merged = false;
  status =
      cache_manager->UpdateRequestTokens(req_id_2, output_token_ids_2, output_token_ids_2.size() - 1, req_block_ids_2,
                                         block_merged, swa_req_block_ids_2, swa_block_idx_offset_2);
  EXPECT_TRUE(status.OK());

  // update second request token, then the 3rd block should be merged.
  block_merged = false;
  status =
      cache_manager->UpdateRequestTokens(req_id_3, output_token_ids_3, output_token_ids_3.size() - 1, req_block_ids_3,
                                         block_merged, swa_req_block_ids_3, swa_block_idx_offset_3);
  EXPECT_TRUE(status.OK());

  // Recheck prefix block of req 5, should be changed.
  cache_manager->GetRequestPrefixBlockNumber(req_id_5, output_token_ids_5, output_token_ids_5.size(),
                                             shared_block_num_5, unique_block_num_5, shared_token_num_5);

  // Should be 3 matched blocks and 1 unique block, becuase new block generated by req 2 and 3
  EXPECT_EQ(shared_block_num_5, 3);
  EXPECT_EQ(unique_block_num_5, ((54 + block_token_num - 1) / block_token_num) - 3);

  // Remove req 5.
  cache_manager->DestroyFinishedRequest(req_id_5);

  // block 3rd now must qual
  EXPECT_EQ(cache_manager->cached_requests_[req_id_2]->cached_blocks[2],
            cache_manager->cached_requests_[req_id_3]->cached_blocks[2]);

  // No reusable should be 1, and timeline should be 4.
  EXPECT_EQ(cache_manager->reusable_cached_blocks_.size(), 1);
  EXPECT_EQ(cache_manager->timed_cached_blocks_.size(), 4);

  cache_manager->DestroyFinishedRequest(req_id_2);
  cache_manager->DestroyFinishedRequest(req_id_3);

  // test: 1 block in req, 5 block in cache_manager, expect fill 4 block and append 2 new block
  constexpr int64_t req_id_6 = 6;
  std::vector<int> req_block_ids_6(1);

  cache_manager->cached_requests_[req_id_6] = std::make_unique<PrefixCachedRequest>();
  for (size_t i = 0; i < 5; ++i) {
    PrefixCachedBlock* block = new PrefixCachedBlock();
    block->memory_block_id = 0;
    cache_manager->cached_requests_[req_id_6]->cached_blocks.emplace_back(block);
  }

  std::vector<int> swa_req_block_ids_6;
  std::vector<size_t> swa_block_idx_offset_6;
  status =
      cache_manager->AllocateRequestBlocks(req_id_6, 2, req_block_ids_6, swa_req_block_ids_6, swa_block_idx_offset_6);
  EXPECT_EQ(req_block_ids_6.size(), 7);
}

TEST_F(PrefixCacheManagerTest, FlexibleCacheTest) {
  if (cache_manager != nullptr) {
    delete cache_manager;
  }
  cache_manager_config.min_flexible_cache_num = 32;
  cache_manager = new PrefixCacheManager(cache_manager_config, block_allocator_group_);
  cache_manager->InitializeCachedBlocks();

  // All blocks should be used.
  EXPECT_EQ(block_allocator_group_->GetDeviceBlockAllocator()->GetFreeBlockNumber(), 0);
  EXPECT_EQ(cache_manager->GetUsableBlockNumber(), device_block_num);
  EXPECT_EQ(cache_manager->GetHostFreeBlockNumber(), host_block_num);

  // Create a faked request.
  std::vector<int> output_token_ids;
  faked_token_generator->GeneratePromptTokens({std::make_pair(1, 89)}, output_token_ids);

  // Check needed block num.
  int64_t req_id = 1;
  size_t shared_token_num;
  size_t shared_block_num;
  size_t unique_block_num;
  cache_manager->GetRequestPrefixBlockNumber(req_id, output_token_ids, output_token_ids.size(), shared_block_num,
                                             unique_block_num, shared_token_num);

  // Allocate request block.
  std::vector<int> req_block_ids;
  std::vector<int> swa_req_block_ids;
  std::vector<size_t> swa_block_idx_offset;
  Status status = cache_manager->AllocateRequestBlocks(req_id, unique_block_num, req_block_ids, swa_req_block_ids,
                                                       swa_block_idx_offset);
  EXPECT_TRUE(status.OK());

  // Generate new token and update request.
  faked_token_generator->GenerateOneToken(1, output_token_ids);
  bool block_merged = false;
  status = cache_manager->UpdateRequestTokens(req_id, output_token_ids, output_token_ids.size() - 1, req_block_ids,
                                              block_merged, swa_req_block_ids, swa_block_idx_offset);
  EXPECT_TRUE(status.OK());
  int req_id_2 = 2;
  std::vector<int> dst_prefix16_tokens = output_token_ids;
  for (int i = block_token_num; i < dst_prefix16_tokens.size(); i++) {
    dst_prefix16_tokens[i] = (dst_prefix16_tokens[i] + 1) % faked_token_generator->GetVocabSize();
  }
  cache_manager->GetRequestPrefixBlockNumber(req_id_2, dst_prefix16_tokens, dst_prefix16_tokens.size(),
                                             shared_block_num, unique_block_num, shared_token_num);
  status = cache_manager->AllocateRequestBlocks(req_id_2, unique_block_num, req_block_ids, swa_req_block_ids,
                                                swa_block_idx_offset);
  // req 1 |prefix cache tokens|prefix_last_token|delete_token|flexible_cache_token|
  // req 2 |prefix cache tokens|prefix_last_token|flexible_cache_token|
  for (int last_token_num = 1; last_token_num < block_token_num; last_token_num++) {
    for (int delete_token_num = 1; delete_token_num <= block_token_num; delete_token_num++) {
      std::vector<int> dst_tokens;
      int src_token_index = 0;
      for (; src_token_index < shared_token_num + last_token_num; src_token_index++) {
        dst_tokens.push_back(output_token_ids[src_token_index]);
      }
      src_token_index += delete_token_num;
      for (; src_token_index < output_token_ids.size(); src_token_index++) {
        dst_tokens.push_back(output_token_ids[src_token_index]);
      }
      std::vector<FlexibleCachedCopyTask> flexible_cached_copy_tasks;
      size_t flexible_cache_len = 0;
      cache_manager->UpdateFlexibleCache(req_id_2, dst_tokens, shared_token_num, flexible_cached_copy_tasks,
                                         flexible_cache_len);
      auto hit_num = output_token_ids.size() / block_token_num * block_token_num - last_token_num - shared_token_num -
                     delete_token_num;
      hit_num = hit_num >= cache_manager_config.min_flexible_cache_num ? hit_num : 0;
      hit_num += last_token_num;
      if (shared_token_num + cache_manager_config.min_flexible_cache_num + block_token_num * 2 > dst_tokens.size()) {
        hit_num = 0;
      }
      EXPECT_EQ(hit_num, flexible_cached_copy_tasks.size());
      EXPECT_EQ(hit_num, flexible_cache_len);
    }
  }

  cache_manager->DestroyFinishedRequest(req_id);
  cache_manager->DestroyFinishedRequest(req_id_2);
}

TEST_F(PrefixCacheManagerTest, InvalidRequestTest) {
  cache_manager->InitializeCachedBlocks();
  int64_t invalid_id = 999;

  // Allocate request block.
  std::vector<int> req_block_ids;
  std::vector<int> swa_req_block_ids;
  std::vector<size_t> swa_block_idx_offset;
  size_t unique_block_num;
  Status status = cache_manager->AllocateRequestBlocks(invalid_id, unique_block_num, req_block_ids, swa_req_block_ids,
                                                       swa_block_idx_offset);
  EXPECT_EQ(status.GetCode(), RET_RUNTIME_FAILED);

  std::vector<int> output_token_ids;
  bool block_merged = false;
  status = cache_manager->UpdateRequestTokens(invalid_id, output_token_ids, 0, req_block_ids, block_merged,
                                              swa_req_block_ids, swa_block_idx_offset);
  EXPECT_EQ(status.GetCode(), RET_RUNTIME_FAILED);
}

// ============================================================================
// SWA (Sliding Window Attention) cache tests
// ============================================================================

// Test fixture that creates a PrefixCacheManager with SWA enabled.
// Configuration:
//   block_token_num = 16
//   sliding_window_size = 32  →  sliding_window_blocks = 2
//   FA blocks: device_block_num = 100
//   SWA blocks: swa_device_block_num = 50  (intentionally smaller to test eviction)
class PrefixCacheManagerSWATest : public testing::Test {
 protected:
  void SetUp() override {
    block_allocator_creation_fn_ = [](MemoryLocation location, size_t block_num, size_t block_size, int rank,
                                      std::shared_ptr<MemoryAllocatorInterface> memory_allocator,
                                      std::shared_ptr<Context> context) {
      return std::make_shared<FakedBlockAllocator>(location, block_num, block_size, rank, memory_allocator, context);
    };

    context_ = std::make_shared<Context>();
    memory_allocator_ = std::make_shared<FakedMemoryAllocator>();

    // FA block allocator group.
    BlockAllocatorGroupConfig fa_group_config;
    fa_group_config.devices = {0, 1};
    fa_group_config.device_block_num = fa_device_block_num;
    fa_group_config.host_block_num = fa_host_block_num;
    fa_group_config.block_size = block_token_num * 1024 * 1024;

    fa_block_allocator_group_ = std::make_shared<FakedBlockAllocatorGroup>(fa_group_config, memory_allocator_, context_,
                                                                           block_allocator_creation_fn_);

    // SWA block allocator group (device-only, no host swap for SWA).
    BlockAllocatorGroupConfig swa_group_config;
    swa_group_config.devices = {0};
    swa_group_config.device_block_num = swa_device_block_num;
    swa_group_config.host_block_num = 0;
    swa_group_config.block_size = block_token_num * 1024 * 1024;

    swa_block_allocator_group_ = std::make_shared<FakedBlockAllocatorGroup>(swa_group_config, memory_allocator_,
                                                                            context_, block_allocator_creation_fn_);

    cache_manager_config_.block_token_num = block_token_num;
    cache_manager_config_.tensor_para_size = 2;
    cache_manager_config_.enable_prefix_caching = true;
    cache_manager_config_.sliding_window_size = sliding_window_size;

    cache_manager_ =
        new PrefixCacheManager(cache_manager_config_, fa_block_allocator_group_, swa_block_allocator_group_);
    cache_manager_->InitializeCachedBlocks();

    faked_token_generator_ = new FakedTokenGenerator();
  }

  void TearDown() override {
    delete faked_token_generator_;
    delete cache_manager_;
  }

  // Run a complete request lifecycle: GetRequestPrefixBlockNumber →
  // AllocateRequestBlocks → UpdateRequestTokens (fills all blocks) →
  // DestroyFinishedRequest.
  // Returns the shared_block_num reported by GetRequestPrefixBlockNumber.
  size_t RunFullRequest(int64_t req_id, const std::vector<int>& token_ids) {
    size_t shared_block_num = 0, unique_block_num = 0, shared_token_num = 0;
    cache_manager_->GetRequestPrefixBlockNumber(req_id, token_ids, token_ids.size(), shared_block_num, unique_block_num,
                                                shared_token_num);

    std::vector<int> req_block_ids;
    std::vector<int> swa_req_block_ids;
    std::vector<size_t> swa_block_idx_offset;
    Status status = cache_manager_->AllocateRequestBlocks(req_id, unique_block_num, req_block_ids, swa_req_block_ids,
                                                          swa_block_idx_offset);
    EXPECT_TRUE(status.OK());

    // Simulate filling all blocks by calling UpdateRequestTokens with all tokens.
    bool block_merged = false;
    status = cache_manager_->UpdateRequestTokens(req_id, token_ids, token_ids.size(), req_block_ids, block_merged,
                                                 swa_req_block_ids, swa_block_idx_offset);
    EXPECT_TRUE(status.OK());

    cache_manager_->DestroyFinishedRequest(req_id);
    return shared_block_num;
  }

  // Run a request up to and including UpdateRequestTokens but WITHOUT calling
  // DestroyFinishedRequest.  The caller is responsible for cleanup.
  // Returns the req_id so the caller can inspect cached_requests_[req_id].
  // Output params req_block_ids / swa_req_block_ids / swa_block_idx_offset are
  // populated so the caller can pass them to subsequent API calls.
  int64_t RunRequestUntilUpdate(int64_t req_id, const std::vector<int>& token_ids, std::vector<int>& req_block_ids,
                                std::vector<int>& swa_req_block_ids, std::vector<size_t>& swa_block_idx_offset) {
    size_t shared_block_num = 0, unique_block_num = 0, shared_token_num = 0;
    cache_manager_->GetRequestPrefixBlockNumber(req_id, token_ids, token_ids.size(), shared_block_num, unique_block_num,
                                                shared_token_num);

    swa_block_idx_offset.clear();
    Status status = cache_manager_->AllocateRequestBlocks(req_id, unique_block_num, req_block_ids, swa_req_block_ids,
                                                          swa_block_idx_offset);
    EXPECT_TRUE(status.OK());

    bool block_merged = false;
    status = cache_manager_->UpdateRequestTokens(req_id, token_ids, token_ids.size(), req_block_ids, block_merged,
                                                 swa_req_block_ids, swa_block_idx_offset);
    EXPECT_TRUE(status.OK());

    return req_id;
  }

 protected:
  CacheManagerConfig cache_manager_config_;
  PrefixCacheManager* cache_manager_ = nullptr;
  FakedTokenGenerator* faked_token_generator_ = nullptr;

  std::shared_ptr<Context> context_ = nullptr;
  std::shared_ptr<MemoryAllocatorInterface> memory_allocator_ = nullptr;
  BlockAllocatorCreationFunc block_allocator_creation_fn_ = nullptr;
  std::shared_ptr<BlockAllocatorGroupInterface> fa_block_allocator_group_ = nullptr;
  std::shared_ptr<BlockAllocatorGroupInterface> swa_block_allocator_group_ = nullptr;

  // block_token_num=16, sliding_window_size=32 → sliding_window_blocks=2
  const size_t block_token_num = 16;
  const size_t sliding_window_size = 32;
  const size_t sliding_window_blocks = 2;  // ceil(32/16)

  const size_t fa_device_block_num = 100;
  const size_t fa_host_block_num = 100;
  const size_t swa_device_block_num = 50;
};

// Test: SWA blocks are allocated alongside FA blocks and reported in swa_req_block_ids.
// After UpdateRequestTokens the SWA blocks become shareable.
// After DestroyFinishedRequest the SWA blocks become reclaimable.
TEST_F(PrefixCacheManagerSWATest, SWABlockAllocatedWithFABlock) {
  // 5 full blocks = 80 tokens.
  std::vector<int> token_ids;
  faked_token_generator_->GeneratePromptTokens({{1, 80}}, token_ids);

  int64_t req_id = 1;
  size_t shared_block_num = 0, unique_block_num = 0, shared_token_num = 0;
  cache_manager_->GetRequestPrefixBlockNumber(req_id, token_ids, token_ids.size(), shared_block_num, unique_block_num,
                                              shared_token_num);

  // No prefix cache yet.
  EXPECT_EQ(shared_block_num, 0u);
  EXPECT_EQ(unique_block_num, 5u);  // 80/16 = 5

  std::vector<int> req_block_ids;
  std::vector<int> swa_req_block_ids;
  std::vector<size_t> swa_block_idx_offset;
  Status status = cache_manager_->AllocateRequestBlocks(req_id, unique_block_num, req_block_ids, swa_req_block_ids,
                                                        swa_block_idx_offset);
  EXPECT_TRUE(status.OK());

  // FA blocks allocated.
  EXPECT_EQ(req_block_ids.size(), 5u);

  // SWA blocks should be allocated for all 5 new FA blocks.
  // swa_window_fa_block_ids covers all 5 blocks (window = 2, but all are new).
  EXPECT_EQ(swa_req_block_ids.size(), 5u);

  // swa_block_idx_offset should point to the first SWA window block (index 0).
  EXPECT_EQ(swa_block_idx_offset[0], 0u);

  // Fill all blocks.
  bool block_merged = false;
  status = cache_manager_->UpdateRequestTokens(req_id, token_ids, token_ids.size(), req_block_ids, block_merged,
                                               swa_req_block_ids, swa_block_idx_offset);
  EXPECT_TRUE(status.OK());

  // After UpdateRequestTokens the SWA blocks for blocks 0-2 (outside window) should
  // have been marked reclaimable (window covers last 2 blocks: indices 3 and 4).
  // swa_req_block_ids should now contain only the 2 window blocks.
  EXPECT_EQ(swa_req_block_ids.size(), 2u);
  EXPECT_EQ(swa_block_idx_offset[0], 3u);  // window starts at block index 3

  // Verify SWA manager state via the public accessor on the cache manager.
  // All 5 FA blocks should have valid (shareable) SWA blocks.
  for (size_t i = 0; i < 5; ++i) {
    size_t fa_block_id = cache_manager_->cached_requests_[req_id]->cached_blocks[i]->block_id;
    EXPECT_TRUE(cache_manager_->swa_cache_manager_->IsSWABlockValid(fa_block_id));
  }

  cache_manager_->DestroyFinishedRequest(req_id);
}

// Test: Normal prefix cache hit with SWA enabled (design doc §5.1).
// Request A fills 8 blocks; Request B has the same tokens and should get
// shared_block_num = 8 with SWA window blocks registered.
TEST_F(PrefixCacheManagerSWATest, NormalPrefixCacheHitWithSWA) {
  // 8 full blocks = 128 tokens.
  std::vector<int> token_ids;
  faked_token_generator_->GeneratePromptTokens({{1, 128}}, token_ids);

  // Request A: fill the prefix tree.
  RunFullRequest(1, token_ids);

  // Request B: same tokens, should hit all 8 blocks.
  int64_t req_id_b = 2;
  size_t shared_block_num = 0, unique_block_num = 0, shared_token_num = 0;
  cache_manager_->GetRequestPrefixBlockNumber(req_id_b, token_ids, token_ids.size(), shared_block_num, unique_block_num,
                                              shared_token_num);

  // All 8 blocks should be shared (SWA cache is valid for all of them).
  EXPECT_EQ(shared_block_num, 8u);
  EXPECT_EQ(unique_block_num, 0u);
  EXPECT_EQ(shared_token_num, 128u);

  // Phase 2 should have registered sliding_window_blocks + 1 (3) blocks for full hit
  // (the extra block covers the rollback by HandleFullPrefixCacheHit).
  EXPECT_EQ(cache_manager_->cached_requests_[req_id_b]->swa_window_fa_block_ids.size(), sliding_window_blocks + 1);
  EXPECT_EQ(cache_manager_->cached_requests_[req_id_b]->swa_window_block_idxs.front(),
            8u - (sliding_window_blocks + 1));

  cache_manager_->DestroyFinishedRequest(req_id_b);
}

// Test: SWA block eviction causes prefix cache truncation (design doc §5.2).
// After Request A fills 8 blocks, we manually evict the SWA block for block 5
// (index 5, 0-based) by calling FreeSWABlock.  Request C should then see
// shared_block_num = 5 (blocks 0-4 form a valid window ending at index 5).
TEST_F(PrefixCacheManagerSWATest, SWABlockEvictionTruncatesPrefixCache) {
  // 8 full blocks = 128 tokens.
  std::vector<int> token_ids;
  faked_token_generator_->GeneratePromptTokens({{1, 128}}, token_ids);

  // Request A: fill the prefix tree.
  RunFullRequest(1, token_ids);

  // Collect the FA block ids for blocks 0-7 from the prefix tree.
  // After DestroyFinishedRequest the blocks are in reusable_cached_blocks_.
  // We can find them by walking the prefix tree from root.
  // Use a second request to get the cached_blocks list.
  int64_t req_id_probe = 99;
  size_t sb = 0, ub = 0, st = 0;
  cache_manager_->GetRequestPrefixBlockNumber(req_id_probe, token_ids, token_ids.size(), sb, ub, st);
  ASSERT_EQ(sb, 8u);

  // Evict the SWA block for block index 5 (fa_block_id of cached_blocks[5]).
  size_t evict_fa_block_id = cache_manager_->cached_requests_[req_id_probe]->cached_blocks[5]->block_id;
  // Mark it reclaimable first (simulating that no request holds it), then free it.
  cache_manager_->swa_cache_manager_->SetReclaimable(req_id_probe, evict_fa_block_id);
  cache_manager_->swa_cache_manager_->FreeSWABlock(evict_fa_block_id);

  // Also evict block index 7 to create a second gap.
  size_t evict_fa_block_id_7 = cache_manager_->cached_requests_[req_id_probe]->cached_blocks[7]->block_id;
  cache_manager_->swa_cache_manager_->SetReclaimable(req_id_probe, evict_fa_block_id_7);
  cache_manager_->swa_cache_manager_->FreeSWABlock(evict_fa_block_id_7);

  cache_manager_->DestroyFinishedRequest(req_id_probe);

  // Request C: same tokens.
  // SWA state: blocks 0-4 valid, block 5 evicted, block 6 valid, block 7 evicted.
  // sliding_window_blocks = 2.
  // Streak analysis (design doc §4.4 example):
  //   i=0..4: streak grows to 5, last_swa_valid_idx=5
  //   i=5: invalid → streak=0
  //   i=6: valid → streak=1 (<2, no update)
  //   i=7: invalid → streak=0
  // → last_swa_valid_idx = 5, shared_block_num = 5.
  int64_t req_id_c = 3;
  cache_manager_->GetRequestPrefixBlockNumber(req_id_c, token_ids, token_ids.size(), sb, ub, st);

  EXPECT_EQ(sb, 5u);
  EXPECT_EQ(ub, 3u);  // 8 - 5 = 3 unique blocks needed

  // Phase 2 window: [5-2, 5) = [3, 5), so blocks 3 and 4 are registered.
  EXPECT_EQ(cache_manager_->cached_requests_[req_id_c]->swa_window_fa_block_ids.size(), sliding_window_blocks);
  EXPECT_EQ(cache_manager_->cached_requests_[req_id_c]->swa_window_block_idxs.front(), 3u);

  cache_manager_->DestroyFinishedRequest(req_id_c);
}

// Test: SWA window slides as tokens are generated (design doc §4.5).
// After allocating 6 blocks (96 tokens), UpdateRequestTokens is called with
// increasing shareable_kvcache_token_num values to simulate token generation.
// Blocks that slide out of the window should become reclaimable.
TEST_F(PrefixCacheManagerSWATest, SWAWindowSlides) {
  // 6 full blocks = 96 tokens.
  std::vector<int> token_ids;
  faked_token_generator_->GeneratePromptTokens({{1, 96}}, token_ids);

  int64_t req_id = 1;
  size_t shared_block_num = 0, unique_block_num = 0, shared_token_num = 0;
  cache_manager_->GetRequestPrefixBlockNumber(req_id, token_ids, token_ids.size(), shared_block_num, unique_block_num,
                                              shared_token_num);
  EXPECT_EQ(shared_block_num, 0u);
  EXPECT_EQ(unique_block_num, 6u);

  std::vector<int> req_block_ids;
  std::vector<int> swa_req_block_ids;
  std::vector<size_t> swa_block_idx_offset;
  Status status = cache_manager_->AllocateRequestBlocks(req_id, unique_block_num, req_block_ids, swa_req_block_ids,
                                                        swa_block_idx_offset);
  EXPECT_TRUE(status.OK());

  // All 6 SWA blocks allocated.
  EXPECT_EQ(swa_req_block_ids.size(), 6u);

  // Simulate filling 2 blocks (32 tokens).  Window = 32 tokens = 2 blocks.
  // swa_left_block = (32 - 32) / 16 = 0 → no blocks slide out yet.
  bool block_merged = false;
  status = cache_manager_->UpdateRequestTokens(req_id, token_ids, 32, req_block_ids, block_merged, swa_req_block_ids,
                                               swa_block_idx_offset);
  EXPECT_TRUE(status.OK());
  // Window covers blocks 0-1; no blocks have slid out.
  EXPECT_EQ(swa_req_block_ids.size(), 6u);
  EXPECT_EQ(swa_block_idx_offset[0], 0u);

  // Simulate filling 4 blocks (64 tokens).
  // swa_left_block = (64 - 32) / 16 = 2 → blocks 0 and 1 slide out.
  status = cache_manager_->UpdateRequestTokens(req_id, token_ids, 64, req_block_ids, block_merged, swa_req_block_ids,
                                               swa_block_idx_offset);
  EXPECT_TRUE(status.OK());
  // Window now covers blocks 2-3; blocks 0 and 1 have slid out.
  EXPECT_EQ(swa_req_block_ids.size(), 4u);
  EXPECT_EQ(swa_block_idx_offset[0], 2u);

  // Simulate filling 6 blocks (96 tokens).
  // swa_left_block = (96 - 32) / 16 = 4 → blocks 2 and 3 also slide out.
  status = cache_manager_->UpdateRequestTokens(req_id, token_ids, 96, req_block_ids, block_merged, swa_req_block_ids,
                                               swa_block_idx_offset);
  EXPECT_TRUE(status.OK());
  // Window now covers blocks 4-5; blocks 0-3 have slid out.
  EXPECT_EQ(swa_req_block_ids.size(), 2u);
  EXPECT_EQ(swa_block_idx_offset[0], 4u);

  // Verify that the SWA blocks for blocks 0-3 are now reclaimable (valid but reclaimable).
  // We check via IsSWABlockValid: shareable blocks remain valid even when reclaimable.
  for (size_t i = 0; i < 4; ++i) {
    size_t fa_block_id = cache_manager_->cached_requests_[req_id]->cached_blocks[i]->block_id;
    EXPECT_TRUE(cache_manager_->swa_cache_manager_->IsSWABlockValid(fa_block_id));
  }

  cache_manager_->DestroyFinishedRequest(req_id);
}

// Test: DestroyFinishedRequest marks all SWA window blocks as reclaimable.
TEST_F(PrefixCacheManagerSWATest, DestroyFinishedRequestReleasesSWAWindow) {
  // 4 full blocks = 64 tokens.
  std::vector<int> token_ids;
  faked_token_generator_->GeneratePromptTokens({{1, 64}}, token_ids);

  int64_t req_id = 1;
  size_t shared_block_num = 0, unique_block_num = 0, shared_token_num = 0;
  cache_manager_->GetRequestPrefixBlockNumber(req_id, token_ids, token_ids.size(), shared_block_num, unique_block_num,
                                              shared_token_num);

  std::vector<int> req_block_ids;
  std::vector<int> swa_req_block_ids;
  std::vector<size_t> swa_block_idx_offset;
  Status status = cache_manager_->AllocateRequestBlocks(req_id, unique_block_num, req_block_ids, swa_req_block_ids,
                                                        swa_block_idx_offset);
  EXPECT_TRUE(status.OK());

  bool block_merged = false;
  status = cache_manager_->UpdateRequestTokens(req_id, token_ids, token_ids.size(), req_block_ids, block_merged,
                                               swa_req_block_ids, swa_block_idx_offset);
  EXPECT_TRUE(status.OK());

  // After UpdateRequestTokens with 64 tokens and window=32:
  // swa_left_block = (64-32)/16 = 2 → blocks 0 and 1 slid out.
  // swa_window_fa_block_ids should contain blocks 2 and 3.
  EXPECT_EQ(cache_manager_->cached_requests_[req_id]->swa_window_fa_block_ids.size(), 2u);

  // Collect the FA block ids still in the window before destroy.
  std::vector<size_t> window_ids = cache_manager_->cached_requests_[req_id]->swa_window_fa_block_ids;

  cache_manager_->DestroyFinishedRequest(req_id);

  // After destroy, the SWA blocks for the window blocks should still be valid
  // (shareable) but now reclaimable.  IsSWABlockValid returns true for shareable
  // blocks regardless of reclaimable state.
  for (size_t fa_block_id : window_ids) {
    EXPECT_TRUE(cache_manager_->swa_cache_manager_->IsSWABlockValid(fa_block_id));
  }

  // The available count should have increased (reclaimable blocks are counted).
  size_t available = cache_manager_->swa_cache_manager_->GetAvailableBlockCount();
  EXPECT_GE(available, 2u);
}

// Test: SWA block merge path — when AppendFilledCachedBlock merges a new FA block
// into an existing tree node, the new FA block's SWA block is freed immediately.
//
// Scenario 1 (original): Request A fills the prefix tree; Request B has the same
// tokens and gets shared_block_num=4, unique_block_num=0.  Since ub=0,
// AllocateRequestBlocks allocates no new FA/SWA blocks.  After allocation we
// verify that Request B's cached_blocks and swa_window_fa_block_ids match
// Request A's (i.e. they share the same tree nodes and SWA blocks).
//
// Scenario 2 (new): Two independent requests (A and B) are created with the same
// tokens BEFORE either has been committed to the prefix tree.  Both allocate their
// own unique FA and SWA blocks.  After both call UpdateRequestTokens the duplicate
// FA block is merged: the second request's last block pointer is replaced by the
// first request's last block pointer, and the redundant SWA block is freed.
TEST_F(PrefixCacheManagerSWATest, SWABlockFreedOnFABlockMerge) {
  // 4 full blocks = 64 tokens.
  std::vector<int> token_ids;
  faked_token_generator_->GeneratePromptTokens({{1, 64}}, token_ids);

  // -----------------------------------------------------------------------
  // Scenario 1: Request A fills the prefix tree first; Request B then hits
  // the full prefix cache (ub=0) and should share all FA and SWA blocks.
  // -----------------------------------------------------------------------

  // Request A fills the prefix tree.
  {
    std::vector<int> rbi_a, srbi_a;
    std::vector<size_t> sbo_a;
    RunRequestUntilUpdate(1, token_ids, rbi_a, srbi_a, sbo_a);
    // Collect the FA block ids and SWA window block ids from request A before destroy.
    std::vector<PrefixCachedBlock*> blocks_a = cache_manager_->cached_requests_[1]->cached_blocks;
    std::vector<size_t> swa_window_a = cache_manager_->cached_requests_[1]->swa_window_fa_block_ids;
    cache_manager_->DestroyFinishedRequest(1);

    // Record SWA free count after Request A is destroyed (window blocks become reclaimable).
    size_t swa_free_before_b = swa_block_allocator_group_->GetDeviceBlockAllocator(0)->GetFreeBlockNumber();

    // Request B has the same tokens.  All 4 FA blocks will be found in the tree
    // (shared_block_num=4, unique_block_num=0), so AllocateRequestBlocks allocates
    // 0 new FA blocks and 0 new SWA blocks.
    int64_t req_id_b = 2;
    size_t sb = 0, ub = 0, st = 0;
    cache_manager_->GetRequestPrefixBlockNumber(req_id_b, token_ids, token_ids.size(), sb, ub, st);
    EXPECT_EQ(sb, 4u);
    EXPECT_EQ(ub, 0u);

    std::vector<int> req_block_ids_b;
    std::vector<int> swa_req_block_ids_b;
    std::vector<size_t> swa_block_idx_offset_b;
    Status status = cache_manager_->AllocateRequestBlocks(req_id_b, ub, req_block_ids_b, swa_req_block_ids_b,
                                                          swa_block_idx_offset_b);
    EXPECT_TRUE(status.OK());

    // No new SWA blocks should have been allocated (no new FA blocks).
    size_t swa_free_after_b = swa_block_allocator_group_->GetDeviceBlockAllocator(0)->GetFreeBlockNumber();
    EXPECT_EQ(swa_free_before_b, swa_free_after_b);

    // Request B's cached_blocks must point to the same tree nodes as Request A's.
    ASSERT_EQ(cache_manager_->cached_requests_[req_id_b]->cached_blocks.size(), blocks_a.size());
    for (size_t i = 0; i < blocks_a.size(); ++i) {
      EXPECT_EQ(cache_manager_->cached_requests_[req_id_b]->cached_blocks[i], blocks_a[i])
          << "FA block mismatch at index " << i;
    }

    // Request B's SWA window covers sliding_window_blocks + 1 = 3 blocks for full hit
    // (the extra block covers the rollback by HandleFullPrefixCacheHit).
    // The window spans blocks [1, 2, 3] while Request A's post-slide window was [2, 3].
    EXPECT_EQ(cache_manager_->cached_requests_[req_id_b]->swa_window_fa_block_ids.size(), sliding_window_blocks + 1);
    EXPECT_EQ(cache_manager_->cached_requests_[req_id_b]->swa_window_block_idxs.front(),
              4u - (sliding_window_blocks + 1));
    // The tail of B's window (last sliding_window_blocks entries) must still match A's window.
    const auto& b_window = cache_manager_->cached_requests_[req_id_b]->swa_window_fa_block_ids;
    std::vector<size_t> b_window_tail(b_window.end() - sliding_window_blocks, b_window.end());
    EXPECT_EQ(b_window_tail, swa_window_a);

    cache_manager_->DestroyFinishedRequest(req_id_b);
  }

  // -----------------------------------------------------------------------
  // Scenario 2: Two independent requests (C and D) are created with the same
  // tokens before either is committed to the prefix tree.  Both allocate their
  // own unique FA and SWA blocks.  After both call UpdateRequestTokens the
  // duplicate last FA block is merged: C and D end up sharing the same block
  // pointer, and the redundant SWA block is freed.
  // -----------------------------------------------------------------------
  {
    // Use a fresh set of tokens so there is no prefix cache hit from Scenario 1.
    std::vector<int> token_ids_cd;
    faked_token_generator_->GeneratePromptTokens({{2, 64}}, token_ids_cd);

    int64_t req_id_c = 3;
    int64_t req_id_d = 4;

    // Both requests call GetRequestPrefixBlockNumber before either is committed.
    // Neither should see any shared blocks.
    size_t sb_c = 0, ub_c = 0, st_c = 0;
    cache_manager_->GetRequestPrefixBlockNumber(req_id_c, token_ids_cd, token_ids_cd.size(), sb_c, ub_c, st_c);
    EXPECT_EQ(sb_c, 0u);
    EXPECT_EQ(ub_c, 4u);

    size_t sb_d = 0, ub_d = 0, st_d = 0;
    cache_manager_->GetRequestPrefixBlockNumber(req_id_d, token_ids_cd, token_ids_cd.size(), sb_d, ub_d, st_d);
    EXPECT_EQ(sb_d, 0u);
    EXPECT_EQ(ub_d, 4u);

    // Both requests allocate their own unique FA and SWA blocks.
    std::vector<int> req_block_ids_c, swa_req_block_ids_c;
    std::vector<size_t> swa_block_idx_offset_c;
    Status status = cache_manager_->AllocateRequestBlocks(req_id_c, ub_c, req_block_ids_c, swa_req_block_ids_c,
                                                          swa_block_idx_offset_c);
    EXPECT_TRUE(status.OK());
    EXPECT_EQ(req_block_ids_c.size(), 4u);
    EXPECT_EQ(swa_req_block_ids_c.size(), 4u);

    std::vector<int> req_block_ids_d, swa_req_block_ids_d;
    std::vector<size_t> swa_block_idx_offset_d;
    status = cache_manager_->AllocateRequestBlocks(req_id_d, ub_d, req_block_ids_d, swa_req_block_ids_d,
                                                   swa_block_idx_offset_d);
    EXPECT_TRUE(status.OK());
    EXPECT_EQ(req_block_ids_d.size(), 4u);
    EXPECT_EQ(swa_req_block_ids_d.size(), 4u);

    // Before UpdateRequestTokens the last blocks are distinct (each request owns its own).
    EXPECT_NE(cache_manager_->cached_requests_[req_id_c]->cached_blocks.back(),
              cache_manager_->cached_requests_[req_id_d]->cached_blocks.back());

    // Record SWA free count before any UpdateRequestTokens call.
    size_t swa_free_before_update = swa_block_allocator_group_->GetDeviceBlockAllocator(0)->GetFreeBlockNumber();

    // Request C calls UpdateRequestTokens first — its last block is appended to the tree.
    bool block_merged_c = false;
    status = cache_manager_->UpdateRequestTokens(req_id_c, token_ids_cd, token_ids_cd.size(), req_block_ids_c,
                                                 block_merged_c, swa_req_block_ids_c, swa_block_idx_offset_c);
    EXPECT_TRUE(status.OK());

    // Request D calls UpdateRequestTokens — its last block duplicates C's, so it
    // should be merged (replaced by C's block pointer) and D's redundant SWA block freed.
    bool block_merged_d = false;
    status = cache_manager_->UpdateRequestTokens(req_id_d, token_ids_cd, token_ids_cd.size(), req_block_ids_d,
                                                 block_merged_d, swa_req_block_ids_d, swa_block_idx_offset_d);
    EXPECT_TRUE(status.OK());

    // After merge: C and D must share the same last FA block pointer.
    EXPECT_EQ(cache_manager_->cached_requests_[req_id_c]->cached_blocks.back(),
              cache_manager_->cached_requests_[req_id_d]->cached_blocks.back())
        << "Expected last FA block to be merged (same pointer) after both UpdateRequestTokens calls";

    // After merge: C and D must share the same SWA window FA block ids.
    EXPECT_EQ(cache_manager_->cached_requests_[req_id_c]->swa_window_fa_block_ids,
              cache_manager_->cached_requests_[req_id_d]->swa_window_fa_block_ids)
        << "Expected SWA window FA block ids to match after merge";

    // The redundant SWA block allocated for D's last block must have been freed:
    // the SWA free count should be higher than before the update (one block returned).
    size_t swa_free_after_update = swa_block_allocator_group_->GetDeviceBlockAllocator(0)->GetFreeBlockNumber();
    EXPECT_GT(swa_free_after_update, swa_free_before_update)
        << "Expected redundant SWA block to be freed after FA block merge";

    cache_manager_->DestroyFinishedRequest(req_id_c);
    cache_manager_->DestroyFinishedRequest(req_id_d);
  }
}

// Test: Two independent requests merge blocks in two UpdateRequestTokens steps.
// After the first step the SWA window has already slid forward, so the second
// merge only partially overlaps the window (swa_window_block_idxs.front() > 0).
// Verifies that AppendFilledCachedBlock correctly updates swa_window_fa_block_ids
// only for the blocks that are still inside the window.
//
// Setup (block_token_num=16, sliding_window_size=32 → window=2 blocks):
//   6 blocks = 96 tokens.
//   Request C fills all 6 blocks in one shot.
//   Request D allocates 6 independent blocks, then calls UpdateRequestTokens twice:
//     Step 1: 64 tokens → fills blocks 0-3, window slides to start_idx=2.
//     Step 2: 96 tokens → fills blocks 4-5 (merge with C), window slides to start_idx=4.
//   After both steps D's swa_window_fa_block_ids must equal C's {c_b4, c_b5}.
TEST_F(PrefixCacheManagerSWATest, SWAMergePartialWindowHit) {
  // 6 full blocks = 96 tokens.
  std::vector<int> token_ids;
  faked_token_generator_->GeneratePromptTokens({{3, 96}}, token_ids);

  int64_t req_id_c = 10;
  int64_t req_id_d = 11;

  // Both requests call GetRequestPrefixBlockNumber before either commits to the
  // tree, so both see sb=0, ub=6 and allocate independent blocks.
  size_t sb_c = 0, ub_c = 0, st_c = 0;
  cache_manager_->GetRequestPrefixBlockNumber(req_id_c, token_ids, token_ids.size(), sb_c, ub_c, st_c);
  EXPECT_EQ(sb_c, 0u);
  EXPECT_EQ(ub_c, 6u);

  size_t sb_d = 0, ub_d = 0, st_d = 0;
  cache_manager_->GetRequestPrefixBlockNumber(req_id_d, token_ids, token_ids.size(), sb_d, ub_d, st_d);
  EXPECT_EQ(sb_d, 0u);
  EXPECT_EQ(ub_d, 6u);

  std::vector<int> req_block_ids_c, swa_req_block_ids_c;
  std::vector<size_t> swa_block_idx_offset_c;
  EXPECT_TRUE(cache_manager_
                  ->AllocateRequestBlocks(req_id_c, ub_c, req_block_ids_c, swa_req_block_ids_c, swa_block_idx_offset_c)
                  .OK());

  std::vector<int> req_block_ids_d, swa_req_block_ids_d;
  std::vector<size_t> swa_block_idx_offset_d;
  EXPECT_TRUE(cache_manager_
                  ->AllocateRequestBlocks(req_id_d, ub_d, req_block_ids_d, swa_req_block_ids_d, swa_block_idx_offset_d)
                  .OK());
  EXPECT_EQ(req_block_ids_d.size(), 6u);
  EXPECT_EQ(swa_req_block_ids_d.size(), 6u);

  // Before any UpdateRequestTokens: C and D own distinct blocks.
  EXPECT_NE(cache_manager_->cached_requests_[req_id_c]->cached_blocks.back(),
            cache_manager_->cached_requests_[req_id_d]->cached_blocks.back());

  // ── Request C: fill all 6 blocks in one shot ──────────────────────────────
  bool block_merged_c = false;
  EXPECT_TRUE(cache_manager_
                  ->UpdateRequestTokens(req_id_c, token_ids, token_ids.size(), req_block_ids_c, block_merged_c,
                                        swa_req_block_ids_c, swa_block_idx_offset_c)
                  .OK());
  // After C's update: window slides to start_idx=4, ids=[c_b4, c_b5].
  EXPECT_EQ(cache_manager_->cached_requests_[req_id_c]->swa_window_block_idxs.front(), 4u);
  EXPECT_EQ(cache_manager_->cached_requests_[req_id_c]->swa_window_fa_block_ids.size(), 2u);
  std::vector<size_t> swa_window_c = cache_manager_->cached_requests_[req_id_c]->swa_window_fa_block_ids;

  // ── Step 1: D fills blocks 0-3 (64 tokens) ───────────────────────────────
  // After this call the SWA window slides: swa_left_block=(64-32)/16=2,
  // so blocks 0-1 slide out → start_idx=2, ids=[c_b2, c_b3, d_b4, d_b5].
  bool block_merged_d = false;
  EXPECT_TRUE(cache_manager_
                  ->UpdateRequestTokens(req_id_d, token_ids, 64, req_block_ids_d, block_merged_d, swa_req_block_ids_d,
                                        swa_block_idx_offset_d)
                  .OK());
  EXPECT_EQ(cache_manager_->cached_requests_[req_id_d]->swa_window_block_idxs.front(), 2u);
  EXPECT_EQ(cache_manager_->cached_requests_[req_id_d]->swa_window_fa_block_ids.size(), 4u);

  // The first two entries in D's window must now be C's block ids (merged).
  const auto& ids_d_after_step1 = cache_manager_->cached_requests_[req_id_d]->swa_window_fa_block_ids;
  EXPECT_EQ(ids_d_after_step1[0], cache_manager_->cached_requests_[req_id_c]->cached_blocks[2]->block_id)
      << "D's window[0] should be C's block 2 after merge";
  EXPECT_EQ(ids_d_after_step1[1], cache_manager_->cached_requests_[req_id_c]->cached_blocks[3]->block_id)
      << "D's window[1] should be C's block 3 after merge";

  // ── Step 2: D fills blocks 4-5 (96 tokens) ───────────────────────────────
  // Blocks 4-5 are still in D's window (start_idx=2, positions 2 and 3).
  // After merge: ids[2]=c_b4, ids[3]=c_b5.
  // Then window slides: swa_left_block=(96-32)/16=4 → blocks 2-3 slide out.
  // Final: start_idx=4, ids=[c_b4, c_b5].
  block_merged_d = false;
  EXPECT_TRUE(cache_manager_
                  ->UpdateRequestTokens(req_id_d, token_ids, token_ids.size(), req_block_ids_d, block_merged_d,
                                        swa_req_block_ids_d, swa_block_idx_offset_d)
                  .OK());

  // D's window must now match C's exactly.
  EXPECT_EQ(cache_manager_->cached_requests_[req_id_d]->swa_window_block_idxs.front(), 4u);
  EXPECT_EQ(cache_manager_->cached_requests_[req_id_d]->swa_window_fa_block_ids, swa_window_c)
      << "D's SWA window must equal C's after two-step merge";

  // The last FA block pointer must be shared (merged).
  EXPECT_EQ(cache_manager_->cached_requests_[req_id_c]->cached_blocks.back(),
            cache_manager_->cached_requests_[req_id_d]->cached_blocks.back())
      << "Last FA block must be merged after two-step UpdateRequestTokens";

  cache_manager_->DestroyFinishedRequest(req_id_c);
  cache_manager_->DestroyFinishedRequest(req_id_d);
}

// Test: SWA disabled (swa_block_allocator_group = nullptr) — PrefixCacheManager
// behaves exactly as before: swa_req_block_ids is empty, no SWA state changes.
TEST_F(PrefixCacheManagerSWATest, SWADisabledWhenNoAllocatorGroup) {
  // Create a PrefixCacheManager without SWA.
  CacheManagerConfig no_swa_config = cache_manager_config_;
  no_swa_config.sliding_window_size = 0;

  BlockAllocatorGroupConfig fa_group_config;
  fa_group_config.devices = {0, 1};
  fa_group_config.device_block_num = fa_device_block_num;
  fa_group_config.host_block_num = fa_host_block_num;
  fa_group_config.block_size = block_token_num * 1024 * 1024;

  auto fa_group = std::make_shared<FakedBlockAllocatorGroup>(fa_group_config, memory_allocator_, context_,
                                                             block_allocator_creation_fn_);

  PrefixCacheManager* no_swa_mgr = new PrefixCacheManager(no_swa_config, fa_group, nullptr);
  no_swa_mgr->InitializeCachedBlocks();

  std::vector<int> token_ids;
  faked_token_generator_->GeneratePromptTokens({{1, 64}}, token_ids);

  int64_t req_id = 1;
  size_t sb = 0, ub = 0, st = 0;
  no_swa_mgr->GetRequestPrefixBlockNumber(req_id, token_ids, token_ids.size(), sb, ub, st);
  EXPECT_EQ(sb, 0u);
  EXPECT_EQ(ub, 4u);

  std::vector<int> req_block_ids;
  std::vector<int> swa_req_block_ids;
  std::vector<size_t> swa_block_idx_offset;
  Status status = no_swa_mgr->AllocateRequestBlocks(req_id, ub, req_block_ids, swa_req_block_ids, swa_block_idx_offset);
  EXPECT_TRUE(status.OK());

  // No SWA blocks should be allocated.
  EXPECT_TRUE(swa_req_block_ids.empty());
  EXPECT_TRUE(swa_block_idx_offset.empty());

  bool block_merged = false;
  status = no_swa_mgr->UpdateRequestTokens(req_id, token_ids, token_ids.size(), req_block_ids, block_merged,
                                           swa_req_block_ids, swa_block_idx_offset);
  EXPECT_TRUE(status.OK());
  EXPECT_TRUE(swa_req_block_ids.empty());

  no_swa_mgr->DestroyFinishedRequest(req_id);
  delete no_swa_mgr;
}

// Test: GetAvailableBlockCount reflects free + reclaimable SWA blocks.
// After a request finishes, its window blocks become reclaimable and the
// available count should increase accordingly.
TEST_F(PrefixCacheManagerSWATest, GetAvailableBlockCountReflectsReclaimable) {
  size_t initial_available = cache_manager_->swa_cache_manager_->GetAvailableBlockCount();
  EXPECT_EQ(initial_available, swa_device_block_num);

  // 4 full blocks = 64 tokens.
  std::vector<int> token_ids;
  faked_token_generator_->GeneratePromptTokens({{1, 64}}, token_ids);

  int64_t req_id = 1;
  size_t sb = 0, ub = 0, st = 0;
  cache_manager_->GetRequestPrefixBlockNumber(req_id, token_ids, token_ids.size(), sb, ub, st);

  std::vector<int> req_block_ids;
  std::vector<int> swa_req_block_ids;
  std::vector<size_t> swa_block_idx_offset;
  cache_manager_->AllocateRequestBlocks(req_id, ub, req_block_ids, swa_req_block_ids, swa_block_idx_offset);

  // After allocation, 4 SWA blocks are in use (non-reclaimable).
  size_t after_alloc = cache_manager_->swa_cache_manager_->GetAvailableBlockCount();
  EXPECT_EQ(after_alloc, swa_device_block_num - 4u);

  bool block_merged = false;
  cache_manager_->UpdateRequestTokens(req_id, token_ids, token_ids.size(), req_block_ids, block_merged,
                                      swa_req_block_ids, swa_block_idx_offset);

  // After UpdateRequestTokens with 64 tokens and window=32:
  // blocks 0-1 slid out (reclaimable), blocks 2-3 still in window (non-reclaimable).
  // Available = free_pool (swa_device_block_num - 4) + reclaimable (2) = swa_device_block_num - 2.
  size_t after_update = cache_manager_->swa_cache_manager_->GetAvailableBlockCount();
  EXPECT_EQ(after_update, swa_device_block_num - 2u);

  cache_manager_->DestroyFinishedRequest(req_id);

  // After destroy, blocks 2-3 also become reclaimable.
  // Available = free_pool (swa_device_block_num - 4) + reclaimable (4) = swa_device_block_num.
  size_t after_destroy = cache_manager_->swa_cache_manager_->GetAvailableBlockCount();
  EXPECT_EQ(after_destroy, swa_device_block_num);
}

// Test: Two requests share the same prefix; the second request's Phase 2
// increments hit_count for the shared SWA blocks.
TEST_F(PrefixCacheManagerSWATest, HitCountIncrementedOnPrefixCacheHit) {
  // 4 full blocks = 64 tokens.
  std::vector<int> token_ids;
  faked_token_generator_->GeneratePromptTokens({{1, 64}}, token_ids);

  // Request A fills the prefix tree.
  RunFullRequest(1, token_ids);

  // Collect FA block ids for blocks 2 and 3 (the SWA window for a 4-block prefix).
  int64_t req_id_probe = 99;
  size_t sb = 0, ub = 0, st = 0;
  cache_manager_->GetRequestPrefixBlockNumber(req_id_probe, token_ids, token_ids.size(), sb, ub, st);
  ASSERT_EQ(sb, 4u);

  size_t fa_id_2 = cache_manager_->cached_requests_[req_id_probe]->cached_blocks[2]->block_id;
  size_t fa_id_3 = cache_manager_->cached_requests_[req_id_probe]->cached_blocks[3]->block_id;

  // hit_count should be 1 after the first GetRequestPrefixBlockNumber call
  // (Phase 2 of req_id_probe incremented it).
  EXPECT_EQ(cache_manager_->swa_cache_manager_->swa_blocks_.at(fa_id_2)->hit_count, 1u);
  EXPECT_EQ(cache_manager_->swa_cache_manager_->swa_blocks_.at(fa_id_3)->hit_count, 1u);

  cache_manager_->DestroyFinishedRequest(req_id_probe);

  // Request B: same tokens again.
  int64_t req_id_b = 2;
  cache_manager_->GetRequestPrefixBlockNumber(req_id_b, token_ids, token_ids.size(), sb, ub, st);
  ASSERT_EQ(sb, 4u);

  // hit_count should now be 2.
  EXPECT_EQ(cache_manager_->swa_cache_manager_->swa_blocks_.at(fa_id_2)->hit_count, 2u);
  EXPECT_EQ(cache_manager_->swa_cache_manager_->swa_blocks_.at(fa_id_3)->hit_count, 2u);

  cache_manager_->DestroyFinishedRequest(req_id_b);
}

// ============================================================================
// Remote Cache Loading tests (AllocateRequestBlocksInternal,
// AllocateBlocksForRemoteCache, ConfirmRemoteCacheLoadingResult)
// ============================================================================
//
// Test fixture configuration:
//   block_token_num = 16
//   sliding_window_size = 16  →  sliding_window_blocks = 1
//   FA blocks: fa_device_block_num = 100
//   SWA blocks: swa_device_block_num = 50
//
// The sliding window is intentionally set to exactly 1 block so that
// matched_remote_swa_block_indexs.size() <= 1 is always satisfied (plan §3.4).
class PrefixCacheManagerRemoteCacheTest : public testing::Test {
 protected:
  void SetUp() override {
    block_allocator_creation_fn_ = [](MemoryLocation location, size_t block_num, size_t block_size, int rank,
                                      std::shared_ptr<MemoryAllocatorInterface> memory_allocator,
                                      std::shared_ptr<Context> context) {
      return std::make_shared<FakedBlockAllocator>(location, block_num, block_size, rank, memory_allocator, context);
    };

    context_ = std::make_shared<Context>();
    memory_allocator_ = std::make_shared<FakedMemoryAllocator>();

    // FA block allocator group.
    BlockAllocatorGroupConfig fa_group_config;
    fa_group_config.devices = {0, 1};
    fa_group_config.device_block_num = fa_device_block_num;
    fa_group_config.host_block_num = fa_host_block_num;
    fa_group_config.block_size = block_token_num * 1024 * 1024;

    fa_block_allocator_group_ = std::make_shared<FakedBlockAllocatorGroup>(fa_group_config, memory_allocator_, context_,
                                                                           block_allocator_creation_fn_);

    // SWA block allocator group (device-only).
    BlockAllocatorGroupConfig swa_group_config;
    swa_group_config.devices = {0};
    swa_group_config.device_block_num = swa_device_block_num;
    swa_group_config.host_block_num = 0;
    swa_group_config.block_size = block_token_num * 1024 * 1024;

    swa_block_allocator_group_ = std::make_shared<FakedBlockAllocatorGroup>(swa_group_config, memory_allocator_,
                                                                            context_, block_allocator_creation_fn_);

    cache_manager_config_.block_token_num = block_token_num;
    cache_manager_config_.tensor_para_size = 2;
    cache_manager_config_.enable_prefix_caching = true;
    cache_manager_config_.sliding_window_size = sliding_window_size;

    cache_manager_ =
        new PrefixCacheManager(cache_manager_config_, fa_block_allocator_group_, swa_block_allocator_group_);
    cache_manager_->InitializeCachedBlocks();

    faked_token_generator_ = new FakedTokenGenerator();
  }

  void TearDown() override {
    delete faked_token_generator_;
    delete cache_manager_;
  }

  // Helper: put req_id into kRemoteLoading state with fa_prefix_block_num FA blocks
  // already matched in the prefix tree (by running a "seed" request first).
  // seed_token_num controls how many tokens the seed request fills into the prefix tree;
  // it must be a multiple of block_token_num and <= token_ids.size().
  // Returns the fa_shared_block_num reported by GetRequestPrefixBlockNumberBeforeRemoteCacheLoading.
  size_t PrepareRemoteLoadingRequest(int64_t seed_req_id, int64_t req_id, const std::vector<int>& token_ids,
                                     size_t& fa_unique_block_num, size_t& swa_unique_block_num,
                                     size_t seed_token_num = 0) {
    // Default: seed fills all tokens (backward-compatible).
    if (seed_token_num == 0) {
      seed_token_num = token_ids.size();
    }
    // Run a seed request to populate the prefix tree with only the first seed_token_num tokens.
    {
      std::vector<int> seed_tokens(token_ids.begin(), token_ids.begin() + seed_token_num);
      size_t sb = 0, ub = 0, st = 0;
      cache_manager_->GetRequestPrefixBlockNumber(seed_req_id, seed_tokens, seed_tokens.size(), sb, ub, st);
      std::vector<int> rbi, srbi;
      std::vector<size_t> sbo;
      EXPECT_TRUE(cache_manager_->AllocateRequestBlocks(seed_req_id, ub, rbi, srbi, sbo).OK());
      bool bm = false;
      EXPECT_TRUE(
          cache_manager_->UpdateRequestTokens(seed_req_id, seed_tokens, seed_tokens.size(), rbi, bm, srbi, sbo).OK());
      cache_manager_->DestroyFinishedRequest(seed_req_id);
    }

    // Now call the "before remote cache loading" variant for the real request.
    size_t fa_sb = 0, fa_ub = 0, fa_st = 0;
    size_t swa_sb = 0, swa_ub = 0, swa_st = 0;
    Status s = cache_manager_->GetRequestPrefixBlockNumberBeforeRemoteCacheLoading(
        req_id, token_ids, token_ids.size(), fa_sb, fa_ub, fa_st, swa_sb, swa_ub, swa_st);
    EXPECT_TRUE(s.OK());
    EXPECT_TRUE(cache_manager_->cached_requests_[req_id]->is_remote_loading);

    fa_unique_block_num = fa_ub;
    swa_unique_block_num = swa_ub;
    return fa_sb;
  }

 protected:
  CacheManagerConfig cache_manager_config_;
  PrefixCacheManager* cache_manager_ = nullptr;
  FakedTokenGenerator* faked_token_generator_ = nullptr;

  std::shared_ptr<Context> context_ = nullptr;
  std::shared_ptr<MemoryAllocatorInterface> memory_allocator_ = nullptr;
  BlockAllocatorCreationFunc block_allocator_creation_fn_ = nullptr;
  std::shared_ptr<BlockAllocatorGroupInterface> fa_block_allocator_group_ = nullptr;
  std::shared_ptr<BlockAllocatorGroupInterface> swa_block_allocator_group_ = nullptr;

  // block_token_num=16, sliding_window_size=16 → sliding_window_blocks=1
  const size_t block_token_num = 16;
  const size_t sliding_window_size = 16;
  const size_t sliding_window_blocks = 1;

  const size_t fa_device_block_num = 100;
  const size_t fa_host_block_num = 100;
  const size_t swa_device_block_num = 50;
};

// ============================================================================
// §6.1 — FA block consumption correctness
// AllocateBlocksForRemoteCache(fa_block_num=5, swa_block_num=3) must consume
// exactly 5 FA blocks from free_cached_blocks_, not 8.
// ============================================================================
TEST_F(PrefixCacheManagerRemoteCacheTest, AllocateBlocksForRemoteCache_FAConsumptionCorrect) {
  // 3 full blocks = 48 tokens → seed fills 3 blocks into the prefix tree.
  std::vector<int> token_ids;
  faked_token_generator_->GeneratePromptTokens({{1, 48}}, token_ids);

  // Extend to 8 blocks = 128 tokens for the remote-loading request.
  faked_token_generator_->GeneratePromptTokens({{2, 80}}, token_ids);  // 48 + 80 = 128 tokens

  int64_t seed_req_id = 1;
  int64_t req_id = 2;
  size_t fa_ub = 0, swa_ub = 0;
  // Seed only the first 48 tokens (3 blocks) into the prefix tree.
  size_t fa_sb = PrepareRemoteLoadingRequest(seed_req_id, req_id, token_ids, fa_ub, swa_ub, /*seed_token_num=*/48);

  // After seeding 3 blocks, fa_sb should be 3; fa_ub = 8 - 3 = 5.
  ASSERT_EQ(fa_sb, 3u);
  ASSERT_EQ(fa_ub, 5u);

  // Record free FA block count before allocation.
  size_t fa_free_before = cache_manager_->GetUsableBlockNumber();

  // Allocate 5 FA blocks and 3 SWA blocks for remote loading.
  std::vector<int> req_block_ids;
  std::vector<int> remote_fa_block_ids;
  std::vector<int> remote_swa_block_ids;
  Status s =
      cache_manager_->AllocateBlocksForRemoteCache(req_id, req_block_ids, /*fa_block_num=*/5,
                                                   /*swa_block_num=*/3, remote_fa_block_ids, remote_swa_block_ids);
  ASSERT_TRUE(s.OK());

  // §6.1.1 — GetUsableBlockNumber() (= free + reusable) decreased by 8:
  //   • 3 already-matched shared blocks move from reusable→active (§4.0.4 registration).
  //   • 5 new FA blocks taken from free_cached_blocks_.
  size_t fa_free_after = cache_manager_->GetUsableBlockNumber();
  EXPECT_EQ(fa_free_before - fa_free_after, 8u);

  // §6.1.2 — cached_blocks grew by exactly 5 entries (no placeholder blocks).
  // Before allocation cached_blocks had fa_sb=3 entries; now it should have 3+5=8.
  EXPECT_EQ(cache_manager_->cached_requests_[req_id]->cached_blocks.size(), 8u);

  // remote_fa_block_ids must have exactly 5 entries.
  EXPECT_EQ(remote_fa_block_ids.size(), 5u);

  cache_manager_->DestroyFinishedRequest(req_id);
}

// ============================================================================
// §6.2 — SWA binding correctness
// After AllocateBlocksForRemoteCache(fa_block_num=5, swa_block_num=3):
//   • remote_swa_block_ids.size() == 3
//   • remote_cache_swa_fa_block_ids.size() == 3
//   • remote_cache_swa_fa_block_ids[k] == cached_blocks[fa_prefix_block_num + k]->block_id
//   • GetSWAMemoryBlockId(remote_cache_swa_fa_block_ids[k]) == remote_swa_block_ids[k]
// ============================================================================
TEST_F(PrefixCacheManagerRemoteCacheTest, AllocateBlocksForRemoteCache_SWABindingCorrect) {
  // 3 full blocks = 48 tokens seed.
  std::vector<int> token_ids;
  faked_token_generator_->GeneratePromptTokens({{1, 48}}, token_ids);
  faked_token_generator_->GeneratePromptTokens({{2, 80}}, token_ids);  // total 128 tokens = 8 blocks

  int64_t seed_req_id = 1;
  int64_t req_id = 2;
  size_t fa_ub = 0, swa_ub = 0;
  // Seed only the first 48 tokens (3 blocks) into the prefix tree.
  size_t fa_sb = PrepareRemoteLoadingRequest(seed_req_id, req_id, token_ids, fa_ub, swa_ub, /*seed_token_num=*/48);
  ASSERT_EQ(fa_sb, 3u);

  std::vector<int> req_block_ids;
  std::vector<int> remote_fa_block_ids;
  std::vector<int> remote_swa_block_ids;
  Status s =
      cache_manager_->AllocateBlocksForRemoteCache(req_id, req_block_ids, /*fa_block_num=*/5,
                                                   /*swa_block_num=*/3, remote_fa_block_ids, remote_swa_block_ids);
  ASSERT_TRUE(s.OK());

  // §6.2.1 — remote_swa_block_ids has exactly 3 entries.
  EXPECT_EQ(remote_swa_block_ids.size(), 3u);

  // §6.2.2 — remote_cache_swa_fa_block_ids has exactly 3 entries.
  const auto& swa_fa_ids = cache_manager_->cached_requests_[req_id]->remote_cache_swa_fa_block_ids;
  EXPECT_EQ(swa_fa_ids.size(), 3u);

  // §6.2.2 — each entry equals the block_id of the corresponding newly allocated FA block.
  // fa_prefix_block_num == fa_sb == 3; new FA blocks start at cached_blocks[3].
  // swa_block_idx corresponds to last 3 fa blocks, which are at indices 5,6,7 in cached_blocks.
  size_t fa_prefix_block_num = cache_manager_->cached_requests_[req_id]->fa_prefix_block_num;
  EXPECT_EQ(fa_prefix_block_num, 3u);
  size_t swa_fa_block_idx =
      cache_manager_->cached_requests_[req_id]->cached_blocks.size() - remote_swa_block_ids.size();  // should be 5
  for (size_t k = 0; k < 3; ++k) {
    size_t expected_fa_block_id =
        cache_manager_->cached_requests_[req_id]->cached_blocks[swa_fa_block_idx + k]->block_id;
    EXPECT_EQ(swa_fa_ids[k], expected_fa_block_id);
  }

  // §6.2.3 — GetSWAMemoryBlockId(swa_fa_ids[k]) == remote_swa_block_ids[k].
  for (size_t k = 0; k < 3; ++k) {
    int mem_id = cache_manager_->swa_cache_manager_->GetSWAMemoryBlockId(swa_fa_ids[k]);
    EXPECT_EQ(mem_id, remote_swa_block_ids[k]);
  }

  cache_manager_->DestroyFinishedRequest(req_id);
}

// ============================================================================
// §6.3 — AllocateRequestBlocks behavior unchanged after refactor
// A normal (non-remote-loading) request must still get the correct
// req_block_ids, swa_req_block_ids, and swa_block_idx_offset.
// ============================================================================
TEST_F(PrefixCacheManagerRemoteCacheTest, AllocateRequestBlocks_BehaviorUnchanged) {
  // 5 full blocks = 80 tokens, no prefix cache hit.
  std::vector<int> token_ids;
  faked_token_generator_->GeneratePromptTokens({{3, 80}}, token_ids);

  int64_t req_id = 10;
  size_t sb = 0, ub = 0, st = 0;
  cache_manager_->GetRequestPrefixBlockNumber(req_id, token_ids, token_ids.size(), sb, ub, st);
  EXPECT_EQ(sb, 0u);
  EXPECT_EQ(ub, 5u);

  std::vector<int> req_block_ids;
  std::vector<int> swa_req_block_ids;
  std::vector<size_t> swa_block_idx_offset;
  Status s = cache_manager_->AllocateRequestBlocks(req_id, ub, req_block_ids, swa_req_block_ids, swa_block_idx_offset);
  ASSERT_TRUE(s.OK());

  // 5 FA blocks allocated.
  EXPECT_EQ(req_block_ids.size(), 5u);

  // SWA enabled: 5 SWA blocks allocated (one per FA block, window=1 block).
  EXPECT_EQ(swa_req_block_ids.size(), 5u);

  // swa_block_idx_offset starts at 0 (no shared prefix).
  EXPECT_EQ(swa_block_idx_offset[0], 0u);

  // All 5 FA blocks have SWA blocks bound.
  for (size_t i = 0; i < 5; ++i) {
    size_t fa_block_id = cache_manager_->cached_requests_[req_id]->cached_blocks[i]->block_id;
    EXPECT_GE(cache_manager_->swa_cache_manager_->GetSWAMemoryBlockId(fa_block_id), 0);
  }

  // Fill all blocks.
  bool block_merged = false;
  s = cache_manager_->UpdateRequestTokens(req_id, token_ids, token_ids.size(), req_block_ids, block_merged,
                                          swa_req_block_ids, swa_block_idx_offset);
  ASSERT_TRUE(s.OK());

  // After UpdateRequestTokens with window=1: only the last block stays in the window.
  // swa_req_block_ids should contain exactly 1 entry (the last block).
  EXPECT_EQ(swa_req_block_ids.size(), 1u);
  EXPECT_EQ(swa_block_idx_offset[0], 4u);  // window starts at block index 4

  cache_manager_->DestroyFinishedRequest(req_id);
}

// ============================================================================
// §6.4' pre-check — shareable_kvcache_token_num must not reduce coverage
// ============================================================================

// §6.4'.1 — SWA enabled: shareable_kvcache_token_num < swa_window right boundary → RET_RUNTIME_FAILED.
TEST_F(PrefixCacheManagerRemoteCacheTest, ConfirmRemoteCache_PreCheck_SWAEnabled_TooSmall) {
  // 4 full blocks = 64 tokens seed; request targets 5 blocks = 80 tokens.
  std::vector<int> token_ids;
  faked_token_generator_->GeneratePromptTokens({{1, 64}}, token_ids);
  faked_token_generator_->GeneratePromptTokens({{2, 16}}, token_ids);  // total 80 tokens = 5 blocks

  int64_t seed_req_id = 1;
  int64_t req_id = 2;
  size_t fa_ub = 0, swa_ub = 0;
  // Seed only the first 64 tokens (4 blocks) into the prefix tree.
  PrepareRemoteLoadingRequest(seed_req_id, req_id, token_ids, fa_ub, swa_ub, /*seed_token_num=*/64);

  // Allocate remote blocks: fa_block_num=1, swa_block_num=1 (window=1).
  std::vector<int> req_block_ids;
  std::vector<int> remote_fa_block_ids;
  std::vector<int> remote_swa_block_ids;
  ASSERT_TRUE(cache_manager_
                  ->AllocateBlocksForRemoteCache(req_id, req_block_ids, /*fa_block_num=*/1,
                                                 /*swa_block_num=*/1, remote_fa_block_ids, remote_swa_block_ids)
                  .OK());

  // swa_window_fa_block_ids covers the last 1 block of the 4 already-matched blocks.
  // swa_window right boundary = (swa_window_block_idxs.front() + 1) * block_token_num = 4 * 16 = 64.
  // Passing shareable_kvcache_token_num = 48 (< 64) must fail.
  std::vector<int> kvcached_token_ids(48, 0);
  Status s = cache_manager_->ConfirmRemoteCacheLoadingResult(req_id, kvcached_token_ids,
                                                             /*shareable_kvcache_token_num=*/48,
                                                             /*matched_remote_swa_block_indexs=*/{});
  EXPECT_EQ(s.GetCode(), RET_RUNTIME_FAILED);

  cache_manager_->DestroyFinishedRequest(req_id);
}

// §6.4'.2 — SWA disabled: shareable_kvcache_token_num < fa_prefix_block_num * block_token_num → RET_RUNTIME_FAILED.
TEST_F(PrefixCacheManagerRemoteCacheTest, ConfirmRemoteCache_PreCheck_NoSWA_TooSmall) {
  // Create a no-SWA cache manager.
  CacheManagerConfig no_swa_config = cache_manager_config_;
  no_swa_config.sliding_window_size = 0;

  BlockAllocatorGroupConfig fa_group_config;
  fa_group_config.devices = {0, 1};
  fa_group_config.device_block_num = fa_device_block_num;
  fa_group_config.host_block_num = fa_host_block_num;
  fa_group_config.block_size = block_token_num * 1024 * 1024;
  auto fa_group = std::make_shared<FakedBlockAllocatorGroup>(fa_group_config, memory_allocator_, context_,
                                                             block_allocator_creation_fn_);

  PrefixCacheManager* no_swa_mgr = new PrefixCacheManager(no_swa_config, fa_group, nullptr);
  no_swa_mgr->InitializeCachedBlocks();

  // Seed 3 blocks = 48 tokens.
  std::vector<int> token_ids;
  faked_token_generator_->GeneratePromptTokens({{1, 48}}, token_ids);
  faked_token_generator_->GeneratePromptTokens({{2, 32}}, token_ids);  // total 80 tokens = 5 blocks

  // Seed request: only the first 48 tokens (3 blocks) so that the remote-loading
  // request sees fa_sb=3 (not 5).
  {
    std::vector<int> seed_tokens(token_ids.begin(), token_ids.begin() + 48);
    size_t sb = 0, ub = 0, st = 0;
    no_swa_mgr->GetRequestPrefixBlockNumber(1, seed_tokens, seed_tokens.size(), sb, ub, st);
    std::vector<int> rbi, srbi;
    std::vector<size_t> sbo;
    EXPECT_TRUE(no_swa_mgr->AllocateRequestBlocks(1, ub, rbi, srbi, sbo).OK());
    bool bm = false;
    EXPECT_TRUE(no_swa_mgr->UpdateRequestTokens(1, seed_tokens, seed_tokens.size(), rbi, bm, srbi, sbo).OK());
    no_swa_mgr->DestroyFinishedRequest(1);
  }

  // Remote-loading request: full 80 tokens; only 3 blocks are in the prefix tree.
  int64_t req_id = 2;
  size_t fa_sb = 0, fa_ub = 0, fa_st = 0, swa_sb = 0, swa_ub = 0, swa_st = 0;
  ASSERT_TRUE(no_swa_mgr
                  ->GetRequestPrefixBlockNumberBeforeRemoteCacheLoading(req_id, token_ids, token_ids.size(), fa_sb,
                                                                        fa_ub, fa_st, swa_sb, swa_ub, swa_st)
                  .OK());
  ASSERT_EQ(fa_sb, 3u);  // only 3 blocks in tree

  std::vector<int> req_block_ids;
  std::vector<int> remote_fa_block_ids;
  std::vector<int> remote_swa_block_ids;
  ASSERT_TRUE(no_swa_mgr
                  ->AllocateBlocksForRemoteCache(req_id, req_block_ids, /*fa_block_num=*/2,
                                                 /*swa_block_num=*/0, remote_fa_block_ids, remote_swa_block_ids)
                  .OK());

  // fa_prefix_block_num == 3; minimum shareable_kvcache_token_num = 3 * 16 = 48.
  // Passing 32 (< 48) must fail.
  std::vector<int> kvcached_token_ids(32, 0);
  Status s = no_swa_mgr->ConfirmRemoteCacheLoadingResult(req_id, kvcached_token_ids,
                                                         /*shareable_kvcache_token_num=*/32,
                                                         /*matched_remote_swa_block_indexs=*/{});
  EXPECT_EQ(s.GetCode(), RET_RUNTIME_FAILED);

  no_swa_mgr->DestroyFinishedRequest(req_id);
  delete no_swa_mgr;
}

// ============================================================================
// §6.4' no-SWA path
// ============================================================================

// §6.4'.3 — cached_blocks.size() == shareable_block_num after truncation.
// §6.4'.4 — UpdateRequestTokens called when shareable_block_num > fa_prefix_block_num.
// §6.4'.5 — UpdateRequestTokens NOT called when shareable_block_num == fa_prefix_block_num.
TEST_F(PrefixCacheManagerRemoteCacheTest, ConfirmRemoteCache_NoSWA_TruncationAndUpdateTokens) {
  // No-SWA cache manager.
  CacheManagerConfig no_swa_config = cache_manager_config_;
  no_swa_config.sliding_window_size = 0;

  BlockAllocatorGroupConfig fa_group_config;
  fa_group_config.devices = {0, 1};
  fa_group_config.device_block_num = fa_device_block_num;
  fa_group_config.host_block_num = fa_host_block_num;
  fa_group_config.block_size = block_token_num * 1024 * 1024;
  auto fa_group = std::make_shared<FakedBlockAllocatorGroup>(fa_group_config, memory_allocator_, context_,
                                                             block_allocator_creation_fn_);

  PrefixCacheManager* no_swa_mgr = new PrefixCacheManager(no_swa_config, fa_group, nullptr);
  no_swa_mgr->InitializeCachedBlocks();

  // Build token_ids: first 48 tokens (3 blocks) seeded into tree; last 32 tokens (2 blocks) are remote.
  std::vector<int> token_ids;
  faked_token_generator_->GeneratePromptTokens({{1, 48}}, token_ids);
  faked_token_generator_->GeneratePromptTokens({{2, 32}}, token_ids);  // total 80 tokens = 5 blocks

  // Seed request: only the first 48 tokens (3 blocks) so that the remote-loading
  // request sees fa_sb=3 (not 5).
  {
    std::vector<int> seed_tokens(token_ids.begin(), token_ids.begin() + 48);
    size_t sb = 0, ub = 0, st = 0;
    no_swa_mgr->GetRequestPrefixBlockNumber(1, seed_tokens, seed_tokens.size(), sb, ub, st);
    std::vector<int> rbi, srbi;
    std::vector<size_t> sbo;
    EXPECT_TRUE(no_swa_mgr->AllocateRequestBlocks(1, ub, rbi, srbi, sbo).OK());
    bool bm = false;
    EXPECT_TRUE(no_swa_mgr->UpdateRequestTokens(1, seed_tokens, seed_tokens.size(), rbi, bm, srbi, sbo).OK());
    no_swa_mgr->DestroyFinishedRequest(1);
  }

  // -----------------------------------------------------------------------
  // Sub-case A: shareable_block_num > fa_prefix_block_num (new blocks loaded).
  // -----------------------------------------------------------------------
  {
    int64_t req_id = 2;
    size_t fa_sb = 0, fa_ub = 0, fa_st = 0, swa_sb = 0, swa_ub = 0, swa_st = 0;
    ASSERT_TRUE(no_swa_mgr
                    ->GetRequestPrefixBlockNumberBeforeRemoteCacheLoading(req_id, token_ids, token_ids.size(), fa_sb,
                                                                          fa_ub, fa_st, swa_sb, swa_ub, swa_st)
                    .OK());
    ASSERT_EQ(fa_sb, 3u);

    std::vector<int> req_block_ids;
    std::vector<int> remote_fa_block_ids;
    std::vector<int> remote_swa_block_ids;
    ASSERT_TRUE(no_swa_mgr
                    ->AllocateBlocksForRemoteCache(req_id, req_block_ids, /*fa_block_num=*/2,
                                                   /*swa_block_num=*/0, remote_fa_block_ids, remote_swa_block_ids)
                    .OK());
    // cached_blocks now has 5 entries.
    ASSERT_EQ(no_swa_mgr->cached_requests_[req_id]->cached_blocks.size(), 5u);

    // Confirm: shareable_kvcache_token_num = 5 * 16 = 80 → shareable_block_num = 5.
    // 5 > fa_prefix_block_num(3) → UpdateRequestTokens should be called.
    std::vector<int> kvcached_token_ids(token_ids.begin(), token_ids.end());
    Status s = no_swa_mgr->ConfirmRemoteCacheLoadingResult(req_id, kvcached_token_ids,
                                                           /*shareable_kvcache_token_num=*/80,
                                                           /*matched_remote_swa_block_indexs=*/{});
    ASSERT_TRUE(s.OK());

    // §6.4'.3 — cached_blocks.size() == shareable_block_num == 5.
    EXPECT_EQ(no_swa_mgr->cached_requests_[req_id]->cached_blocks.size(), 5u);

    // Request should no longer be in remote loading state; is_remote_loading_block_allocated reset.
    EXPECT_FALSE(no_swa_mgr->cached_requests_[req_id]->is_remote_loading);
    EXPECT_FALSE(no_swa_mgr->cached_requests_[req_id]->is_remote_loading_block_allocated);

    // G4a: req_state must be kRunning (blocks were allocated before Confirm).
    EXPECT_EQ(no_swa_mgr->cached_requests_[req_id]->req_state, RequestState::kRunning);

    no_swa_mgr->DestroyFinishedRequest(req_id);
  }

  // -----------------------------------------------------------------------
  // Sub-case B: shareable_block_num == fa_prefix_block_num (no new blocks).
  // -----------------------------------------------------------------------
  {
    int64_t req_id = 3;
    size_t fa_sb = 0, fa_ub = 0, fa_st = 0, swa_sb = 0, swa_ub = 0, swa_st = 0;
    ASSERT_TRUE(no_swa_mgr
                    ->GetRequestPrefixBlockNumberBeforeRemoteCacheLoading(req_id, token_ids, token_ids.size(), fa_sb,
                                                                          fa_ub, fa_st, swa_sb, swa_ub, swa_st)
                    .OK());
    ASSERT_EQ(fa_sb, 5u);  // all 5 blocks now in tree after sub-case A

    std::vector<int> req_block_ids;
    std::vector<int> remote_fa_block_ids;
    std::vector<int> remote_swa_block_ids;
    ASSERT_TRUE(no_swa_mgr
                    ->AllocateBlocksForRemoteCache(req_id, req_block_ids, /*fa_block_num=*/0,
                                                   /*swa_block_num=*/0, remote_fa_block_ids, remote_swa_block_ids)
                    .OK());

    // Confirm: shareable_kvcache_token_num = 5 * 16 = 80 → shareable_block_num = 5 == fa_prefix_block_num.
    // UpdateRequestTokens should NOT be called (no new blocks).
    std::vector<int> kvcached_token_ids(token_ids.begin(), token_ids.end());
    Status s = no_swa_mgr->ConfirmRemoteCacheLoadingResult(req_id, kvcached_token_ids,
                                                           /*shareable_kvcache_token_num=*/80,
                                                           /*matched_remote_swa_block_indexs=*/{});
    ASSERT_TRUE(s.OK());

    // §6.4'.5 — cached_blocks.size() == fa_prefix_block_num == 5.
    EXPECT_EQ(no_swa_mgr->cached_requests_[req_id]->cached_blocks.size(), 5u);
    EXPECT_FALSE(no_swa_mgr->cached_requests_[req_id]->is_remote_loading);
    EXPECT_FALSE(no_swa_mgr->cached_requests_[req_id]->is_remote_loading_block_allocated);

    // G4a: fa_block_num=0 → is_remote_loading_block_allocated was set to true by AllocateBlocksForRemoteCache
    // (even with 0 new FA blocks, the function still sets the flag). req_state must be kRunning.
    EXPECT_EQ(no_swa_mgr->cached_requests_[req_id]->req_state, RequestState::kRunning);

    no_swa_mgr->DestroyFinishedRequest(req_id);
  }

  delete no_swa_mgr;
}

// ============================================================================
// §6.4' SWA path — Case 1: no matched remote SWA block
// ============================================================================

// §6.4'.6 — remote_cache_swa_fa_block_ids cleared (FreeSWABlock called for each entry, G7).
// §6.4'.7 — cached_blocks.size() == swa_aligned_block_num == shareable_block_num.
// §6.4'.8 — UpdateRequestTokens not called (cached_blocks.size() <= fa_prefix_block_num).
// G7 fix: SWA blocks freed before clear; SWA allocator count restored.
TEST_F(PrefixCacheManagerRemoteCacheTest, ConfirmRemoteCache_SWA_Case1_NoMatch) {
  // Seed 4 full blocks = 64 tokens.
  std::vector<int> token_ids;
  faked_token_generator_->GeneratePromptTokens({{1, 64}}, token_ids);
  faked_token_generator_->GeneratePromptTokens({{2, 16}}, token_ids);  // total 80 tokens = 5 blocks

  int64_t seed_req_id = 1;
  int64_t req_id = 2;
  size_t fa_ub = 0, swa_ub = 0;
  // Seed only the first 64 tokens (4 blocks) into the prefix tree.
  size_t fa_sb = PrepareRemoteLoadingRequest(seed_req_id, req_id, token_ids, fa_ub, swa_ub, /*seed_token_num=*/64);
  // With window=1: swa_sb == fa_sb - 1 (last block of fa_sb has no valid SWA yet after seed).
  // Actually after seed all 4 blocks have valid SWA; swa_sb == 4, fa_sb == 4.
  ASSERT_EQ(fa_sb, 4u);

  // Allocate 1 FA block and 1 SWA block for remote loading.
  std::vector<int> req_block_ids;
  std::vector<int> remote_fa_block_ids;
  std::vector<int> remote_swa_block_ids;
  ASSERT_TRUE(cache_manager_
                  ->AllocateBlocksForRemoteCache(req_id, req_block_ids, /*fa_block_num=*/1,
                                                 /*swa_block_num=*/1, remote_fa_block_ids, remote_swa_block_ids)
                  .OK());
  ASSERT_EQ(cache_manager_->cached_requests_[req_id]->cached_blocks.size(), 5u);
  ASSERT_EQ(cache_manager_->cached_requests_[req_id]->remote_cache_swa_fa_block_ids.size(), 1u);

  // swa_window_fa_block_ids covers the last 1 block of the 4 matched blocks (index 3).
  // swa_aligned_block_num = swa_window_block_idxs.back() + 1 = 3 + 1 = 4.
  ASSERT_FALSE(cache_manager_->cached_requests_[req_id]->swa_window_block_idxs.empty());
  size_t swa_aligned_block_num = cache_manager_->cached_requests_[req_id]->swa_window_block_idxs.back() + 1;
  ASSERT_EQ(swa_aligned_block_num, 4u);

  // Confirm with no match: shareable_kvcache_token_num = swa_aligned_block_num * block_token_num = 64.
  std::vector<int> kvcached_token_ids(64, 0);
  Status s = cache_manager_->ConfirmRemoteCacheLoadingResult(req_id, kvcached_token_ids,
                                                             /*shareable_kvcache_token_num=*/64,
                                                             /*matched_remote_swa_block_indexs=*/{});
  ASSERT_TRUE(s.OK());

  // §6.4'.6 — remote_cache_swa_fa_block_ids cleared.
  EXPECT_TRUE(cache_manager_->cached_requests_[req_id]->remote_cache_swa_fa_block_ids.empty());

  // §6.4'.7 — cached_blocks.size() == swa_aligned_block_num == 4.
  EXPECT_EQ(cache_manager_->cached_requests_[req_id]->cached_blocks.size(), 4u);

  // Request no longer in remote loading state; is_remote_loading_block_allocated reset.
  EXPECT_FALSE(cache_manager_->cached_requests_[req_id]->is_remote_loading);
  EXPECT_FALSE(cache_manager_->cached_requests_[req_id]->is_remote_loading_block_allocated);

  // G4b: req_state must be kRunning (blocks were allocated before Confirm).
  EXPECT_EQ(cache_manager_->cached_requests_[req_id]->req_state, RequestState::kRunning);

  cache_manager_->DestroyFinishedRequest(req_id);
}

// ============================================================================
// §6.4' SWA path — Case 2: matched remote SWA block
// ============================================================================

// §6.4'.9  — old SWA window released via SetReclaimable; swa_window_fa_block_ids cleared.
// §6.4'.10 — unmatched remote SWA blocks freed via FreeSWABlock.
// §6.4'.11 — matched SWA block transferred to cached_blocks[shareable_block_num-1].
// §6.4'.12 — swa_window_fa_block_ids updated to [cached_blocks[shareable_block_num-1]->block_id].
// §6.4'.13 — swa_window_block_idxs.front() updated to shareable_block_num - 1.
// §6.4'.14 — cached_blocks.size() == shareable_block_num.
// §6.4'.15 — remote_cache_swa_fa_block_ids cleared.
// §6.4'.16 — UpdateRequestTokens called when shareable_block_num > fa_prefix_block_num.
TEST_F(PrefixCacheManagerRemoteCacheTest, ConfirmRemoteCache_SWA_Case2_Matched) {
  // Seed 4 full blocks = 64 tokens.
  std::vector<int> token_ids;
  faked_token_generator_->GeneratePromptTokens({{1, 64}}, token_ids);
  faked_token_generator_->GeneratePromptTokens({{2, 16}}, token_ids);  // total 80 tokens = 5 blocks

  int64_t seed_req_id = 1;
  int64_t req_id = 2;
  size_t fa_ub = 0, swa_ub = 0;
  // Seed only the first 64 tokens (4 blocks) into the prefix tree.
  size_t fa_sb = PrepareRemoteLoadingRequest(seed_req_id, req_id, token_ids, fa_ub, swa_ub, /*seed_token_num=*/64);
  ASSERT_EQ(fa_sb, 4u);

  // Allocate 1 FA block and 1 SWA block for remote loading.
  std::vector<int> req_block_ids;
  std::vector<int> remote_fa_block_ids;
  std::vector<int> remote_swa_block_ids;
  ASSERT_TRUE(cache_manager_
                  ->AllocateBlocksForRemoteCache(req_id, req_block_ids, /*fa_block_num=*/1,
                                                 /*swa_block_num=*/1, remote_fa_block_ids, remote_swa_block_ids)
                  .OK());
  ASSERT_EQ(cache_manager_->cached_requests_[req_id]->cached_blocks.size(), 5u);
  ASSERT_EQ(cache_manager_->cached_requests_[req_id]->remote_cache_swa_fa_block_ids.size(), 1u);

  // Record the old SWA window FA block id (index 3, the last matched block).
  std::vector<size_t> old_window_ids = cache_manager_->cached_requests_[req_id]->swa_window_fa_block_ids;
  ASSERT_EQ(old_window_ids.size(), 1u);

  // Record the destination FA block id: cached_blocks[shareable_block_num - 1] = cached_blocks[4].
  // shareable_block_num = 5 * 16 / 16 = 5.
  size_t dst_fa_block_id = cache_manager_->cached_requests_[req_id]->cached_blocks[4]->block_id;

  // Confirm with match at index 0: matched_remote_swa_block_indexs = {0}.
  // shareable_kvcache_token_num = 5 * 16 = 80.
  std::vector<int> kvcached_token_ids(token_ids.begin(), token_ids.end());
  Status s = cache_manager_->ConfirmRemoteCacheLoadingResult(req_id, kvcached_token_ids,
                                                             /*shareable_kvcache_token_num=*/80,
                                                             /*matched_remote_swa_block_indexs=*/{0});
  ASSERT_TRUE(s.OK());

  // §6.4'.9 — old SWA window released; swa_window_fa_block_ids no longer contains old_window_ids[0].
  // (After the call, swa_window_fa_block_ids should contain only dst_fa_block_id.)
  const auto& new_window = cache_manager_->cached_requests_[req_id]->swa_window_fa_block_ids;
  EXPECT_EQ(new_window.size(), 1u);
  if (!new_window.empty()) {
    EXPECT_NE(new_window[0], old_window_ids[0]);
  }

  // §6.4'.11 — matched SWA block is now bound to dst_fa_block_id (Transfer or SetFixed).
  // The SWA block for dst_fa_block_id must be valid.
  EXPECT_TRUE(cache_manager_->swa_cache_manager_->IsSWABlockValid(dst_fa_block_id));

  // §6.4'.12 — swa_window_fa_block_ids == [dst_fa_block_id].
  EXPECT_EQ(new_window[0], dst_fa_block_id);

  // §6.4'.13 — swa_window_block_idxs.front() == shareable_block_num - 1 == 4.
  EXPECT_EQ(cache_manager_->cached_requests_[req_id]->swa_window_block_idxs.front(), 4u);

  // §6.4'.14 — cached_blocks.size() == shareable_block_num == 5.
  EXPECT_EQ(cache_manager_->cached_requests_[req_id]->cached_blocks.size(), 5u);

  // §6.4'.15 — remote_cache_swa_fa_block_ids cleared.
  EXPECT_TRUE(cache_manager_->cached_requests_[req_id]->remote_cache_swa_fa_block_ids.empty());

  // Request no longer in remote loading state; is_remote_loading_block_allocated reset.
  EXPECT_FALSE(cache_manager_->cached_requests_[req_id]->is_remote_loading);
  EXPECT_FALSE(cache_manager_->cached_requests_[req_id]->is_remote_loading_block_allocated);

  // G4b: req_state must be kRunning (blocks were allocated before Confirm).
  EXPECT_EQ(cache_manager_->cached_requests_[req_id]->req_state, RequestState::kRunning);

  cache_manager_->DestroyFinishedRequest(req_id);
}

// ============================================================================
// §6.5 — No memory leak on AllocateBlocksForRemoteCache OOM
// When the SWA allocator runs out of blocks mid-allocation, already-allocated
// FA blocks must be rolled back (returned to free_cached_blocks_).
// ============================================================================
TEST_F(PrefixCacheManagerRemoteCacheTest, AllocateBlocksForRemoteCache_OOMRollback) {
  // Create a cache manager with only 2 SWA blocks available.
  CacheManagerConfig tight_swa_config = cache_manager_config_;

  BlockAllocatorGroupConfig fa_group_config;
  fa_group_config.devices = {0, 1};
  fa_group_config.device_block_num = fa_device_block_num;
  fa_group_config.host_block_num = fa_host_block_num;
  fa_group_config.block_size = block_token_num * 1024 * 1024;
  auto fa_group = std::make_shared<FakedBlockAllocatorGroup>(fa_group_config, memory_allocator_, context_,
                                                             block_allocator_creation_fn_);

  // Only 2 SWA blocks total.
  BlockAllocatorGroupConfig tight_swa_group_config;
  tight_swa_group_config.devices = {0};
  tight_swa_group_config.device_block_num = 2;
  tight_swa_group_config.host_block_num = 0;
  tight_swa_group_config.block_size = block_token_num * 1024 * 1024;
  auto tight_swa_group = std::make_shared<FakedBlockAllocatorGroup>(tight_swa_group_config, memory_allocator_, context_,
                                                                    block_allocator_creation_fn_);

  PrefixCacheManager* tight_mgr = new PrefixCacheManager(tight_swa_config, fa_group, tight_swa_group);
  tight_mgr->InitializeCachedBlocks();

  // No prefix cache; request directly enters remote loading with 0 matched blocks.
  std::vector<int> token_ids;
  faked_token_generator_->GeneratePromptTokens({{5, 80}}, token_ids);  // 5 blocks

  int64_t req_id = 1;
  size_t fa_sb = 0, fa_ub = 0, fa_st = 0, swa_sb = 0, swa_ub = 0, swa_st = 0;
  ASSERT_TRUE(tight_mgr
                  ->GetRequestPrefixBlockNumberBeforeRemoteCacheLoading(req_id, token_ids, token_ids.size(), fa_sb,
                                                                        fa_ub, fa_st, swa_sb, swa_ub, swa_st)
                  .OK());

  size_t fa_free_before = tight_mgr->GetUsableBlockNumber();

  // Request 5 FA blocks and 5 SWA blocks, but only 2 SWA blocks are available → OOM.
  std::vector<int> req_block_ids;
  std::vector<int> remote_fa_block_ids;
  std::vector<int> remote_swa_block_ids;
  Status s = tight_mgr->AllocateBlocksForRemoteCache(req_id, req_block_ids, /*fa_block_num=*/5,
                                                     /*swa_block_num=*/5, remote_fa_block_ids, remote_swa_block_ids);
  // Allocation must fail.
  EXPECT_FALSE(s.OK());

  // §6.5 — FA blocks must be rolled back: usable count unchanged.
  size_t fa_free_after = tight_mgr->GetUsableBlockNumber();
  EXPECT_EQ(fa_free_before, fa_free_after);

  // cached_blocks must not have grown (rollback restored the original size).
  size_t expected_cached_blocks = fa_sb;  // only the matched prefix blocks
  EXPECT_EQ(tight_mgr->cached_requests_[req_id]->cached_blocks.size(), expected_cached_blocks);

  tight_mgr->DestroyFinishedRequest(req_id);
  delete tight_mgr;
}

// ============================================================================
// §6.2 (extended) — AllocateBlocksForRemoteCache with swa_block_num=0
// When swa_block_num=0, no SWA blocks are allocated and
// remote_cache_swa_fa_block_ids remains empty.
// ============================================================================
TEST_F(PrefixCacheManagerRemoteCacheTest, AllocateBlocksForRemoteCache_NoSWABlocks) {
  // 3 full blocks = 48 tokens seed.
  std::vector<int> token_ids;
  faked_token_generator_->GeneratePromptTokens({{1, 48}}, token_ids);
  faked_token_generator_->GeneratePromptTokens({{2, 32}}, token_ids);  // total 80 tokens = 5 blocks

  int64_t seed_req_id = 1;
  int64_t req_id = 2;
  size_t fa_ub = 0, swa_ub = 0;
  // Seed only the first 48 tokens (3 blocks) into the prefix tree.
  PrepareRemoteLoadingRequest(seed_req_id, req_id, token_ids, fa_ub, swa_ub, /*seed_token_num=*/48);

  size_t fa_free_before = cache_manager_->GetUsableBlockNumber();
  size_t swa_free_before = swa_block_allocator_group_->GetDeviceBlockAllocator(0)->GetFreeBlockNumber();

  std::vector<int> req_block_ids;
  std::vector<int> remote_fa_block_ids;
  std::vector<int> remote_swa_block_ids;
  Status s =
      cache_manager_->AllocateBlocksForRemoteCache(req_id, req_block_ids, /*fa_block_num=*/2,
                                                   /*swa_block_num=*/0, remote_fa_block_ids, remote_swa_block_ids);
  ASSERT_TRUE(s.OK());

  // GetUsableBlockNumber() (= free + reusable) decreased by 5:
  //   • 3 already-matched shared blocks move from reusable→active (§4.0.4 registration).
  //   • 2 new FA blocks taken from free_cached_blocks_.
  EXPECT_EQ(fa_free_before - cache_manager_->GetUsableBlockNumber(), 5u);

  // No SWA blocks consumed.
  size_t swa_free_after = swa_block_allocator_group_->GetDeviceBlockAllocator(0)->GetFreeBlockNumber();
  EXPECT_EQ(swa_free_before, swa_free_after);

  // remote_swa_block_ids is empty.
  EXPECT_TRUE(remote_swa_block_ids.empty());

  // remote_cache_swa_fa_block_ids is empty.
  EXPECT_TRUE(cache_manager_->cached_requests_[req_id]->remote_cache_swa_fa_block_ids.empty());

  cache_manager_->DestroyFinishedRequest(req_id);
}

// ============================================================================
// §6.1 / §6.2 (extended) — Binding rule: only the FIRST swa_block_num FA blocks
// get SWA blocks; the remaining (fa_block_num - swa_block_num) FA blocks do not.
// ============================================================================
TEST_F(PrefixCacheManagerRemoteCacheTest, AllocateBlocksForRemoteCache_BindingRuleFirstN) {
  // No prefix cache; request directly enters remote loading.
  std::vector<int> token_ids;
  faked_token_generator_->GeneratePromptTokens({{6, 80}}, token_ids);  // 5 blocks

  int64_t req_id = 1;
  size_t fa_sb = 0, fa_ub = 0, fa_st = 0, swa_sb = 0, swa_ub = 0, swa_st = 0;
  ASSERT_TRUE(cache_manager_
                  ->GetRequestPrefixBlockNumberBeforeRemoteCacheLoading(req_id, token_ids, token_ids.size(), fa_sb,
                                                                        fa_ub, fa_st, swa_sb, swa_ub, swa_st)
                  .OK());
  ASSERT_EQ(fa_sb, 0u);

  // Allocate fa_block_num=5, swa_block_num=3.
  std::vector<int> req_block_ids;
  std::vector<int> remote_fa_block_ids;
  std::vector<int> remote_swa_block_ids;
  Status s =
      cache_manager_->AllocateBlocksForRemoteCache(req_id, req_block_ids, /*fa_block_num=*/5,
                                                   /*swa_block_num=*/3, remote_fa_block_ids, remote_swa_block_ids);
  ASSERT_TRUE(s.OK());

  // remote_fa_block_ids has 5 entries; remote_swa_block_ids has 3 entries.
  ASSERT_EQ(remote_fa_block_ids.size(), 5u);
  ASSERT_EQ(remote_swa_block_ids.size(), 3u);

  // The first 3 FA blocks (indices 0-2) have SWA blocks bound.
  const auto& swa_fa_ids = cache_manager_->cached_requests_[req_id]->remote_cache_swa_fa_block_ids;
  ASSERT_EQ(swa_fa_ids.size(), 3u);

  // swa_fa_ids must correspond to the FA blocks for which SWA blocks were allocated (the last 3 FA blocks).
  size_t swa_fa_block_idx =
      cache_manager_->cached_requests_[req_id]->cached_blocks.size() - remote_swa_block_ids.size();
  for (size_t k = 0; k < 3; ++k) {
    // swa_fa_ids[k] must equal cached_blocks[swa_fa_block_idx + k]->block_id.
    size_t expected_fa_id = cache_manager_->cached_requests_[req_id]->cached_blocks[swa_fa_block_idx + k]->block_id;
    EXPECT_EQ(swa_fa_ids[k], expected_fa_id);

    // SWA block is bound.
    EXPECT_GE(cache_manager_->swa_cache_manager_->GetSWAMemoryBlockId(swa_fa_ids[k]), 0);
  }

  // The first 2 FA blocks (indices 0-1) must NOT have SWA blocks bound.
  for (size_t k = 3; k < 5; ++k) {
    size_t fa_prefix = cache_manager_->cached_requests_[req_id]->fa_prefix_block_num;
    size_t fa_id = cache_manager_->cached_requests_[req_id]->cached_blocks[fa_prefix + k - 3]->block_id;
    EXPECT_EQ(cache_manager_->swa_cache_manager_->GetSWAMemoryBlockId(fa_id), -1);
  }

  cache_manager_->DestroyFinishedRequest(req_id);
}

// ============================================================================
// New tests for G1/G2/G3/G4/G5/G6/G7/G8 fixes
// ============================================================================

// ============================================================================
// T1 (G2): AllocateBlocksForRemoteCache sets is_remote_loading_block_allocated=true.
// T2 (G3): AllocateRequestBlocksInternal does NOT set req_state=kRunning when
//          is_remote_loading=true.
// ============================================================================
TEST_F(PrefixCacheManagerRemoteCacheTest, AllocateBlocksForRemoteCache_SetsBlockAllocatedFlag) {
  // No prefix cache; request directly enters remote loading.
  std::vector<int> token_ids;
  faked_token_generator_->GeneratePromptTokens({{7, 80}}, token_ids);  // 5 blocks

  int64_t req_id = 1;
  size_t fa_sb = 0, fa_ub = 0, fa_st = 0, swa_sb = 0, swa_ub = 0, swa_st = 0;
  ASSERT_TRUE(cache_manager_
                  ->GetRequestPrefixBlockNumberBeforeRemoteCacheLoading(req_id, token_ids, token_ids.size(), fa_sb,
                                                                        fa_ub, fa_st, swa_sb, swa_ub, swa_st)
                  .OK());

  // After GetRequestPrefixBlockNumberBeforeRemoteCacheLoading:
  // is_remote_loading=true, is_remote_loading_block_allocated=false (kRemoteLoading).
  EXPECT_TRUE(cache_manager_->cached_requests_[req_id]->is_remote_loading);
  EXPECT_FALSE(cache_manager_->cached_requests_[req_id]->is_remote_loading_block_allocated);
  // G3: req_state must still be kWaiting (not kRunning) because is_remote_loading=true.
  EXPECT_EQ(cache_manager_->cached_requests_[req_id]->req_state, RequestState::kWaiting);

  std::vector<int> req_block_ids;
  std::vector<int> remote_fa_block_ids;
  std::vector<int> remote_swa_block_ids;
  ASSERT_TRUE(cache_manager_
                  ->AllocateBlocksForRemoteCache(req_id, req_block_ids, /*fa_block_num=*/3,
                                                 /*swa_block_num=*/1, remote_fa_block_ids, remote_swa_block_ids)
                  .OK());

  // After AllocateBlocksForRemoteCache:
  // is_remote_loading=true, is_remote_loading_block_allocated=true (kRemoteLoadingBlockAllocated).
  EXPECT_TRUE(cache_manager_->cached_requests_[req_id]->is_remote_loading);
  EXPECT_TRUE(cache_manager_->cached_requests_[req_id]->is_remote_loading_block_allocated);
  // req_state must still be kWaiting (G3 guard prevents kRunning transition).
  EXPECT_EQ(cache_manager_->cached_requests_[req_id]->req_state, RequestState::kWaiting);

  cache_manager_->DestroyFinishedRequest(req_id);
}

// ============================================================================
// T3 (G4a): ConfirmRemoteCacheLoadingResult (no-SWA) without prior
//           AllocateBlocksForRemoteCache → req_state stays kWaiting.
// ============================================================================
TEST_F(PrefixCacheManagerRemoteCacheTest, ConfirmRemoteCache_NoSWA_NoBlockAllocated_StaysWaiting) {
  // Create a no-SWA cache manager.
  CacheManagerConfig no_swa_config = cache_manager_config_;
  no_swa_config.sliding_window_size = 0;

  BlockAllocatorGroupConfig fa_group_config;
  fa_group_config.devices = {0, 1};
  fa_group_config.device_block_num = fa_device_block_num;
  fa_group_config.host_block_num = fa_host_block_num;
  fa_group_config.block_size = block_token_num * 1024 * 1024;
  auto fa_group = std::make_shared<FakedBlockAllocatorGroup>(fa_group_config, memory_allocator_, context_,
                                                             block_allocator_creation_fn_);
  PrefixCacheManager* no_swa_mgr = new PrefixCacheManager(no_swa_config, fa_group, nullptr);
  no_swa_mgr->InitializeCachedBlocks();

  // Build token_ids: first 48 tokens (3 blocks) seeded into tree; last 32 tokens (2 blocks) are remote.
  std::vector<int> token_ids;
  faked_token_generator_->GeneratePromptTokens({{8, 48}}, token_ids);
  faked_token_generator_->GeneratePromptTokens({{9, 32}}, token_ids);  // total 80 tokens = 5 blocks

  // Seed request: only the first 48 tokens (3 blocks) so that the remote-loading
  // request sees fa_sb=3 (not 5).
  {
    std::vector<int> seed_tokens(token_ids.begin(), token_ids.begin() + 48);
    size_t sb = 0, ub = 0, st = 0;
    no_swa_mgr->GetRequestPrefixBlockNumber(1, seed_tokens, seed_tokens.size(), sb, ub, st);
    std::vector<int> rbi, srbi;
    std::vector<size_t> sbo;
    EXPECT_TRUE(no_swa_mgr->AllocateRequestBlocks(1, ub, rbi, srbi, sbo).OK());
    bool bm = false;
    EXPECT_TRUE(no_swa_mgr->UpdateRequestTokens(1, seed_tokens, seed_tokens.size(), rbi, bm, srbi, sbo).OK());
    no_swa_mgr->DestroyFinishedRequest(1);
  }

  int64_t req_id = 2;
  size_t fa_sb = 0, fa_ub = 0, fa_st = 0, swa_sb = 0, swa_ub = 0, swa_st = 0;
  ASSERT_TRUE(no_swa_mgr
                  ->GetRequestPrefixBlockNumberBeforeRemoteCacheLoading(req_id, token_ids, token_ids.size(), fa_sb,
                                                                        fa_ub, fa_st, swa_sb, swa_ub, swa_st)
                  .OK());
  ASSERT_EQ(fa_sb, 3u);  // only 3 blocks in tree

  // Confirm WITHOUT calling AllocateBlocksForRemoteCache first.
  // is_remote_loading_block_allocated remains false → req_state must stay kWaiting.
  std::vector<int> kvcached_token_ids(48, 0);
  Status s = no_swa_mgr->ConfirmRemoteCacheLoadingResult(req_id, kvcached_token_ids,
                                                         /*shareable_kvcache_token_num=*/48,
                                                         /*matched_remote_swa_block_indexs=*/{});
  ASSERT_TRUE(s.OK());

  EXPECT_FALSE(no_swa_mgr->cached_requests_[req_id]->is_remote_loading);
  EXPECT_FALSE(no_swa_mgr->cached_requests_[req_id]->is_remote_loading_block_allocated);
  // G4a: no blocks allocated → kWaiting.
  EXPECT_EQ(no_swa_mgr->cached_requests_[req_id]->req_state, RequestState::kWaiting);

  no_swa_mgr->DestroyFinishedRequest(req_id);
  delete no_swa_mgr;
}

// ============================================================================
// T4 (G7): ConfirmRemoteCacheLoadingResult SWA Case 1 (no match) frees the
//          remote SWA block via FreeSWABlock; SWA allocator count is restored.
// ============================================================================
TEST_F(PrefixCacheManagerRemoteCacheTest, ConfirmRemoteCache_SWA_Case1_FreeSWABlockCalled) {
  // Seed 4 full blocks = 64 tokens.
  std::vector<int> token_ids;
  faked_token_generator_->GeneratePromptTokens({{10, 64}}, token_ids);
  faked_token_generator_->GeneratePromptTokens({{11, 16}}, token_ids);  // total 80 tokens = 5 blocks

  int64_t seed_req_id = 1;
  int64_t req_id = 2;
  size_t fa_ub = 0, swa_ub = 0;
  PrepareRemoteLoadingRequest(seed_req_id, req_id, token_ids, fa_ub, swa_ub, /*seed_token_num=*/64);

  // Record SWA free count before allocating the remote SWA block.
  size_t swa_free_before = swa_block_allocator_group_->GetDeviceBlockAllocator(0)->GetFreeBlockNumber();

  std::vector<int> req_block_ids;
  std::vector<int> remote_fa_block_ids;
  std::vector<int> remote_swa_block_ids;
  ASSERT_TRUE(cache_manager_
                  ->AllocateBlocksForRemoteCache(req_id, req_block_ids, /*fa_block_num=*/1,
                                                 /*swa_block_num=*/1, remote_fa_block_ids, remote_swa_block_ids)
                  .OK());

  // One SWA block consumed.
  size_t swa_free_after_alloc = swa_block_allocator_group_->GetDeviceBlockAllocator(0)->GetFreeBlockNumber();
  EXPECT_EQ(swa_free_before - swa_free_after_alloc, 1u);

  // Confirm with no match (Case 1): the remote SWA block must be freed.
  ASSERT_FALSE(cache_manager_->cached_requests_[req_id]->swa_window_block_idxs.empty());
  size_t swa_aligned_block_num = cache_manager_->cached_requests_[req_id]->swa_window_block_idxs.back() + 1;
  std::vector<int> kvcached_token_ids(swa_aligned_block_num * block_token_num, 0);
  Status s = cache_manager_->ConfirmRemoteCacheLoadingResult(
      req_id, kvcached_token_ids,
      /*shareable_kvcache_token_num=*/swa_aligned_block_num * block_token_num,
      /*matched_remote_swa_block_indexs=*/{});
  ASSERT_TRUE(s.OK());

  // G7: SWA block freed → count restored to swa_free_before (the window SWA blocks
  // are still held by the request, so only the remote SWA block is freed here).
  // The swa_window_fa_block_ids blocks are released by DestroyFinishedRequest.
  size_t swa_free_after_confirm = swa_block_allocator_group_->GetDeviceBlockAllocator(0)->GetFreeBlockNumber();
  // The remote SWA block (1 block) must have been freed back.
  EXPECT_EQ(swa_free_after_confirm, swa_free_after_alloc + 1u);

  // remote_cache_swa_fa_block_ids cleared.
  EXPECT_TRUE(cache_manager_->cached_requests_[req_id]->remote_cache_swa_fa_block_ids.empty());

  cache_manager_->DestroyFinishedRequest(req_id);
}

// ============================================================================
// T5 (G5): DestroyFinishedRequest in kRemoteLoadingBlockAllocated sub-state
//          correctly calls TruncateRequestBlocks (FA blocks returned to free pool).
// ============================================================================
TEST_F(PrefixCacheManagerRemoteCacheTest, DestroyFinishedRequest_kRemoteLoadingBlockAllocated_FreesBlocks) {
  // No prefix cache; request directly enters remote loading.
  std::vector<int> token_ids;
  faked_token_generator_->GeneratePromptTokens({{12, 80}}, token_ids);  // 5 blocks

  int64_t req_id = 1;
  size_t fa_sb = 0, fa_ub = 0, fa_st = 0, swa_sb = 0, swa_ub = 0, swa_st = 0;
  ASSERT_TRUE(cache_manager_
                  ->GetRequestPrefixBlockNumberBeforeRemoteCacheLoading(req_id, token_ids, token_ids.size(), fa_sb,
                                                                        fa_ub, fa_st, swa_sb, swa_ub, swa_st)
                  .OK());

  size_t fa_free_before = cache_manager_->GetUsableBlockNumber();

  std::vector<int> req_block_ids;
  std::vector<int> remote_fa_block_ids;
  std::vector<int> remote_swa_block_ids;
  ASSERT_TRUE(cache_manager_
                  ->AllocateBlocksForRemoteCache(req_id, req_block_ids, /*fa_block_num=*/5,
                                                 /*swa_block_num=*/1, remote_fa_block_ids, remote_swa_block_ids)
                  .OK());

  // Verify kRemoteLoadingBlockAllocated sub-state.
  EXPECT_TRUE(cache_manager_->cached_requests_[req_id]->is_remote_loading);
  EXPECT_TRUE(cache_manager_->cached_requests_[req_id]->is_remote_loading_block_allocated);

  size_t fa_free_after_alloc = cache_manager_->GetUsableBlockNumber();
  EXPECT_LT(fa_free_after_alloc, fa_free_before);  // blocks consumed

  // Destroy while in kRemoteLoadingBlockAllocated state.
  cache_manager_->DestroyFinishedRequest(req_id);

  // G5: FA blocks must be returned to the free pool (TruncateRequestBlocks called).
  size_t fa_free_after_destroy = cache_manager_->GetUsableBlockNumber();
  EXPECT_EQ(fa_free_after_destroy, fa_free_before);
}

// ============================================================================
// T6 (G5): DestroyFinishedRequest in kRemoteLoading sub-state (no blocks
//          allocated) does NOT call TruncateRequestBlocks; FA pool unchanged.
// ============================================================================
TEST_F(PrefixCacheManagerRemoteCacheTest, DestroyFinishedRequest_kRemoteLoading_NoBlocksFreed) {
  // No prefix cache; request directly enters remote loading.
  std::vector<int> token_ids;
  faked_token_generator_->GeneratePromptTokens({{13, 80}}, token_ids);  // 5 blocks

  int64_t req_id = 1;
  size_t fa_sb = 0, fa_ub = 0, fa_st = 0, swa_sb = 0, swa_ub = 0, swa_st = 0;
  ASSERT_TRUE(cache_manager_
                  ->GetRequestPrefixBlockNumberBeforeRemoteCacheLoading(req_id, token_ids, token_ids.size(), fa_sb,
                                                                        fa_ub, fa_st, swa_sb, swa_ub, swa_st)
                  .OK());

  // Verify kRemoteLoading sub-state (no AllocateBlocksForRemoteCache called).
  EXPECT_TRUE(cache_manager_->cached_requests_[req_id]->is_remote_loading);
  EXPECT_FALSE(cache_manager_->cached_requests_[req_id]->is_remote_loading_block_allocated);

  size_t fa_free_before_destroy = cache_manager_->GetUsableBlockNumber();

  // Destroy while in kRemoteLoading state (no blocks allocated).
  cache_manager_->DestroyFinishedRequest(req_id);

  // G5: No TruncateRequestBlocks called; FA pool unchanged.
  size_t fa_free_after_destroy = cache_manager_->GetUsableBlockNumber();
  EXPECT_EQ(fa_free_after_destroy, fa_free_before_destroy);
}
