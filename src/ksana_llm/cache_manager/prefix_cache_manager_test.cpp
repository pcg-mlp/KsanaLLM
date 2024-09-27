/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/cache_manager/prefix_cache_manager.h"

#include <gtest/gtest.h>

#include <memory>
#include <utility>

#include "ksana_llm/runtime/request_state.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/memory_utils.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/string_utils.h"
#include "test.h"

#include "ksana_llm/cache_manager/prefix_cache_manager_test_helper.h"

using namespace ksana_llm;

class PrefixCacheManagerTest : public testing::Test {
 protected:
  void SetUp() override {
    block_manager_config.host_allocator_config.blocks_num = device_block_num;
    block_manager_config.device_allocator_config.blocks_num = host_block_num;
    block_manager_config.device_allocator_config.block_token_num = block_token_num;

    faked_block_manager = new FakedBlockManager(block_manager_config, tensor_para_size);
    faked_block_manager->PreAllocateBlocks();
    SetBlockManager(faked_block_manager);

    cache_manager_config.block_token_num = 16;
    cache_manager_config.tensor_para_size = 2;
    cache_manager_config.swap_threadpool_size = 2;
    cache_manager_config.enable_prefix_caching = true;

    cache_manager = new PrefixCacheManager(cache_manager_config);

    faked_token_generator = new FakedTokenGenerator();

    block_token_num = block_manager_config.device_allocator_config.block_token_num;
  }

  void TearDown() override {
    delete faked_token_generator;
    delete cache_manager;
    delete faked_block_manager;
  }

 protected:
  BlockManagerConfig block_manager_config;

  CacheManagerConfig cache_manager_config;

  PrefixCacheManager* cache_manager = nullptr;

  FakedBlockManager* faked_block_manager = nullptr;
  FakedTokenGenerator* faked_token_generator = nullptr;

  size_t device_block_num = 100;
  size_t host_block_num = 100;

  size_t block_token_num = 16;
  size_t tensor_para_size = 2;
};

TEST_F(PrefixCacheManagerTest, SingleRequestTest) {
  cache_manager->InitializeCachedBlocks();

  // All blocks should be used.
  EXPECT_EQ(GetBlockManager()->GetDeviceFreeBlockNumber(), 0);
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
  cache_manager->GetRequestPrefixBlockNumber(req_id, output_token_ids, shared_block_num, unique_block_num,
                                             shared_token_num);

  // Check request state.
  EXPECT_EQ(cache_manager->cached_requests_.size(), 1);
  EXPECT_EQ(cache_manager->cached_requests_[req_id]->shared_block_num, shared_block_num);
  EXPECT_EQ(cache_manager->cached_requests_[req_id]->req_state, RequestState::REQUEST_STATE_WAITING);
  EXPECT_EQ(cache_manager->cached_requests_[req_id]->cached_blocks.size(), shared_block_num);

  // No shared block, all unqiue block.
  EXPECT_EQ(shared_block_num, 0);
  EXPECT_EQ(unique_block_num, (82 + block_token_num - 1) / block_token_num);

  // No block on tree.
  EXPECT_EQ(cache_manager->root_cached_block_->children.size(), 0);

  // Allocate request block.
  std::vector<std::vector<int>> req_block_ids;
  req_block_ids.resize(tensor_para_size);
  Status status = cache_manager->AllocateRequestBlocks(req_id, unique_block_num, req_block_ids);
  EXPECT_TRUE(status.OK());

  // Check memory number.
  EXPECT_EQ(req_block_ids[0].size(), unique_block_num);
  EXPECT_EQ(req_block_ids[1].size(), unique_block_num);

  // Recheck usable block num.
  EXPECT_EQ(cache_manager->GetUsableBlockNumber(), device_block_num - unique_block_num);

  // Generate new token and update request.
  faked_token_generator->GenerateOneToken(1, output_token_ids);
  status = cache_manager->UpdateRequestTokens(req_id, output_token_ids, req_block_ids);
  EXPECT_TRUE(status.OK());

  // All blocks except last should be on tree, check state of every block.
  size_t req_block_num = cache_manager->cached_requests_[req_id]->cached_blocks.size();
  for (size_t i = 0; i < req_block_num; ++i) {
    PrefixCachedBlock* cb = cache_manager->cached_requests_[req_id]->cached_blocks[i];
    PrefixCachedBlock* cb_prev =
        (i == 0) ? cache_manager->root_cached_block_ : cache_manager->cached_requests_[req_id]->cached_blocks[i - 1];

    if (i != req_block_num - 1) {
      EXPECT_EQ(cb->parent, cb_prev);
      EXPECT_EQ(cb->parent->children[cb->hash_code].front(), cb);
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

  // Check freeable block of this request.
  size_t freeable_block_num;
  status = cache_manager->GetRequestFreeableBlockNum(req_id, freeable_block_num);
  EXPECT_TRUE(status.OK());
  EXPECT_EQ(freeable_block_num, unique_block_num);

  // Swap out this request.
  size_t swapped_block_num = 0;
  size_t free_block_num = 0;
  status = cache_manager->SwapoutRequestAsync(req_id, swapped_block_num, free_block_num);
  EXPECT_TRUE(status.OK());

  // Check swapped and free block num.
  EXPECT_EQ(swapped_block_num, unique_block_num);
  EXPECT_EQ(free_block_num, 0);

  // Wait swapout done, and merge it.
  size_t left_req_num = 0;
  std::vector<int64_t> req_ids;
  status = cache_manager->WaitSwapoutRequests(req_ids, left_req_num, true);
  EXPECT_TRUE(status.OK());

  // All blocks should be on host, no parent, no children, no active reqeust, some inactive reqeust.
  req_block_num = cache_manager->cached_requests_[req_id]->cached_blocks.size();
  for (size_t i = 0; i < req_block_num; ++i) {
    PrefixCachedBlock* cb = cache_manager->cached_requests_[req_id]->cached_blocks[i];

    EXPECT_EQ(cb->parent, nullptr);
    EXPECT_EQ(cb->children.size(), 0);
    EXPECT_EQ(cb->active_requests.size(), 0);
    if (cb->is_shareable) {
      EXPECT_EQ(cb->inactive_requests.size(), 1);
    } else {
      EXPECT_EQ(cb->inactive_requests.size(), 0);
    }
    EXPECT_EQ(cb->is_device_location, false);
    if (i != req_block_num - 1) {
      EXPECT_EQ(cb->is_shareable, true);
    } else {
      EXPECT_EQ(cb->is_shareable, false);
    }
  }

  // Recheck block num, the block is not usable before merged.
  EXPECT_EQ(cache_manager->GetUsableBlockNumber(), device_block_num - unique_block_num);
  EXPECT_EQ(cache_manager->GetHostFreeBlockNumber(), host_block_num - (unique_block_num * tensor_para_size));

  // Merge swapped blocks.
  for (int64_t id : req_ids) {
    status = cache_manager->MergeSwapoutRequest(id);
    EXPECT_TRUE(status.OK());
  }

  // Recheck block num, now the block should usable again.
  EXPECT_EQ(cache_manager->GetUsableBlockNumber(), device_block_num);
  EXPECT_EQ(cache_manager->GetHostFreeBlockNumber(), host_block_num - (unique_block_num * tensor_para_size));

  // Swap in request again.

  size_t swapin_need_block_num;
  status = cache_manager->GetRequestNeededBlockNum(req_id, swapin_need_block_num);
  EXPECT_TRUE(status.OK());

  // Check needed block num.
  EXPECT_EQ(swapin_need_block_num, unique_block_num);

  // Start to swapin.
  size_t swapin_block_num;
  status = cache_manager->SwapinRequestAsync(req_id, swapin_block_num, req_block_ids);
  EXPECT_TRUE(status.OK());

  // check swapin block num.
  EXPECT_EQ(swapin_block_num, swapin_need_block_num);

  // Check block is unusable at the begining of swapin.
  EXPECT_EQ(cache_manager->GetUsableBlockNumber(), device_block_num - unique_block_num);

  // Wait swap in done
  size_t swapin_left_req_num;
  std::vector<int64_t> swapin_req_ids;
  status = cache_manager->WaitSwapinRequests(swapin_req_ids, swapin_left_req_num, true);
  EXPECT_TRUE(status.OK());

  // All blocks should be on device, but no parent, no children, no active reqeust, some inactive reqeust.
  req_block_num = cache_manager->cached_requests_[req_id]->cached_blocks.size();
  for (size_t i = 0; i < req_block_num; ++i) {
    PrefixCachedBlock* cb = cache_manager->cached_requests_[req_id]->cached_blocks[i];

    EXPECT_EQ(cb->parent, nullptr);
    EXPECT_EQ(cb->children.size(), 0);
    EXPECT_EQ(cb->active_requests.size(), 0);
    if (cb->is_shareable) {
      EXPECT_EQ(cb->inactive_requests.size(), 1);
    } else {
      EXPECT_EQ(cb->inactive_requests.size(), 0);
    }
    EXPECT_EQ(cb->is_device_location, true);
    if (i != req_block_num - 1) {
      EXPECT_EQ(cb->is_shareable, true);
    } else {
      EXPECT_EQ(cb->is_shareable, false);
    }
  }

  // Recheck block num, the host block is usable here.
  EXPECT_EQ(cache_manager->GetHostFreeBlockNumber(), host_block_num);

  // Merge swapin request.
  for (int64_t id : swapin_req_ids) {
    status = cache_manager->MergeSwapinRequest(id, req_block_ids);
    EXPECT_TRUE(status.OK());
  }

  // All block is on the tree again.
  req_block_num = cache_manager->cached_requests_[req_id]->cached_blocks.size();
  for (size_t i = 0; i < req_block_num; ++i) {
    PrefixCachedBlock* cb = cache_manager->cached_requests_[req_id]->cached_blocks[i];
    PrefixCachedBlock* cb_prev =
        (i == 0) ? cache_manager->root_cached_block_ : cache_manager->cached_requests_[req_id]->cached_blocks[i - 1];

    if (i != req_block_num - 1) {
      EXPECT_EQ(cb->parent, cb_prev);
      EXPECT_EQ(cb->parent->children[cb->hash_code].front(), cb);
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
  cache_manager->GetRequestPrefixBlockNumber(req_id_2, output_token_ids_2, shared_block_num_2, unique_block_num_2,
                                             shared_token_num_2);

  // Should be 2 matched blocks and 2 unique block.
  EXPECT_EQ(shared_block_num_2, 2);
  EXPECT_EQ(unique_block_num_2, ((54 + block_token_num - 1) / block_token_num) - 2);

  std::vector<std::vector<int>> req_block_ids_2;
  req_block_ids_2.resize(tensor_para_size);
  status = cache_manager->AllocateRequestBlocks(req_id_2, unique_block_num_2, req_block_ids_2);
  EXPECT_TRUE(status.OK());

  // Create request 3.
  std::vector<int> output_token_ids_3;
  faked_token_generator->GeneratePromptTokens({std::make_pair(1, 36), std::make_pair(2, 18)}, output_token_ids_3);

  // Check needed block num.
  int64_t req_id_3 = 3;
  size_t shared_token_num_3 = 0;
  size_t shared_block_num_3 = 0;
  size_t unique_block_num_3 = 0;
  cache_manager->GetRequestPrefixBlockNumber(req_id_3, output_token_ids_3, shared_block_num_3, unique_block_num_3,
                                             shared_token_num_3);

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
  cache_manager->GetRequestPrefixBlockNumber(req_id_5, output_token_ids_5, shared_block_num_5, unique_block_num_5,
                                             shared_token_num_5);

  // Should be 2 matched blocks and 2 unique block, used to check recalc of prefix.
  EXPECT_EQ(shared_block_num_5, 2);
  EXPECT_EQ(unique_block_num_5, ((54 + block_token_num - 1) / block_token_num) - 2);

  std::vector<std::vector<int>> req_block_ids_3;
  req_block_ids_3.resize(tensor_para_size);
  status = cache_manager->AllocateRequestBlocks(req_id_3, unique_block_num_3, req_block_ids_3);
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
  status = cache_manager->UpdateRequestTokens(req_id_2, output_token_ids_2, req_block_ids_2);
  EXPECT_TRUE(status.OK());

  // update second request token, then the 3rd block should be merged.
  status = cache_manager->UpdateRequestTokens(req_id_3, output_token_ids_3, req_block_ids_3);
  EXPECT_TRUE(status.OK());

  // Recheck prefix block of req 5, should be changed.
  cache_manager->GetRequestPrefixBlockNumber(req_id_5, output_token_ids_5, shared_block_num_5, unique_block_num_5,
                                             shared_token_num_5);

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

  // Swap out req 2.
  size_t freeable_block_num_2;
  status = cache_manager->GetRequestFreeableBlockNum(req_id_2, freeable_block_num_2);
  EXPECT_TRUE(status.OK());

  // Here only 1 block could be freed, that is, the unique block of req 2.
  EXPECT_EQ(freeable_block_num_2, 1);

  size_t swapped_block_num_2 = 0;
  size_t free_block_num_2 = 0;
  status = cache_manager->SwapoutRequestAsync(req_id_2, swapped_block_num_2, free_block_num_2);
  EXPECT_TRUE(status.OK());

  // Swap out req 3.
  size_t freeable_block_num_3;
  status = cache_manager->GetRequestFreeableBlockNum(req_id_3, freeable_block_num_3);
  EXPECT_TRUE(status.OK());

  // Here 4 block could be freed, that is, the 3 shared ones and 1 unique block.
  EXPECT_EQ(freeable_block_num_3, 4);

  size_t swapped_block_num_3 = 0;
  size_t free_block_num_3 = 0;
  status = cache_manager->SwapoutRequestAsync(req_id_3, swapped_block_num_3, free_block_num_3);
  EXPECT_TRUE(status.OK());

  // 5 used, (2 shared) + (1 mergee) + 2 * (1 unique) = 5, the 1 unmatched was free when swapped out.
  EXPECT_EQ(cache_manager->GetUsableBlockNumber(), device_block_num - 5);
  EXPECT_EQ(cache_manager->GetHostFreeBlockNumber(), host_block_num - (5 * 2));

  // Wait all requests merge done.
  size_t left_req_num_2 = 0;
  std::vector<int64_t> req_ids_2;
  do {
    status = cache_manager->WaitSwapoutRequests(req_ids_2, left_req_num_2, true);
    EXPECT_TRUE(status.OK());

    // Merge swapped blocks.
    for (int64_t id : req_ids_2) {
      status = cache_manager->MergeSwapoutRequest(id);
      EXPECT_TRUE(status.OK());
    }
  } while (left_req_num_2 > 0);

  // Now all block is usable.
  EXPECT_EQ(cache_manager->GetUsableBlockNumber(), device_block_num);
  EXPECT_EQ(cache_manager->GetHostFreeBlockNumber(), host_block_num - (5 * 2));

  // Create req 4 that have same sequence with 2 and 3.
  std::vector<int> output_token_ids_4;
  faked_token_generator->GeneratePromptTokens({std::make_pair(1, 36), std::make_pair(2, 18)}, output_token_ids_4);

  int64_t req_id_4 = 4;
  size_t shared_token_num_4 = 0;
  size_t shared_block_num_4 = 0;
  size_t unique_block_num_4 = 0;
  cache_manager->GetRequestPrefixBlockNumber(req_id_4, output_token_ids_4, shared_block_num_4, unique_block_num_4,
                                             shared_token_num_4);

  std::vector<std::vector<int>> req_block_ids_4;
  req_block_ids_4.resize(tensor_para_size);
  status = cache_manager->AllocateRequestBlocks(req_id_4, unique_block_num_4, req_block_ids_4);
  EXPECT_TRUE(status.OK());

  // The req 4 have same data with req 2/3.
  status = cache_manager->UpdateRequestTokens(req_id_4, output_token_ids_4, req_block_ids_4);
  EXPECT_TRUE(status.OK());

  // Swap in req 2.
  size_t swapin_need_block_num_2;
  status = cache_manager->GetRequestNeededBlockNum(req_id_2, swapin_need_block_num_2);
  EXPECT_TRUE(status.OK());

  size_t swapin_block_num_2;
  status = cache_manager->SwapinRequestAsync(req_id_2, swapin_block_num_2, req_block_ids_2);
  EXPECT_TRUE(status.OK());

  // Wait req 2 swapped in.
  size_t swapin_left_req_num_2;
  std::vector<int64_t> swapin_req_ids_2;
  status = cache_manager->WaitSwapinRequests(swapin_req_ids_2, swapin_left_req_num_2, true);
  EXPECT_TRUE(status.OK());

  // 4 blocks for req 2 and req 4
  EXPECT_EQ(cache_manager->GetUsableBlockNumber(), device_block_num - (4 + 4));
  // only 1 unqiue block(2 device) for req 3 in host.
  EXPECT_EQ(cache_manager->GetHostFreeBlockNumber(), host_block_num - (1 * 2));

  // Merge swapin req 2.
  for (int64_t id : swapin_req_ids_2) {
    status = cache_manager->MergeSwapinRequest(id, req_block_ids_2);
    EXPECT_TRUE(status.OK());
  }

  // Now req 2's leading 3 block was merge to req 4.
  // and req 3 is shared the leading 3 block with req 2, so they should have 3 same blocks.
  EXPECT_EQ(cache_manager->cached_requests_[req_id_2]->cached_blocks[0],
            cache_manager->cached_requests_[req_id_3]->cached_blocks[0]);
  EXPECT_EQ(cache_manager->cached_requests_[req_id_2]->cached_blocks[0],
            cache_manager->cached_requests_[req_id_4]->cached_blocks[0]);
  EXPECT_EQ(cache_manager->cached_requests_[req_id_2]->cached_blocks[1],
            cache_manager->cached_requests_[req_id_3]->cached_blocks[1]);
  EXPECT_EQ(cache_manager->cached_requests_[req_id_2]->cached_blocks[1],
            cache_manager->cached_requests_[req_id_4]->cached_blocks[1]);
  EXPECT_EQ(cache_manager->cached_requests_[req_id_2]->cached_blocks[2],
            cache_manager->cached_requests_[req_id_3]->cached_blocks[2]);
  EXPECT_EQ(cache_manager->cached_requests_[req_id_2]->cached_blocks[2],
            cache_manager->cached_requests_[req_id_4]->cached_blocks[2]);

  // 3 shared + 2 * (1 unique)
  EXPECT_EQ(cache_manager->GetUsableBlockNumber(), device_block_num - (3 + 1 + 1));
  // 1 unqiue of req 3
  EXPECT_EQ(cache_manager->GetHostFreeBlockNumber(), host_block_num - (1 * 2));

  // Drop swapped req 3.
  cache_manager->DestroySwapedRequest(req_id_3);

  // All host memory is usable.
  EXPECT_EQ(cache_manager->GetUsableBlockNumber(), device_block_num - (3 + 1 + 1));
  EXPECT_EQ(cache_manager->GetHostFreeBlockNumber(), host_block_num);

  cache_manager->DestroyFinishedRequest(req_id_2);
  cache_manager->DestroyFinishedRequest(req_id_4);
}
