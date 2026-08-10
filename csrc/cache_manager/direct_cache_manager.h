/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "cache_manager/base_cache_manager.h"
#include "cache_manager/cache_manager_interface.h"
#include "cache_manager/mamba_cache_manager.h"
#include "cache_manager/swa_cache_manager.h"

#include "configure/environment.h"
#include "runtime/request_state.h"
#include "runtime/threadpool.h"

namespace ksana_llm {

struct DirectCachedRequest;

// Describe a cached data, on either device or host.
struct DirectCachedBlock {
  // The unique id of this block, independent with content, not changed after created.
  size_t block_id = 0;

  // Whether the block is on device or host.
  bool is_device_location = true;

  // The block id shared by all devices.
  // If this cached block is swaped out, the value is host block id.
  int memory_block_id = -1;
};

// Describe the cache information for a infer request.
struct DirectCachedRequest {
  // The id of this request, as same as the id of InferRequest.
  int64_t req_id;

  // kvcached token num, used to get next step block number.
  size_t kvcached_token_num;

  // The cached blocks associated with this request, not include root node.
  // Contain all blocks even when request is waiting, or blocks that haved been swapped.
  std::vector<DirectCachedBlock*> cached_blocks;

  // FA block ids that are currently inside this request's SWA window.
  // The list is ordered by block index (smallest index first).
  // Used to call FreeSWABlock for each id when the request ends or the SWA
  // window slides past a block in UpdateRequestTokens.
  std::vector<size_t> swa_window_fa_block_ids;

  // Per-block FA global index: swa_window_block_idxs[i] is the FA block global
  // index corresponding to swa_window_fa_block_ids[i].
  // Must always have the same length as swa_window_fa_block_ids.
  // Replaces the old scalar swa_window_start_block_idx to support non-contiguous windows.
  std::vector<size_t> swa_window_block_idxs;
};

// Used to support prefix caching.
class DirectCacheManager : public CacheManagerInterface,
                           public BaseCacheManager<DirectCachedBlock, DirectCachedRequest> {
 public:
  explicit DirectCacheManager(const CacheManagerConfig& cache_manager_config,
                              std::shared_ptr<BlockAllocatorGroupInterface> block_allocator_group = nullptr,
                              std::shared_ptr<BlockAllocatorGroupInterface> swa_block_allocator_group = nullptr,
                              std::shared_ptr<BlockAllocatorGroupInterface> mamba_block_allocator_group = nullptr);

  // Initialize all the memory blocks.
  void InitializeCachedBlocks() override;

  std::shared_ptr<BlockAllocatorGroupInterface> GetBlockAllocatorGroup() const override;

  // Get all usable block number, including free and reusable ones.
  size_t GetUsableBlockNumber() override;

  // The value is from block manager.
  size_t GetHostFreeBlockNumber() override;

  // The value is from block manager.
  size_t GetUsedBlockNumber() override;

  // Calculate the actual number of unallocated blocks by passing the input length and obtaining the required block
  // number for the specific request.
  size_t GetRequestStepBlockNumber(int64_t req_id, size_t input_token_lens, size_t mtp_step_num = 0) override;

  // Get the needed block num for specific request if only one next token.
  size_t GetRequestStepBlockNumberForOneNextToken(int64_t req_id) override;

  // Get the usable block num for specific request.
  size_t GetRequestUsableBlockNumber(int64_t req_id) override;

  // Check the block num of specific request, the token number must be enough for next generation.
  // The shared block num always 0.
  Status GetRequestPrefixBlockNumber(int64_t req_id, const std::vector<int>& input_token_ids, size_t check_token_num,
                                     size_t& shared_block_num, size_t& unique_block_num,
                                     size_t& shared_token_num) override;

  // Not supported by DirectCacheManager; all FA/SWA shared counts are 0.
  // FA group (first): fa_shared_block_num, fa_unique_block_num, fa_shared_token_num.
  // SWA group (second): swa_shared_block_num, swa_unique_block_num, swa_shared_token_num.
  Status GetRequestPrefixBlockNumberBeforeRemoteCacheLoading(int64_t req_id, const std::vector<int>& input_token_ids,
                                                             size_t check_token_num, size_t& fa_shared_block_num,
                                                             size_t& fa_unique_block_num, size_t& fa_shared_token_num,
                                                             size_t& swa_shared_block_num, size_t& swa_unique_block_num,
                                                             size_t& swa_shared_token_num) override;

  // Not supported by DirectCacheManager; returns RET_RUNTIME_FAILED.
  Status AllocateBlocksForRemoteCache(int64_t /*req_id*/, std::vector<int>& /*req_block_ids*/, size_t /*fa_block_num*/,
                                      size_t /*swa_block_num*/, std::vector<int>& /*remote_fa_block_ids*/,
                                      std::vector<int>& /*remote_swa_block_ids*/) override {
    return Status(RET_RUNTIME_FAILED, "AllocateBlocksForRemoteCache is not supported by DirectCacheManager.");
  }

  // Not supported by DirectCacheManager; returns RET_RUNTIME_FAILED.
  Status ConfirmRemoteCacheLoadingResult(int64_t /*req_id*/, const std::vector<int>& /*kvcached_token_ids*/,
                                         size_t /*shareable_kvcache_token_num*/,
                                         const std::vector<size_t>& /*matched_remote_swa_block_indexs*/) override {
    return Status(RET_RUNTIME_FAILED, "ConfirmRemoteCacheLoadingResult is not supported by DirectCacheManager.");
  }

  // Allocate new blocks for request, called only when req is running.
  // mamba_kv_cache_block_id 出参：Direct 在分配内完成 working block 分配后写回。
  Status AllocateRequestBlocks(int64_t req_id, size_t block_num, std::vector<int>& req_block_ids,
                               std::vector<int>& swa_req_block_ids,
                               std::vector<size_t>& swa_block_idx_offsets,
                               int& mamba_kv_cache_block_id) override;

  // Update the token ids of this request.
  Status UpdateRequestTokens(int64_t req_id, const std::vector<int>& kvcached_token_ids,
                             size_t shareable_kvcache_token_num, std::vector<int>& req_block_ids, bool& block_merged,
                             std::vector<int>& swa_req_block_ids, std::vector<size_t>& swa_block_idx_offsets) override;

  void UpdateFlexibleCache(int64_t req_id, const std::vector<int>& token_ids, int shared_token_num,
                           std::vector<FlexibleCachedCopyTask>& flexible_cached_copy_tasks,
                           size_t& req_flexible_cache_len) override {}

  // Update internal state after request finished.
  void DestroyFinishedRequest(int64_t req_id) override;

  bool IsPrefixCachingEnabled() override;

 private:
  // SWA cache manager; null when SWA is not configured.
  std::unique_ptr<SWACacheManager> swa_cache_manager_;

  // Mamba state manager; null for models without Mamba state.
  std::unique_ptr<MambaCacheManager> mamba_cache_manager_;

  // Create a new cached block.
  DirectCachedBlock* CreateCachedBlock(size_t block_id);

  // Create a temporarily cached block, no block id generated.
  DirectCachedBlock* CreateEmptyCachedBlock();
};

}  // namespace ksana_llm
