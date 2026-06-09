/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "cache_manager/base_cache_manager.h"
#include "cache_manager/cache_manager_interface.h"
#include "cache_manager/swa_cache_manager.h"

#include "configure/environment.h"
#include "runtime/request_state.h"
#include "runtime/threadpool.h"
#include "utils/calc_intvec_hash.h"

namespace ksana_llm {

struct PrefixCachedRequest;

// Describe a cached data, on either device or host.
struct PrefixCachedBlock {
  // ///////////////////////////////////////////////////////////////////////////
  // Common field for base class.

  // The unique id of this block, independent with content, not changed after created.
  size_t block_id = 0;

  // Whether the block is on device or host.
  bool is_device_location = true;

  // The block id shared by all devices.
  // If this cached block is swaped out, the value is host block id.
  int memory_block_id = -1;

  // ///////////////////////////////////////////////////////////////////////////
  // Prefix cached block only.

  // The hash code of this block's content.
  size_t hash_code = 0;

  // Whether it is the root node.
  bool is_root = false;

  // The parent block.
  PrefixCachedBlock* parent = nullptr;

  // The child blocks, key is hash code of tokens.
  // std::unordered_map<size_t, std::list<PrefixCachedBlock*>> children;
  std::unordered_map<std::vector<int>, PrefixCachedBlock*, TokensHash, TokensEqual> children;

  // Whether this block contain valid cache data and be shareable.
  bool is_shareable = false;

  // The token ids of this block.
  std::vector<int> token_ids;

  // All running or finished requests that reference this block.
  // pair.first is position index of current block in request.
  std::unordered_map<int, std::pair<int, PrefixCachedRequest*>> active_requests;
};

// Describe the cache information for a infer request.
struct PrefixCachedRequest {
  // The id of this request, as same as the id of InferRequest.
  int64_t req_id;

  // The request state, waiting or running or finished.
  RequestState req_state = RequestState::kWaiting;

  // Used to update filled cached block, increased only if one block is full.
  size_t shared_block_num = 0;

  // The cached blocks associated with this request, not include root node.
  // Contain all blocks even when request is waiting, or blocks that haved been swapped.
  std::vector<PrefixCachedBlock*> cached_blocks;

  // FA block ids that are currently inside this request's SWA window.
  // The list is ordered by block index (smallest index first).
  // Used to:
  //   1. Call SetReclaimable(true) for each id when the request ends.
  //   2. Call SetReclaimable(true) for ids that slide out of the window in
  //      UpdateRequestTokens.
  std::vector<size_t> swa_window_fa_block_ids;

  // Per-block FA global index: swa_window_block_idxs[i] is the FA block global
  // index corresponding to swa_window_fa_block_ids[i].
  // Must always have the same length as swa_window_fa_block_ids.
  // Replaces the old scalar swa_window_start_block_idx to support non-contiguous windows.
  std::vector<size_t> swa_window_block_idxs;

  // Whether the request is in remote_loading state.
  // Set to true after GetRequestPrefixBlockNumber(need_align_swa=false).
  // Set to false after ConfirmRemoteCacheLoadingResult.
  // While true, AllocateRequestBlocks, UpdateRequestTokens, and
  // AppendFilledCachedBlock must not be called.
  bool is_remote_loading = false;

  // Set to true after AllocateBlocksForRemoteCache succeeds.
  // Combined with is_remote_loading to encode two sub-states:
  //   is_remote_loading=true,  is_remote_loading_block_allocated=false → kRemoteLoading
  //   is_remote_loading=true,  is_remote_loading_block_allocated=true  → kRemoteLoadingBlockAllocated
  // Reset to false in ConfirmRemoteCacheLoadingResult and Destroy* methods.
  bool is_remote_loading_block_allocated = false;

  // When need_align_swa=false, the actual furthest FA cache hit block count
  // (not truncated by SWA alignment). Equals cached_blocks.size() after
  // GetRequestPrefixBlockNumberBeforeRemoteCacheLoading.
  // Used by AllocateBlocksForRemoteCache to:
  //   1. Determine the FA block range to load (new blocks start at fa_prefix_block_num).
  //   2. Re-scan cached_blocks[0..fa_prefix_block_num) to register SWA blocks for
  //      the cached FA blocks (deferred from the query phase per §4.0.3).
  size_t fa_prefix_block_num = 0;

  // FA block ids of the newly allocated FA blocks that have a SWA block bound to them,
  // as recorded by AllocateBlocksForRemoteCache.
  // remote_cache_swa_fa_block_ids[k] == cached_blocks[fa_prefix_block_num + k]->block_id.
  // After ConfirmRemoteCacheLoadingResult:
  //   - Matched entries (size() == 1): SetSWABlockContentFixed is called to mark data available.
  //   - Unmatched entries are freed via SWACacheManager::FreeSWABlock.
  //   - The list is cleared.
  // If the request is destroyed while is_remote_loading=true, all entries must be freed.
  std::vector<size_t> remote_cache_swa_fa_block_ids;
};

// Used to support prefix caching.
class PrefixCacheManager : public CacheManagerInterface,
                           public BaseCacheManager<PrefixCachedBlock, PrefixCachedRequest> {
 public:
  explicit PrefixCacheManager(const CacheManagerConfig& cache_manager_config,
                              std::shared_ptr<BlockAllocatorGroupInterface> block_allocator_group = nullptr,
                              std::shared_ptr<BlockAllocatorGroupInterface> swa_block_allocator_group = nullptr);
  virtual ~PrefixCacheManager();

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
  size_t GetRequestStepBlockNumber(int64_t req_id, size_t input_token_lens) override;

  // Get the needed block num for specific request if only one next token.
  size_t GetRequestStepBlockNumberForOneNextToken(int64_t req_id) override;

  // Get the usable block num for specific request.
  // The method will exclude the cached blocks of this request.
  size_t GetRequestUsableBlockNumber(int64_t req_id) override;

  // FA and SWA caches are aligned (need_swa_align=true).
  // shared_block_num: number of FA blocks matched (= last_fa_valid_idx, which equals last_swa_valid_idx when aligned).
  //                   When SWA is disabled this is simply the full FA-matched count.
  // unique_block_num: remaining FA blocks needed beyond shared_block_num to cover check_token_num.
  // shared_token_num: shared_block_num * block_token_num.
  Status GetRequestPrefixBlockNumber(int64_t req_id, const std::vector<int>& input_token_ids, size_t check_token_num,
                                     size_t& shared_block_num, size_t& unique_block_num,
                                     size_t& shared_token_num) override;
  // FA and SWA caches are NOT aligned (need_swa_align=false).
  // FA group (first): fa_shared_block_num, fa_unique_block_num, fa_shared_token_num.
  // SWA group (second): swa_shared_block_num, swa_unique_block_num, swa_shared_token_num.
  // Request enters kRemoteLoading state; caller must invoke AllocateBlocksForRemoteCache +
  // ConfirmRemoteCacheLoadingResult before inference.
  Status GetRequestPrefixBlockNumberBeforeRemoteCacheLoading(int64_t req_id, const std::vector<int>& input_token_ids,
                                                             size_t check_token_num, size_t& fa_shared_block_num,
                                                             size_t& fa_unique_block_num, size_t& fa_shared_token_num,
                                                             size_t& swa_shared_block_num, size_t& swa_unique_block_num,
                                                             size_t& swa_shared_token_num) override;
  // Allocate new blocks for request, called only when req is running.
  Status AllocateRequestBlocks(int64_t req_id, size_t block_num, std::vector<int>& req_block_ids,
                               std::vector<int>& swa_req_block_ids,
                               std::vector<size_t>& swa_block_idx_offsets) override;

  // Allocate blocks for remote cache loading (only valid in kRemoteLoading state).
  Status AllocateBlocksForRemoteCache(int64_t req_id, std::vector<int>& req_block_ids, size_t fa_block_num,
                                      size_t swa_block_num, std::vector<int>& remote_fa_block_ids,
                                      std::vector<int>& remote_swa_block_ids) override;

  // Confirm remote cache loading result and transition out of kRemoteLoading state.
  // See CacheManagerInterface::ConfirmRemoteCacheLoadingResult for full contract.
  Status ConfirmRemoteCacheLoadingResult(int64_t req_id, const std::vector<int>& kvcached_token_ids,
                                         size_t shareable_kvcache_token_num,
                                         const std::vector<size_t>& matched_remote_swa_block_indexs) override;

  // Update the token ids of this request.
  // This method will update request memory blocks if the origin block is merged and set block_merged as true.
  Status UpdateRequestTokens(int64_t req_id, const std::vector<int>& kvcached_token_ids,
                             size_t shareable_kvcache_token_num, std::vector<int>& req_block_ids, bool& block_merged,
                             std::vector<int>& swa_req_block_ids, std::vector<size_t>& swa_block_idx_offsets) override;

  // Update internal state after request finished.
  void DestroyFinishedRequest(int64_t req_id) override;

  // Free at least block_num cached blocks that could resued, reserve some blocks if needed.
  bool FreeCachedBlocks(size_t block_num, size_t& free_block_num,
                        const std::vector<PrefixCachedBlock*>& reserved_blocks = {});

  bool IsPrefixCachingEnabled() override;

 private:
  // Ensures free_cached_blocks_ has >= fa_block_num entries (calling FreeCachedBlocks if
  // needed), then allocates fa_block_num new FA blocks and appends them to
  // cached_request->cached_blocks.
  //
  // SWA block binding is query-driven: AcquireSWABlock is called for each block in
  // swa_window_fa_block_ids, then GetAllocatableSWACache(cur, cur + minimum_swa_block_num,
  // minimum_swa_block_num) determines which new FA block positions should receive SWA blocks.
  // Only those positions get AllocateSWABlock(); remaining FA blocks receive no SWA block.
  // GetAllocatableSWACache's return value (-1) is the sole SWA availability gate; no separate
  // GetAvailableBlockCount pre-check is performed.
  //
  // First synchronizes req_block_ids with cached_request->cached_blocks (invalid prefix
  // truncation, fill missing prefix memory ids, req_state transition when empty,
  // reusable/active_requests updates). Reserved blocks for eviction are then
  // cached_request->cached_blocks iff req_state is kWaiting or kRunning.
  //
  // Space-check and FreeCachedBlocks are owned by this method; callers do NOT need to
  // perform them before calling.
  //
  // Preconditions:
  //   - fa_block_num >= minimum_swa_block_num.
  //   - swa_cache_manager_ != nullptr when minimum_swa_block_num > 0.
  //
  // Output:
  //   - new_fa_block_ids: memory_block_id list of the fa_block_num newly allocated FA blocks.
  //   - new_swa_fa_offsets: FA block global indices (ascending) at which SWA blocks were
  //                         bound.  Length == GetAllocatableSWACache's output size (<= minimum_swa_block_num).
  //
  // Does NOT modify swa_window_fa_block_ids or swa_window_block_idxs.
  Status AllocateRequestBlocksInternal(PrefixCachedRequest* cached_request, int64_t req_id,
                                       std::vector<int>& req_block_ids, size_t fa_block_num,
                                       size_t minimum_swa_block_num, bool disable_swa_block_num_adjust,
                                       std::vector<int>& new_fa_block_ids, std::vector<int>& new_swa_fa_offsets);

  // Internal implementation shared by GetRequestPrefixBlockNumber and
  // GetRequestPrefixBlockNumberBeforeRemoteCacheLoading.
  //
  // need_swa_align=true:  cached_blocks is truncated to last_swa_valid_idx; fa_shared_block_num
  //                       equals swa_shared_block_num.
  // need_swa_align=false: cached_blocks retains the full FA-matched set; fa_shared_block_num may
  //                       exceed swa_shared_block_num; request enters kRemoteLoading state.
  //
  // Outputs (FA group first, SWA group second):
  //   fa_shared_block_num:  number of FA blocks matched (last_fa_valid_idx when need_swa_align=false,
  //                         otherwise equals swa_shared_block_num).
  //   fa_unique_block_num:  blocks needed beyond fa_shared_block_num to cover check_token_num.
  //   fa_shared_token_num:  fa_shared_block_num * block_token_num.
  //   swa_shared_block_num: number of FA blocks with a fully-valid SWA window (last_swa_valid_idx).
  //   swa_unique_block_num: blocks needed beyond swa_shared_block_num to cover check_token_num.
  //   swa_shared_token_num: swa_shared_block_num * block_token_num.
  Status GetRequestPrefixBlockNumberInternal(int64_t req_id, const std::vector<int>& input_token_ids,
                                             size_t check_token_num, bool need_swa_align, size_t& fa_shared_block_num,
                                             size_t& fa_unique_block_num, size_t& fa_shared_token_num,
                                             size_t& swa_shared_block_num, size_t& swa_unique_block_num,
                                             size_t& swa_shared_token_num);
  // Whether the block token is equal to specific ones.
  bool CheckSameTokens(const PrefixCachedBlock* block, const int* start, size_t len);

  // Find tokens from cached blocks, return the matched blocks, return nullptr if not found.
  PrefixCachedBlock* FindChildCacheBlock(PrefixCachedBlock* block, const int* start, size_t len);

  // Create a new cached block.
  PrefixCachedBlock* CreateCachedBlock(size_t block_id);

  // Create a temporarily cached block, no block id generated.
  PrefixCachedBlock* CreateEmptyCachedBlock();

  // The cached block must be reset before reused, keep memory block id unchanged.
  void ResetCachedBlock(PrefixCachedBlock* cached_block);

  // Recursive free block and its all children.
  void FreeCachedBlockRecursively(PrefixCachedBlock* cached_block, std::vector<PrefixCachedBlock*>& free_blocks,
                                  size_t& free_num);

  // Append or remove cached block to/from time block list.
  void AppendCachedBlockToTimedList(PrefixCachedBlock* cached_block);
  void RemoveCachedBlockFromTimedList(PrefixCachedBlock* cached_block);

  // Append specific filled block to the tree.
  Status AppendFilledCachedBlock(PrefixCachedRequest* cached_request, size_t block_index,
                                 PrefixCachedBlock* cached_block, std::vector<int>& req_block_ids, bool& block_merged);

  // Merge specific filled block to another.
  Status MergeFilledCachedBlocks(PrefixCachedRequest* cached_request, size_t block_index,
                                 PrefixCachedBlock* dst_cached_block, PrefixCachedBlock* src_cached_block,
                                 std::vector<int>& req_block_ids);

  void QueryIncompleteBlock(PrefixCachedBlock* last_block, const std::vector<int>& token_ids, int start_idx,
                            PrefixCachedBlock*& prefix_append_block, size_t& prefix_append_len);
  void EraseFlexibleCacheMap(PrefixCachedBlock* cached_block);
  void InsertFlexibleCacheMap(PrefixCachedBlock* cached_block);
  void UpdateFlexibleCache(int64_t req_id, const std::vector<int>& token_ids, int shared_token_num,
                           std::vector<FlexibleCachedCopyTask>& flexible_cached_copy_tasks,
                           size_t& req_flexible_cache_len);
  int GetPrefixLen(PrefixCachedBlock* block);

  // Release FA blocks (and their bound SWA blocks) in cached_request->cached_blocks
  // starting from trunc_start_offset up to the end of the list, then resize the list
  // to trunc_start_offset.
  //
  // For each released FA block:
  //   - If the block is not shareable: FreeSWABlock is called (if swa_cache_manager_ != nullptr),
  //     then the block is reset and returned to free_cached_blocks_.
  //   - If the block is shareable: the request is removed from active_requests;
  //     the block is inserted into reusable_cached_blocks_ when no references remain.
  //
  // Does NOT touch swa_window_fa_block_ids or remote_cache_swa_fa_block_ids; callers are
  // responsible for managing those lists before calling this method.
  void TruncateRequestBlocks(PrefixCachedRequest* cached_request, size_t trunc_start_offset);

 private:
  // SWA cache manager; null when SWA is not configured.
  std::unique_ptr<SWACacheManager> swa_cache_manager_;

  // The root block of block tree, not contain any memory block.
  // The tree struct only contain computed blocks, non-computed block is maintain by cached request itself.
  PrefixCachedBlock* root_cached_block_;
  std::unordered_map<std::vector<int>, std::unordered_set<PrefixCachedBlock*>, TokensHash, TokensEqual>
      flexible_cache_map_;
  // The cached blocks that have filled, but no request referenced it.
  std::unordered_set<PrefixCachedBlock*> reusable_cached_blocks_;

  // All the cached block in tree, sort by time when that node added into the tree.
  // Not contain the cached block that not in node tree.
  std::list<PrefixCachedBlock*> timed_cached_blocks_;
  std::unordered_map<PrefixCachedBlock*, std::list<PrefixCachedBlock*>::iterator> timed_cached_block_iters_;
};

}  // namespace ksana_llm
