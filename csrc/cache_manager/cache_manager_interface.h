/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <list>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "cache_manager/block_allocator/block_allocator_manager.h"
#include "utils/request.h"
#include "utils/status.h"

namespace ksana_llm {

class CacheManagerInterface {
 public:
  // Initialize all the memory blocks.
  virtual void InitializeCachedBlocks() = 0;

  virtual std::shared_ptr<BlockAllocatorGroupInterface> GetBlockAllocatorGroup() const = 0;

  // Get all usable block number, including free and reusable ones.
  virtual size_t GetUsableBlockNumber() = 0;

  // The value is from block mamanger.
  virtual size_t GetHostFreeBlockNumber() = 0;

  // Get all used block number
  virtual size_t GetUsedBlockNumber() = 0;

  // Calculate the actual number of unallocated blocks by passing the input length and obtaining the required block
  // number for the specific request.
  //
  // Reservation policy (single source of truth for both V1 and EDS):
  //   total_blocks = ceil((input_token_lens + 1) / block_token_num)        // +1 for the first sampler-produced
  //                + ceil(mtp_step_num         / block_token_num)          //   token after prefill, + a separately-
  //                                                                       //   reserved chunk for MTP lookahead.
  // The two terms are rounded up INDEPENDENTLY so that MTP always lands on its own block boundary (even when input's
  // tail block has free slots). This matches the "+1 block" design intent reused by the dispatcher in pd_v2 — Decode
  // and Prefill must end up with the same block count for the wire contract to hold.
  virtual size_t GetRequestStepBlockNumber(int64_t req_id, size_t input_token_lens, size_t mtp_step_num = 0) = 0;

  // Get the needed block num for specific request if only one next token.
  virtual size_t GetRequestStepBlockNumberForOneNextToken(int64_t req_id) = 0;

  // Get the usable block num for specific request.
  virtual size_t GetRequestUsableBlockNumber(int64_t req_id) = 0;

  // Check the shared and unique block num of specific request, the token number must be enough for next generation.
  // The unique_block_num will always large than 0.
  // check_token_num is the token number need to check. It is always equal or less than input_token_ids.size().
  // check_token_num is less than input_token_ids.size() during chunked prefilling.
  //
  // FA and SWA caches are aligned (need_swa_align=true internally).
  // shared_block_num: number of FA blocks matched (= last_fa_valid_idx, which equals last_swa_valid_idx when aligned).
  //                   When SWA is disabled this is simply the full FA-matched count.
  // unique_block_num: remaining FA blocks needed beyond shared_block_num to cover check_token_num.
  // shared_token_num: shared_block_num * block_token_num.
  // The request can enter inference immediately after this call.
  virtual Status GetRequestPrefixBlockNumber(int64_t req_id, const std::vector<int>& input_token_ids,
                                             size_t check_token_num, size_t& shared_block_num, size_t& unique_block_num,
                                             size_t& shared_token_num) = 0;

  // Variant used before remote cache loading (need_swa_align=false internally).
  // FA cache and SWA cache hit positions are allowed to be unaligned so that the FA hit length
  // is preserved for subsequent remote loading.
  //
  // FA group (output first):
  //   fa_shared_block_num:  number of FA blocks actually matched in the prefix tree (last_fa_valid_idx).
  //   fa_unique_block_num:  blocks needed beyond fa_shared_block_num to cover check_token_num.
  //   fa_shared_token_num:  fa_shared_block_num * block_token_num.
  //
  // SWA group (output second):
  //   swa_shared_block_num: number of FA blocks whose SWA window is fully valid (last_swa_valid_idx).
  //                         Always <= fa_shared_block_num.
  //   swa_unique_block_num: blocks needed beyond swa_shared_block_num to cover check_token_num.
  //   swa_shared_token_num: swa_shared_block_num * block_token_num (SWA-aligned token boundary).
  //
  // The request enters kRemoteLoading state and must call AllocateBlocksForRemoteCache
  // followed by ConfirmRemoteCacheLoadingResult before inference.
  virtual Status GetRequestPrefixBlockNumberBeforeRemoteCacheLoading(
      int64_t req_id, const std::vector<int>& input_token_ids, size_t check_token_num, size_t& fa_shared_block_num,
      size_t& fa_unique_block_num, size_t& fa_shared_token_num, size_t& swa_shared_block_num,
      size_t& swa_unique_block_num, size_t& swa_shared_token_num) = 0;

  // Allocate blocks for remote cache loading.
  // fa_block_num: number of FA blocks to allocate for remote loading
  //   (= target FA block total - already-hit FA block count).
  // swa_block_num: number of SWA blocks to allocate for remote loading
  //   (= SWA window size, independent of fa_block_num).
  //
  // Allocation logic:
  //   - New FA blocks are appended to cached_blocks (same as AllocateRequestBlocks).
  //   - New SWA blocks are bound to the first swa_block_num new FA blocks; FA ids are recorded in
  //     remote_cache_swa_fa_block_ids after success.
  //
  // req_block_ids: same role as in AllocateRequestBlocks — synchronized with prefix cached_blocks and
  //   appended with newly allocated FA memory_block_ids on success.
  // Returns:
  //   remote_fa_block_ids: memory_block_id list of newly allocated FA blocks (length=fa_block_num).
  //   remote_swa_block_ids: memory_block_id list of newly allocated SWA blocks (length=swa_block_num).
  virtual Status AllocateBlocksForRemoteCache(int64_t req_id, std::vector<int>& req_block_ids, size_t fa_block_num,
                                              size_t swa_block_num, std::vector<int>& remote_fa_block_ids,
                                              std::vector<int>& remote_swa_block_ids) = 0;

  // Confirm the result of remote cache loading and transition out of kRemoteLoading state.
  //
  // kvcached_token_ids: the actual token ids loaded into the remote cache blocks.
  //   Passed through to UpdateRequestTokens for newly loaded blocks.
  //
  // shareable_kvcache_token_num: number of tokens whose KV cache content is confirmed available.
  //   Preconditions (enforced; returns RET_RUNTIME_FAILED on violation):
  //   - If SWA is enabled:  >= token position of the rightmost token in swa_window_fa_block_ids.
  //   - If SWA is disabled: >= fa_prefix_block_num * block_token_num.
  //   This value must not reduce the request's existing coverage.
  //
  // matched_remote_swa_block_indexs: offsets into remote_cache_swa_fa_block_ids that matched.
  //   Size must be <= 1 (SWA window block count is fixed at 1).
  //   matched_remote_swa_block_indexs[0]=k means remote_cache_swa_fa_block_ids[k] holds the
  //   SWA content for the position at shareable_kvcache_token_num - 1.
  //
  // Processing (no SWA):
  //   1. Truncate cached_blocks to shareable_block_num = shareable_kvcache_token_num / block_token_num.
  //   2. If cached_blocks.size() > fa_prefix_block_num, call UpdateRequestTokens for new blocks.
  //
  // Processing (SWA enabled, window size == 1):
  //   Case 0 (no match): free all remote SWA blocks; truncate cached_blocks to fa_prefix_block_num.
  //   Case 1 (matched):  Transfer SWA block to cached_blocks[shareable_block_num-1]; update
  //                      swa_window_fa_block_ids; truncate cached_blocks to shareable_block_num;
  //                      if shareable_block_num > fa_prefix_block_num, call UpdateRequestTokens.
  virtual Status ConfirmRemoteCacheLoadingResult(int64_t req_id, const std::vector<int>& kvcached_token_ids,
                                                 size_t shareable_kvcache_token_num,
                                                 const std::vector<size_t>& matched_remote_swa_block_indexs) = 0;

  // Allocate new blocks for request, called only when req is running.
  // swa_req_block_ids and swa_block_idx_offsets are filled only when swa_block_allocator_group is configured.
  // mamba_kv_cache_block_id（出参，同 SWA out 参数契约）：mamba working block 物理 ID，
  // 未配置 mamba 或分配失败为 -1；随 FA/SWA 分配原子写回，调用方无需二次查询。
  virtual Status AllocateRequestBlocks(int64_t req_id, size_t block_num, std::vector<int>& req_block_ids,
                                       std::vector<int>& swa_req_block_ids,
                                       std::vector<size_t>& swa_block_idx_offsets,
                                       int& mamba_kv_cache_block_id) = 0;

  virtual void UpdateFlexibleCache(int64_t req_id, const std::vector<int>& token_ids, int shared_token_num,
                                   std::vector<FlexibleCachedCopyTask>& flexible_cached_copy_tasks,
                                   size_t& req_flexible_cache_len) = 0;

  // Update the token ids of this request.
  // This method will update request memory blocks if the origin block is merged and set block_merged as true.
  // swa_req_block_ids and swa_block_idx_offsets are updated only when swa_block_allocator_group is configured.
  virtual Status UpdateRequestTokens(int64_t req_id, const std::vector<int>& kvcached_token_ids,
                                     size_t shareable_kvcache_token_num, std::vector<int>& req_block_ids,
                                     bool& block_merged, std::vector<int>& swa_req_block_ids,
                                     std::vector<size_t>& swa_block_idx_offsets) = 0;

  // Release the SWA blocks bound to FA block indices below keep_start_block_idx.
  //
  // Called after a global cache hit extends the request's prefix past its local prefix-cache hit:
  // the FA blocks in that range hold KV content fetched from the global store, but their SWA blocks
  // were never written.  Without this call UpdateRequestTokens would merge those FA blocks into the
  // prefix tree and AppendFilledCachedBlock would mark their garbage SWA content as valid.
  //
  // keep_start_block_idx: the first FA block index whose SWA must be kept.  Callers pass
  //   (hit_block_num - swa_window_block_num) so the boundary window that actually received global
  //   SWA content survives; every SWA binding before it is released.
  //
  // Shareable (in-tree) FA blocks only get SetReclaimable — their SWA blocks may be shared with
  // other requests and must not be freed.  Non-shareable blocks are freed outright.
  //
  // swa_req_block_ids / swa_block_idx_offsets are rebuilt from the surviving window so the caller
  // can write them back to the InferRequest; leaving stale memory ids there would let the model
  // write into SWA memory that has been handed to another request.
  //
  // No-op returning OK when SWA is not enabled.
  virtual Status DropUnbackedSwaBlocks(int64_t req_id, size_t keep_start_block_idx, std::vector<int>& swa_req_block_ids,
                                       std::vector<size_t>& swa_block_idx_offsets) = 0;

  // Park SWA blocks that slide out of the window instead of reclaiming them, until EndSwaPutHold.
  //
  // The executor uploads a request's SWA blocks asynchronously after forward, but the engine runs
  // UpdateRequestTokens for the same step as soon as the result comes back and releases every SWA
  // block that just slid out of the window.  Those are exactly the blocks the upload still has to
  // read, so without this hold another request can win them and the global store ends up with
  // another request's KV under this request's prefix hash — a silent wrong-answer cache entry.
  //
  // The hold only delays the release; it never raises peak SWA usage, because the request already
  // owned these blocks for the whole forward and the window slide happens in one shot right after.
  //
  // No-op when SWA is not enabled.
  virtual void BeginSwaPutHold(int64_t req_id) = 0;

  // Release everything parked by BeginSwaPutHold and stop parking for this request.
  // Must be called once all devices report their put done, and on request teardown so the
  // parked blocks cannot leak.
  virtual void EndSwaPutHold(int64_t req_id) = 0;

  // Update internal state after request finished.
  virtual void DestroyFinishedRequest(int64_t req_id) = 0;

  virtual bool IsPrefixCachingEnabled() = 0;
};

}  // namespace ksana_llm
