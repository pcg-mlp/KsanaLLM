/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "cache_manager/prefix_cache_manager.h"

#include <algorithm>
#include <climits>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_set>

#include "cache_manager/base_cache_manager.h"
#include "runtime/request_state.h"
#include "utils/memory_utils.h"
#include "utils/ret_code.h"
#include "utils/status.h"
#include "utils/string_utils.h"

namespace ksana_llm {

PrefixCacheManager::PrefixCacheManager(const CacheManagerConfig& cache_manager_config,
                                       std::shared_ptr<BlockAllocatorGroupInterface> block_allocator_group,
                                       std::shared_ptr<BlockAllocatorGroupInterface> swa_block_allocator_group)
    : BaseCacheManager<PrefixCachedBlock, PrefixCachedRequest>(cache_manager_config, block_allocator_group) {
  cache_manager_config_ = cache_manager_config;

  root_cached_block_ = new PrefixCachedBlock();
  root_cached_block_->is_root = true;

  // Initialize SWACacheManager only when SWA is configured.
  if (swa_block_allocator_group != nullptr && cache_manager_config_.sliding_window_size > 0) {
    const bool is_contiguous_swa_mode = !cache_manager_config_.enable_non_contiguous_swa_mode;
    swa_cache_manager_ =
        std::make_unique<SWACacheManager>(swa_block_allocator_group, cache_manager_config_.sliding_window_size,
                                          cache_manager_config_.block_token_num, is_contiguous_swa_mode);
  }
}

PrefixCacheManager::~PrefixCacheManager() {
  KLLM_LOG_DEBUG << "PrefixCacheManager destroyed.";
  delete root_cached_block_;
}

void PrefixCacheManager::InitializeCachedBlocks() {
  size_t total_device_block_num = block_allocator_group_->GetDeviceBlockAllocator()->GetFreeBlockNumber();

  // block id 0 is root, so here start with block id 1.
  for (size_t i = 1; i <= total_device_block_num; ++i) {
    PrefixCachedBlock* cached_block = CreateCachedBlock(i);

    // allocate memory block on device0 and reuse its result for all devices.
    std::vector<int> blocks;
    block_allocator_group_->GetDeviceBlockAllocator(0)->AllocateBlocks(1, blocks);
    cached_block->memory_block_id = blocks[0];
    free_cached_blocks_.push(cached_block);
  }
  KLLM_LOG_DEBUG << "PrefixCacheManager initialized, device num:" << block_device_num_
                 << ", device block num:" << free_cached_blocks_.size()
                 << ", host block num:" << block_allocator_group_->GetHostBlockAllocator()->GetFreeBlockNumber();
}

std::shared_ptr<BlockAllocatorGroupInterface> PrefixCacheManager::GetBlockAllocatorGroup() const {
  return BaseCacheManager<PrefixCachedBlock, PrefixCachedRequest>::GetBlockAllocatorGroup();
}

size_t PrefixCacheManager::GetUsableBlockNumber() {
  return free_cached_blocks_.size() + reusable_cached_blocks_.size();
}

size_t PrefixCacheManager::GetRequestUsableBlockNumber(int64_t req_id) {
  auto it = cached_requests_.find(req_id);
  if (it == cached_requests_.end()) {
    KLLM_THROW(FormatStr("Get usable block number for req %d error, req not exist.", req_id));
  }
  PrefixCachedRequest* cached_request = it->second.get();

  size_t reserved_block_num = 0;
  for (PrefixCachedBlock* cached_block : cached_request->cached_blocks) {
    if (reusable_cached_blocks_.find(cached_block) != reusable_cached_blocks_.end()) {
      ++reserved_block_num;
    }
  }

  return free_cached_blocks_.size() + reusable_cached_blocks_.size() - reserved_block_num;
}

size_t PrefixCacheManager::GetHostFreeBlockNumber() {
  return BaseCacheManager<PrefixCachedBlock, PrefixCachedRequest>::GetHostFreeBlockNumber();
}

size_t PrefixCacheManager::GetUsedBlockNumber() {
  return BaseCacheManager<PrefixCachedBlock, PrefixCachedRequest>::GetDeviceUsedBlockNumber() - GetUsableBlockNumber();
}

size_t PrefixCacheManager::GetRequestStepBlockNumber(int64_t req_id, size_t input_token_lens) {
  const auto it = cached_requests_.find(req_id);
  if (it == cached_requests_.end()) {
    KLLM_THROW(FormatStr("Get step block number for req %d error, req not exist.", req_id));
  }
  const size_t block_token_num = cache_manager_config_.block_token_num;
  const size_t total_require_block_num = (input_token_lens + block_token_num) / block_token_num;
  return total_require_block_num - std::min(total_require_block_num, it->second->cached_blocks.size());
}

size_t PrefixCacheManager::GetRequestStepBlockNumberForOneNextToken(int64_t req_id) {
  auto it = cached_requests_.find(req_id);
  if (it == cached_requests_.end()) {
    KLLM_THROW(FormatStr("Get step block number for req %d error, req not exist.", req_id));
  }
  return it->second->cached_blocks.back()->is_shareable ? 1 : 0;
}

Status PrefixCacheManager::GetRequestPrefixBlockNumberInternal(
    int64_t req_id, const std::vector<int>& input_token_ids, size_t check_token_num, bool need_swa_align,
    size_t& fa_shared_block_num, size_t& fa_unique_block_num, size_t& fa_shared_token_num, size_t& swa_shared_block_num,
    size_t& swa_unique_block_num, size_t& swa_shared_token_num) {
  auto it = cached_requests_.find(req_id);
  PrefixCachedRequest* request = (it == cached_requests_.end()) ? CreateCachedRequest(req_id) : it->second.get();
  KLLM_CHECK(input_token_ids.size() >= check_token_num);

  // State guard: both GetRequestPrefixBlockNumber and GetRequestPrefixBlockNumberBeforeRemoteCacheLoading
  // may only be called when the request is in kWaiting or kRemoteLoading state.
  // Calling them in kRunning/kFinished/kSwapped would corrupt the cached_blocks list and
  // the SWA window state that is already in use by the running inference.
  KLLM_CHECK_WITH_INFO(
      request->req_state == RequestState::kWaiting || request->is_remote_loading,
      FormatStr("GetRequestPrefixBlockNumber(Internal): req %d is in state %d, expected kWaiting or kRemoteLoading.",
                req_id, static_cast<int>(request->req_state)));

  // Check whether existed blocks is still valid.
  if (!request->cached_blocks.empty()) {
    for (size_t i = 0; i < request->cached_blocks.size(); ++i) {
      PrefixCachedBlock* cb = request->cached_blocks[i];
      PrefixCachedBlock* cb_prev = (i == 0) ? root_cached_block_ : request->cached_blocks[i - 1];

      // Make sure the block is not changed. Not check token size, just believe it.
      if (cb->parent == cb_prev && cb->is_shareable && cb->is_device_location &&
          CheckSameTokens(cb, input_token_ids.data() + (i * cache_manager_config_.block_token_num),
                          cache_manager_config_.block_token_num)) {
        continue;
      }

      // block changed, remove left blocks and research.
      request->cached_blocks.erase(request->cached_blocks.begin() + i, request->cached_blocks.end());
      break;
    }
  }

  // Research from last block, because new matched block maybe generated yet.
  // When SWA is enabled, we must also verify that SWA cache is valid for all
  // blocks inside the SWA window of the candidate endpoint.
  //
  // The SWA window is dynamic: for a candidate endpoint of (i+1) blocks, the
  // window covers [max(0, (i+1) - sliding_window_blocks), i].  A position is
  // only valid if every block in that window has a valid SWA cache entry.
  //
  // Example: 10 blocks total, sliding_window = 2 blocks, FA matches 0-7,
  //          SWA valid at 4, 5, 7.
  //   candidate last=8: window=[6,7], block 6 missing → invalid
  //   candidate last=7: window=[5,6], block 6 missing → invalid
  //   candidate last=6: window=[4,5], both present  → valid ✓
  //   → last_swa_valid_idx = 6

  const size_t block_token_num = cache_manager_config_.block_token_num;

  // sliding_window_blocks: number of blocks in the SWA window (0 = SWA disabled).
  size_t sliding_window_blocks = 0;
  if (swa_cache_manager_ != nullptr && cache_manager_config_.sliding_window_size > 0) {
    // Round up so that a partial block at the window boundary is included.
    sliding_window_blocks = (cache_manager_config_.sliding_window_size + block_token_num - 1) / block_token_num;
  }

  // Phase 1: pure FA-tree scan — find the furthest valid position.
  // No SWA side-effects here; we only determine last_swa_valid_idx.
  //
  // We use a consecutive-streak counter: swa_valid_in_window counts how many
  // SWA-valid blocks appear in an unbroken run ending at the current position i.
  // When a block without a valid SWA entry is encountered the streak resets to 0,
  // because the SWA window requires every block inside it to be valid.
  //
  // Endpoint (i+1) is fully valid iff the streak has reached at least W
  // (sliding_window_blocks), meaning the last W blocks all have SWA cache.
  //
  // last_swa_valid_idx: number of blocks that are fully valid (FA hit + all SWA
  // blocks in the window valid).
  size_t last_swa_valid_idx = 0;

  // swa_valid_in_window: length of the current consecutive run of SWA-valid
  // blocks ending at position (i).  Resets to 0 whenever a block without a
  // valid SWA entry is seen.
  size_t swa_valid_in_window = 0;

  // Pre-populate the streak counter for already-matched blocks (SWA enabled only).
  // Walk backwards from the last matched block and count the unbroken tail of
  // SWA-valid blocks; that tail length is the initial streak value.
  // Also determine whether the already-matched endpoint itself is valid so that
  // last_swa_valid_idx can be initialised correctly.
  if (sliding_window_blocks > 0 && !request->cached_blocks.empty()) {
    const size_t already = request->cached_blocks.size();
    // Count the consecutive SWA-valid tail of the already-matched blocks.
    for (size_t k = already; k > 0; --k) {
      if (swa_cache_manager_->IsSWABlockValid(request->cached_blocks[k - 1]->block_id)) {
        ++swa_valid_in_window;
      } else {
        break;  // Streak broken; stop counting.
      }
    }
    // The already-matched endpoint is valid iff the streak covers the full window.
    if (swa_valid_in_window >= sliding_window_blocks) {
      last_swa_valid_idx = already;
    }
  }

  PrefixCachedBlock* cur_block = request->cached_blocks.empty() ? root_cached_block_ : request->cached_blocks.back();
  for (size_t i = request->cached_blocks.size(); i < check_token_num / block_token_num; ++i) {
    cur_block = FindChildCacheBlock(cur_block, input_token_ids.data() + (i * block_token_num), block_token_num);
    if (cur_block == nullptr) {
      break;
    }

    // Accumulate matched blocks (no request-state changes yet).
    request->cached_blocks.push_back(cur_block);

    if (sliding_window_blocks == 0) {
      // SWA disabled: every FA hit is unconditionally valid; no truncation needed.
      continue;
    }

    // Update the consecutive-streak counter:
    //   - If the new block has a valid SWA entry, extend the streak.
    //   - Otherwise the streak is broken; reset to 0.
    if (swa_cache_manager_->IsSWABlockValid(cur_block->block_id)) {
      ++swa_valid_in_window;
    } else {
      swa_valid_in_window = 0;
    }

    // Endpoint (i+1) is valid iff the last W blocks all have SWA cache,
    // i.e., the streak length has reached at least sliding_window_blocks.
    if (swa_valid_in_window >= sliding_window_blocks) {
      last_swa_valid_idx = i + 1;
    }
    // If the streak is shorter than W, continue scanning for a further valid
    // position — do not stop early.
  }

  // last_fa_valid_idx: the full FA-matched count before any SWA truncation.
  const size_t last_fa_valid_idx = request->cached_blocks.size();

  if (need_swa_align) {
    // Truncate cached_blocks to the furthest fully-valid SWA position.
    // When SWA is disabled the full FA-matched set is kept unchanged.
    if (sliding_window_blocks > 0) {
      request->cached_blocks.resize(last_swa_valid_idx);
    }
  } else {
    // need_swa_align=false: FA cache is NOT truncated to SWA alignment.
    // cached_blocks retains the full FA-matched set.
    // fa_prefix_block_num records the full FA-matched count for AllocateBlocksForRemoteCache.
    request->fa_prefix_block_num = last_fa_valid_idx;
    request->is_remote_loading = true;
  }

  // Phase 2: Populate swa_window_fa_block_ids with the FA block ids in the SWA window
  // and increment hit_count for each window block that is a prefix-cache hit.
  //
  // IncrementHitCount is called here (at query time) so that the LFU eviction order
  // reflects every prefix-cache hit regardless of whether AllocateRequestBlocks is
  // subsequently called.  FA block registration (active_requests) and SWA holder
  // registration (AcquireSWABlock) are still deferred to AllocateRequestBlocks /
  // AllocateBlocksForRemoteCache (§4.0.3) because they require the availability
  // pre-check to have passed first.
  //
  // By pre-computing swa_window_fa_block_ids here, the Alloc functions can simply iterate
  // the list and call AcquireSWABlock without any re-scan.
  //
  // On re-entry (§4.0.1), cached_blocks may have been extended by a new FA-tree match.
  // We always recompute the window from scratch to reflect the latest state.
  if (swa_cache_manager_ != nullptr && sliding_window_blocks > 0) {
    // The effective SWA window endpoint is last_swa_valid_idx (already computed above).
    // For need_swa_align=true, cached_blocks was truncated to last_swa_valid_idx, so
    // cached_blocks.size() == last_swa_valid_idx.
    // For need_swa_align=false, cached_blocks.size() == last_fa_valid_idx >= last_swa_valid_idx.
    request->swa_window_fa_block_ids.clear();
    request->swa_window_block_idxs.clear();
    if (last_swa_valid_idx > 0) {
      // When full cache hit occurs, HandleFullPrefixCacheHit may rollback one block for
      // extend recomputation, extend the window by 1 to cover it.
      const bool is_full_hit = (last_swa_valid_idx * block_token_num >= check_token_num);
      const size_t window_cover = sliding_window_blocks + (is_full_hit ? 1 : 0);
      const size_t win_start = (last_swa_valid_idx >= window_cover) ? last_swa_valid_idx - window_cover : 0;
      for (size_t i = win_start; i < last_swa_valid_idx; ++i) {
        const size_t fa_id = request->cached_blocks[i]->block_id;
        request->swa_window_fa_block_ids.push_back(fa_id);
        request->swa_window_block_idxs.push_back(i);
        // Increment hit_count here (at query time) so that the LFU eviction order
        // reflects every prefix-cache hit, regardless of whether AllocateRequestBlocks
        // is subsequently called.  AllocateRequestBlocksInternal will call AcquireSWABlock
        // (not ReuseSWABlock) to register the holder without double-counting.
        swa_cache_manager_->IncrementHitCount(fa_id);
      }
    }
  }

  // When SWA is disabled, last_swa_valid_idx is never updated; use the actual
  // number of matched blocks from the FA tree instead.
  swa_shared_block_num = (sliding_window_blocks == 0) ? last_fa_valid_idx : last_swa_valid_idx;
  fa_shared_block_num = (need_swa_align || sliding_window_blocks == 0) ? swa_shared_block_num : last_fa_valid_idx;

  const size_t total_blocks_needed = (check_token_num + block_token_num - 1) / block_token_num;
  fa_unique_block_num = total_blocks_needed - std::min(total_blocks_needed, fa_shared_block_num);
  swa_unique_block_num = total_blocks_needed - std::min(total_blocks_needed, swa_shared_block_num);
  fa_shared_token_num = fa_shared_block_num * block_token_num;
  swa_shared_token_num = swa_shared_block_num * block_token_num;
  request->shared_block_num = swa_shared_block_num;

  return Status();
}

Status PrefixCacheManager::GetRequestPrefixBlockNumber(int64_t req_id, const std::vector<int>& input_token_ids,
                                                       size_t check_token_num, size_t& shared_block_num,
                                                       size_t& unique_block_num, size_t& shared_token_num) {
  // need_swa_align=true: fa_shared_block_num == swa_shared_block_num after the call.
  // Expose FA-side values as the canonical shared/unique/token counts.
  size_t swa_shared_block_num = 0;
  size_t swa_unique_block_num = 0;
  size_t swa_shared_token_num = 0;
  return GetRequestPrefixBlockNumberInternal(req_id, input_token_ids, check_token_num,
                                             /*need_swa_align=*/true, shared_block_num, unique_block_num,
                                             shared_token_num, swa_shared_block_num, swa_unique_block_num,
                                             swa_shared_token_num);
}

Status PrefixCacheManager::GetRequestPrefixBlockNumberBeforeRemoteCacheLoading(
    int64_t req_id, const std::vector<int>& input_token_ids, size_t check_token_num, size_t& fa_shared_block_num,
    size_t& fa_unique_block_num, size_t& fa_shared_token_num, size_t& swa_shared_block_num,
    size_t& swa_unique_block_num, size_t& swa_shared_token_num) {
  return GetRequestPrefixBlockNumberInternal(req_id, input_token_ids, check_token_num,
                                             /*need_swa_align=*/false, fa_shared_block_num, fa_unique_block_num,
                                             fa_shared_token_num, swa_shared_block_num, swa_unique_block_num,
                                             swa_shared_token_num);
}

bool PrefixCacheManager::CheckSameTokens(const PrefixCachedBlock* block, const int* start, size_t len) {
  return std::memcmp(block->token_ids.data(), start, len * sizeof(int)) == 0;
}

PrefixCachedBlock* PrefixCacheManager::FindChildCacheBlock(PrefixCachedBlock* block, const int* start, size_t len) {
  std::vector<int> token_ids(start, start + len);

  auto it = block->children.find(token_ids);
  if (it == block->children.end()) {
    return nullptr;
  }

  return it->second;
}

PrefixCachedBlock* PrefixCacheManager::CreateEmptyCachedBlock() {
  PrefixCachedBlock* cached_block = new PrefixCachedBlock();
  cached_block->token_ids.resize(cache_manager_config_.block_token_num);

  cached_block->is_shareable = false;
  cached_block->is_device_location = true;

  return cached_block;
}

PrefixCachedBlock* PrefixCacheManager::CreateCachedBlock(size_t block_id) {
  PrefixCachedBlock* cached_block = CreateEmptyCachedBlock();
  cached_block->block_id = block_id;
  return cached_block;
}

void PrefixCacheManager::ResetCachedBlock(PrefixCachedBlock* cached_block) {
  // Keep block id unchanged, reset hashcode.

  cached_block->is_shareable = false;
  cached_block->is_device_location = true;
  EraseFlexibleCacheMap(cached_block);
  cached_block->token_ids.clear();
  cached_block->token_ids.resize(cache_manager_config_.block_token_num);

  cached_block->active_requests.clear();
  cached_block->inactive_requests.clear();

  cached_block->parent = nullptr;
  cached_block->children.clear();
}

/**
 * @brief Find the longest matching prefix block among children of the last block.
 */
void PrefixCacheManager::QueryIncompleteBlock(PrefixCachedBlock* last_block, const std::vector<int>& token_ids,
                                              int start_idx, PrefixCachedBlock*& prefix_append_block,
                                              size_t& prefix_append_len) {
  for (auto& try_block : last_block->children) {
    if (!try_block.second->inactive_requests.empty()) {
      continue;
    }
    size_t i = 0;
    for (; i < cache_manager_config_.block_token_num; i++) {
      if (try_block.second->token_ids[i] != token_ids[start_idx + i]) {
        break;
      }
    }
    if (prefix_append_len < i) {
      prefix_append_len = i;
      prefix_append_block = try_block.second;
    }
  }
}

int PrefixCacheManager::GetPrefixLen(PrefixCachedBlock* block) {
  // Remove root node
  int len = -1 * cache_manager_config_.block_token_num;
  if (block == nullptr) {
    return 0;
  }
  while (block->parent) {
    len += cache_manager_config_.block_token_num;
    block = block->parent;
  }
  return len;
}

/**
 * @brief Updates the flexible cache for a given request by finding and copying matching data blocks.
 *
 * This function searches for the longest prefix match and extends it to form a longer matching sequence
 * within the provided token IDs. It then copies the matched data blocks into the flexible cache task list
 * for subsequent processing.
 *
 * @param req_id The request id
 * @param token_ids The input tokens of request
 * @param shared_token_num The prefix cache length of request
 * @param flexible_cached_copy_tasks Request member. The information of queried flexible cache tokens will be stored
 * here
 * @param req_flexible_cache_len Request member. The total num of queried flexible cache tokens will be stored here
 */
void PrefixCacheManager::UpdateFlexibleCache(int64_t req_id, const std::vector<int>& token_ids, int shared_token_num,
                                             std::vector<FlexibleCachedCopyTask>& flexible_cached_copy_tasks,
                                             size_t& req_flexible_cache_len) {
  // Check if the minimum flexible cache size is set and greater than 0
  size_t min_flexible_cache_num = cache_manager_config_.min_flexible_cache_num;
  if (min_flexible_cache_num == 0) {
    return;
  }

  size_t flexible_cache_len = 0;
  size_t flexible_cache_begin = 0;
  auto& block_token_num = cache_manager_config_.block_token_num;
  // Check if there are enough tokens for flexible caching
  if (shared_token_num + min_flexible_cache_num + block_token_num * 2 > token_ids.size() || shared_token_num == 0) {
    return;
  }
  auto it = cached_requests_.find(req_id);
  if (it == cached_requests_.end()) {
    KLLM_THROW(FormatStr("Get usable block number for req %d error, req not exist.", req_id));
  }
  PrefixCachedRequest* cached_request = it->second.get();

  // Find the longest matching prefix tokens among children of the last block
  std::vector<PrefixCachedBlock*> cached_blocks;
  size_t prefix_append_len = 0;
  PrefixCachedBlock* prefix_append_block = nullptr;
  if (shared_token_num > 0) {
    PrefixCachedBlock* last_block = cached_request->cached_blocks[shared_token_num / block_token_num - 1];
    QueryIncompleteBlock(last_block, token_ids, shared_token_num, prefix_append_block, prefix_append_len);
  }

  // Update shared_token_num with the length of the prefix append block
  auto new_shared_token_num = shared_token_num + prefix_append_len;

  // Iterate through the token_ids to find the best matching block for flexible caching
  for (uint32_t offset = new_shared_token_num; offset < new_shared_token_num + block_token_num; offset++) {
    if (min_flexible_cache_num + offset > token_ids.size()) {
      break;
    }
    PrefixCachedBlock* cached_block = nullptr;
    std::vector<int> dst_token_ids(token_ids.begin() + offset, token_ids.begin() + offset + block_token_num);
    auto map_it = flexible_cache_map_.find(dst_token_ids);
    if (map_it != flexible_cache_map_.end()) {
      for (auto block : map_it->second) {
        size_t hit_num = block_token_num;
        uint32_t start_idx = block_token_num - (offset - new_shared_token_num);
        cached_block = block;
        if (!cached_block->inactive_requests.empty()) {
          continue;
        }
        std::vector<PrefixCachedBlock*> new_cached_blocks;
        if (offset != new_shared_token_num) {
          if (cached_block->parent == nullptr) {
            continue;
          }
          if (!cached_block->parent->inactive_requests.empty()) {
            continue;
          }
          // skip if the parent is root block
          if (cached_block->parent->token_ids.empty()) {
            continue;
          }
          for (; start_idx < block_token_num; start_idx++) {
            if (cached_block->parent->token_ids[start_idx] != token_ids[offset - block_token_num + start_idx]) {
              break;
            }
          }
          if (start_idx != block_token_num) {
            continue;
          }
          new_cached_blocks.push_back(cached_block->parent);
        }
        new_cached_blocks.push_back(cached_block);
        bool partial_match = false;
        for (; offset + hit_num + block_token_num < token_ids.size(); hit_num += block_token_num) {
          std::memcpy(dst_token_ids.data(), token_ids.data() + offset + hit_num, block_token_num * sizeof(int));
          auto it = cached_block->children.find(dst_token_ids);
          if (it != cached_block->children.end()) {
            cached_block = it->second;
            if (!cached_block->inactive_requests.empty()) {
              continue;
            }
            new_cached_blocks.push_back(cached_block);
          } else {
            partial_match = true;
            break;
          }
        }
        if (partial_match) {
          PrefixCachedBlock* flexible_append_block = nullptr;
          size_t flexible_append_len = 0;
          QueryIncompleteBlock(cached_block, dst_token_ids, 0, flexible_append_block, flexible_append_len);
          if (flexible_append_len > 0) {
            hit_num += flexible_append_len;
            new_cached_blocks.push_back(flexible_append_block);
          }
        }

        if (hit_num >= min_flexible_cache_num && hit_num > flexible_cache_len) {
          cached_blocks = new_cached_blocks;
          flexible_cache_begin = (block_token_num - (offset - new_shared_token_num)) % block_token_num;
          flexible_cache_len = offset - new_shared_token_num + hit_num;
        }
      }
    }
  }

  // If no matching blocks are found, return early
  if (cached_blocks.empty()) {
    return;
  }

  // Calculate the number of flexible cached tokens to resize the tasks vector
  size_t new_flexible_cached_tokens = flexible_cache_len + prefix_append_len;
  flexible_cached_copy_tasks.resize(new_flexible_cached_tokens);
  size_t idx = 0;
  int src_shared_token_num = GetPrefixLen(prefix_append_block);
  // Copy the prefix append block tokens to the flexible cache tasks
  for (; idx < prefix_append_len; idx++) {
    flexible_cached_copy_tasks[idx].Update(
        shared_token_num + idx, src_shared_token_num + idx,
        cached_request->cached_blocks[shared_token_num / block_token_num]->memory_block_id,
        prefix_append_block->memory_block_id);
  }
  src_shared_token_num = GetPrefixLen(cached_blocks[0]);
  // Copy the remaining cached block tokens to the flexible cache tasks
  for (; idx < new_flexible_cached_tokens; idx++) {
    auto cached_token_idx = (idx - prefix_append_len + flexible_cache_begin);
    auto cached_blocks_idx = cached_token_idx / block_token_num;
    flexible_cached_copy_tasks[idx].Update(
        shared_token_num + idx, src_shared_token_num + cached_token_idx,
        cached_request->cached_blocks[(shared_token_num + idx) / block_token_num]->memory_block_id,
        cached_blocks[cached_blocks_idx]->memory_block_id);
  }
  req_flexible_cache_len = flexible_cached_copy_tasks.size();
}

void PrefixCacheManager::EraseFlexibleCacheMap(PrefixCachedBlock* cached_block) {
  auto map_it = flexible_cache_map_.find(cached_block->token_ids);
  if (map_it != flexible_cache_map_.end()) {
    auto block_set_it = map_it->second.find(cached_block);
    if (block_set_it != map_it->second.end()) {
      map_it->second.erase(block_set_it);
    }
    if (map_it->second.empty()) {
      flexible_cache_map_.erase(map_it);
    }
  }
}

void PrefixCacheManager::InsertFlexibleCacheMap(PrefixCachedBlock* cached_block) {
  flexible_cache_map_[cached_block->token_ids].insert(cached_block);
}

void PrefixCacheManager::FreeCachedBlockRecursively(PrefixCachedBlock* cached_block,
                                                    std::vector<PrefixCachedBlock*>& free_blocks, size_t& free_num) {
  if (cached_block->children.empty()) {
    // From reusable to free
    reusable_cached_blocks_.erase(cached_block);

    // Release the associated SWA block when the FA block is permanently freed.
    if (swa_cache_manager_ != nullptr) {
      swa_cache_manager_->FreeSWABlock(cached_block->block_id);
    }

    ResetCachedBlock(cached_block);
    free_cached_blocks_.push(cached_block);
    free_blocks.push_back(cached_block);
    free_num += 1;
    return;
  }

  // First process all children.
  for (auto pair : cached_block->children) {
    FreeCachedBlockRecursively(pair.second, free_blocks, free_num);
  }

  // Force reuse itself, from reusable to free
  reusable_cached_blocks_.erase(cached_block);

  // Release the associated SWA block when the FA block is permanently freed.
  if (swa_cache_manager_ != nullptr) {
    swa_cache_manager_->FreeSWABlock(cached_block->block_id);
  }

  ResetCachedBlock(cached_block);
  free_cached_blocks_.push(cached_block);
  free_blocks.push_back(cached_block);
  free_num += 1;
}

bool PrefixCacheManager::FreeCachedBlocks(size_t block_num, size_t& free_block_num,
                                          const std::vector<PrefixCachedBlock*>& reserved_blocks) {
  free_block_num = 0;
  std::vector<PrefixCachedBlock*> free_blocks;

  // Loop timeline from end to begin.
  for (auto it = timed_cached_blocks_.rbegin(); it != timed_cached_blocks_.rend(); ++it) {
    PrefixCachedBlock* cb = *it;

    // Skip reserved blocks.
    if (!reserved_blocks.empty()) {
      if (std::find(reserved_blocks.begin(), reserved_blocks.end(), cb) != reserved_blocks.end()) {
        continue;
      }
    }

    // Skip if unfilled, unexpected, all block in timeline should be filled.
    if (!cb->is_shareable) {
      KLLM_LOG_DEBUG << "FreeCachedBlocks should not arrive here, unfilled block in timeline.";
      continue;
    }

    if (cb->active_requests.empty() && cb->inactive_requests.empty()) {
      // Take it and all its children from tree.
      auto it2 = cb->parent->children.find(cb->token_ids);
      if (it2 != cb->parent->children.end()) {
        cb->parent->children.erase(it2);
      }
      FreeCachedBlockRecursively(cb, free_blocks, free_block_num);
    }

    if (free_block_num >= block_num) {
      break;
    }
  }

  // Remove from timed block list here, because could not erase in itself's loop.
  for (PrefixCachedBlock* cb : free_blocks) {
    RemoveCachedBlockFromTimedList(cb);
  }

  return free_block_num >= block_num;
}

Status PrefixCacheManager::AllocateRequestBlocksInternal(PrefixCachedRequest* cached_request, int64_t req_id,
                                                         std::vector<int>& req_block_ids, size_t fa_block_num,
                                                         size_t minimum_swa_block_num,
                                                         bool disable_swa_block_num_adjust,
                                                         std::vector<int>& new_fa_block_ids,
                                                         std::vector<int>& new_swa_fa_offsets) {
  KLLM_CHECK_WITH_INFO(fa_block_num >= minimum_swa_block_num,
                       FormatStr("FA block num (%zu) should be no less than minimum SWA block num (%zu).", fa_block_num,
                                 minimum_swa_block_num));

  // ── SWA resource check BEFORE any allocation (pure query, no state change) ──
  // Query which new FA block positions should receive SWA blocks.
  //   swa_window_fa_block_ids is passed as reserved_fa_block_ids so that reclaimable
  //   SWA blocks in the current window are excluded from the available count. Without
  //   this, GetAvailableBlockCount() would count them as free capacity, but AcquireSWABlock
  //   (called below after this check) removes them from the reclaimable pool, leaving fewer
  //   blocks than expected for the new allocations.
  //   This is a pure query: if it returns -1 nothing has been allocated → clean OOM return.
  new_swa_fa_offsets.clear();
  if (swa_cache_manager_ != nullptr && minimum_swa_block_num > 0) {
    const size_t cur_fa_block_num = cached_request->cached_blocks.size();
    const size_t new_fa_block_num = cur_fa_block_num + fa_block_num;
    if (swa_cache_manager_->GetAllocatableSWACache(req_id, cur_fa_block_num, new_fa_block_num, minimum_swa_block_num,
                                                   disable_swa_block_num_adjust, new_swa_fa_offsets,
                                                   cached_request->swa_window_fa_block_ids) < 0) {
      return Status(RET_OUT_OF_DEVICE_MEMORY,
                    FormatStr("AllocateRequestBlocksInternal: GetAllocatableSWACache failed for req %d.", req_id));
    }
  }
  // ── End SWA resource check ───────────────────────────────────────────────────

  // Ensure enough FA blocks are available; evict unreferenced cached blocks if needed.
  size_t needed_blocks = 0;
  if (fa_block_num > free_cached_blocks_.size()) {
    needed_blocks = fa_block_num - free_cached_blocks_.size();
  }

  const std::vector<PrefixCachedBlock*> reserved_blocks =
      (cached_request->req_state == RequestState::kWaiting || cached_request->req_state == RequestState::kRunning)
          ? cached_request->cached_blocks
          : std::vector<PrefixCachedBlock*>();

  size_t free_block_num = 0;
  if (needed_blocks > 0 && !FreeCachedBlocks(needed_blocks, free_block_num, reserved_blocks)) {
    return Status(RET_OUT_OF_DEVICE_MEMORY, FormatStr("AllocateRequestBlocksInternal: not enough FA blocks "
                                                      "(needed=%zu, free=%zu).",
                                                      fa_block_num, free_cached_blocks_.size()));
  }

  // Both SWA and FA availability confirmed above.  Now register SWA blocks for the cached
  // FA blocks pre-computed in the query phase (§4.0.3: SWA registration must follow FA
  // availability confirmation).  IncrementHitCount was already called in Phase 2; here we
  // only call AcquireSWABlock to register req_id as an active holder.
  if (swa_cache_manager_ != nullptr) {
    for (size_t fa_id : cached_request->swa_window_fa_block_ids) {
      swa_cache_manager_->AcquireSWABlock(req_id, fa_id);
    }
  }

  // I donot want to check req block size here, because caller should guard this.
  const size_t req_block_num = req_block_ids.size();
  // G3: Do not transition to kRunning when the request is in remote-loading state.
  // AllocateBlocksForRemoteCache will set is_remote_loading_block_allocated=true, and
  // ConfirmRemoteCacheLoadingResult will perform the correct kWaiting→kRunning transition.
  if (req_block_num == 0 && !cached_request->is_remote_loading) {
    cached_request->req_state = RequestState::kRunning;
  }

  // The prefix cache may be invalid. Refer to cached_request for the authoritative data and clean up the invalid parts.
  req_block_ids.resize(std::min(req_block_ids.size(), cached_request->cached_blocks.size()));

  for (size_t i = req_block_num; i < cached_request->cached_blocks.size(); ++i) {
    PrefixCachedBlock* cb = cached_request->cached_blocks[i];

    // Fill prefix memory blocks.
    req_block_ids.push_back(cb->memory_block_id);

    // Not reusable if no request associated with it.
    if (cb->active_requests.empty() && cb->inactive_requests.empty()) {
      reusable_cached_blocks_.erase(cb);
    }

    // Associate request to prefill cached block.
    cb->active_requests[req_id] = std::make_pair(i, cached_request);
  }

  // Build a fast lookup set from the positions returned by GetAllocatableSWACache.
  const std::unordered_set<int> swa_positions(new_swa_fa_offsets.begin(), new_swa_fa_offsets.end());

  new_fa_block_ids.clear();
  new_fa_block_ids.reserve(fa_block_num);

  // Allocate fa_block_num FA blocks; bind SWA blocks only at positions in swa_positions.
  for (size_t i = 0; i < fa_block_num; ++i) {
    PrefixCachedBlock* cached_block = free_cached_blocks_.front();
    free_cached_blocks_.pop();

    const size_t global_idx = cached_request->cached_blocks.size();
    new_fa_block_ids.push_back(cached_block->memory_block_id);
    cached_request->cached_blocks.push_back(cached_block);

    if (swa_cache_manager_ != nullptr && swa_positions.count(static_cast<int>(global_idx))) {
      const size_t fa_block_id = cached_block->block_id;
      int swa_mem_id = swa_cache_manager_->AllocateSWABlock(cached_request->req_id, fa_block_id, global_idx);
      if (swa_mem_id < 0) {
        // Unexpected: GetAllocatableSWACache already confirmed availability.
        // Roll back all FA blocks allocated so far (including this one).
        for (size_t j = 0; j <= i; ++j) {
          PrefixCachedBlock* cb = cached_request->cached_blocks.back();
          cached_request->cached_blocks.pop_back();
          ResetCachedBlock(cb);
          free_cached_blocks_.push(cb);
        }
        new_fa_block_ids.clear();
        new_swa_fa_offsets.clear();
        return Status(RET_OUT_OF_DEVICE_MEMORY,
                      FormatStr("AllocateRequestBlocksInternal: unexpected AllocateSWABlock failure "
                                "at global_idx=%zu for req %d.",
                                global_idx, req_id));
      }
    }
  }

  return Status();
}

Status PrefixCacheManager::AllocateRequestBlocks(int64_t req_id, size_t block_num, std::vector<int>& req_block_ids,
                                                 std::vector<int>& swa_req_block_ids,
                                                 std::vector<size_t>& swa_block_idx_offsets) {
  auto it = cached_requests_.find(req_id);
  if (it == cached_requests_.end()) {
    return Status(RET_RUNTIME_FAILED, FormatStr("Allocate block for req %d error, req not exist.", req_id));
  }
  PrefixCachedRequest* cached_request = it->second.get();

  // Compute minimum_swa_block_num the same way as DirectCacheManager:
  // min(new FA blocks beyond current SWA right-boundary, one full sliding window).
  // This ensures we only require the tail-coverage SWA blocks, not one per new FA block.
  const size_t sliding_window_blocks =
      (cache_manager_config_.sliding_window_size > 0 && cache_manager_config_.block_token_num > 0)
          ? (cache_manager_config_.sliding_window_size + cache_manager_config_.block_token_num - 1) /
                    cache_manager_config_.block_token_num +
                1
          : 0;
  const size_t cur_swa_end =
      cached_request->swa_window_block_idxs.empty() ? 0 : cached_request->swa_window_block_idxs.back() + 1;
  const size_t new_fa_total = cached_request->cached_blocks.size() + block_num;
  const size_t minimum_swa_block_num = (swa_cache_manager_ != nullptr && sliding_window_blocks > 0)
                                           ? std::min(new_fa_total - cur_swa_end, sliding_window_blocks)
                                           : 0;

  // Allocate new FA blocks; SWA binding is determined by GetAllocatableSWACache inside.
  std::vector<int> new_fa_block_ids;
  std::vector<int> new_swa_fa_offsets;
  Status status =
      AllocateRequestBlocksInternal(cached_request, req_id, req_block_ids, block_num, minimum_swa_block_num,
                                    /*disable_swa_block_num_adjust*/ false, new_fa_block_ids, new_swa_fa_offsets);
  if (!status.OK()) {
    return Status(status.GetCode(),
                  FormatStr("Allocate %d blocks for req %d error: %s", block_num, req_id, status.GetMessage()));
  }

  // Append new FA block ids to the output list.
  for (int mem_id : new_fa_block_ids) {
    req_block_ids.push_back(mem_id);
  }

  // Append only the FA blocks that actually received SWA blocks to the window tracking arrays.
  // new_swa_fa_offsets contains the global FA block indices (ascending) chosen by
  // GetAllocatableSWACache inside AllocateRequestBlocksInternal.
  if (swa_cache_manager_ != nullptr) {
    for (int pos : new_swa_fa_offsets) {
      const size_t global_idx = static_cast<size_t>(pos);
      const size_t fa_block_id = cached_request->cached_blocks[global_idx]->block_id;
      cached_request->swa_window_fa_block_ids.push_back(fa_block_id);
      cached_request->swa_window_block_idxs.push_back(global_idx);
    }
  }

  // Fill SWA output parameters directly from the per-block index arrays.
  // swa_block_idx_offsets[i] is the FA block global index for swa_req_block_ids[i].
  if (swa_cache_manager_ != nullptr) {
    swa_req_block_ids.clear();
    swa_block_idx_offsets.clear();
    for (size_t i = 0; i < cached_request->swa_window_fa_block_ids.size(); ++i) {
      int mem_id = swa_cache_manager_->GetSWAMemoryBlockId(cached_request->swa_window_fa_block_ids[i]);
      if (mem_id >= 0) {
        swa_req_block_ids.push_back(mem_id);
        swa_block_idx_offsets.push_back(cached_request->swa_window_block_idxs[i]);
      }
    }
  }
  KLLM_LOG_SCHEDULER << "AllocateRequestBlocks, req_id=" << req_id << ", block_num=" << block_num
                     << ", req_block_num=" << cached_request->cached_blocks.size()
                     << ", swa_block_idx_offsets.size=" << swa_block_idx_offsets.size() << ", "
                     << Vector2Str(swa_block_idx_offsets);

  return Status();
}

Status PrefixCacheManager::AllocateBlocksForRemoteCache(int64_t req_id, std::vector<int>& req_block_ids,
                                                        size_t fa_block_num, size_t swa_block_num,
                                                        std::vector<int>& remote_fa_block_ids,
                                                        std::vector<int>& remote_swa_block_ids) {
  auto it = cached_requests_.find(req_id);
  if (it == cached_requests_.end()) {
    return Status(RET_RUNTIME_FAILED, FormatStr("AllocateBlocksForRemoteCache: req %d not found.", req_id));
  }
  PrefixCachedRequest* cached_request = it->second.get();

  if (!cached_request->is_remote_loading) {
    return Status(RET_RUNTIME_FAILED,
                  FormatStr("AllocateBlocksForRemoteCache: req %d is not in remote_loading state.", req_id));
  }

  // Allocate fa_block_num FA blocks; SWA blocks are bound at positions chosen by
  // GetAllocatableSWACache (called inside AllocateRequestBlocksInternal).
  // AllocateRequestBlocksInternal also registers req_id into active_requests for all
  // cached_blocks entries not yet registered (the previously-matched cached FA blocks).
  std::vector<int> new_swa_fa_offsets;
  Status status =
      AllocateRequestBlocksInternal(cached_request, req_id, req_block_ids, fa_block_num, swa_block_num,
                                    /*disable_swa_block_num_adjust*/ true, remote_fa_block_ids, new_swa_fa_offsets);
  if (!status.OK()) {
    return Status(status.GetCode(), FormatStr("AllocateBlocksForRemoteCache req %d: %s", req_id, status.GetMessage()));
  }

  // AcquireSWABlock for the cached FA blocks (swa_window_fa_block_ids) was already called
  // inside AllocateRequestBlocksInternal after the availability checks (§4.0.3).
  // IncrementHitCount was called earlier in GetRequestPrefixBlockNumberInternal (Phase 2).

  // Record the FA block ids that actually received SWA blocks, using the positions returned
  // by GetAllocatableSWACache inside AllocateRequestBlocksInternal.
  cached_request->remote_cache_swa_fa_block_ids.clear();
  if (swa_cache_manager_ != nullptr) {
    for (int pos : new_swa_fa_offsets) {
      cached_request->remote_cache_swa_fa_block_ids.push_back(
          cached_request->cached_blocks[static_cast<size_t>(pos)]->block_id);
    }
  }

  // Fill remote_swa_block_ids with SWA memory block IDs for the caller.
  remote_swa_block_ids.clear();
  if (swa_cache_manager_ != nullptr) {
    for (size_t fa_id : cached_request->remote_cache_swa_fa_block_ids) {
      int mem_id = swa_cache_manager_->GetSWAMemoryBlockId(fa_id);
      if (mem_id >= 0) {
        remote_swa_block_ids.push_back(mem_id);
      }
    }
  }

  // G2: Mark that FA and SWA blocks have been allocated and registered to req_id.
  // This transitions the request from kRemoteLoading to kRemoteLoadingBlockAllocated.
  cached_request->is_remote_loading_block_allocated = true;

  return Status();
}

// Release FA blocks (and their bound SWA blocks) in cached_request->cached_blocks
// starting from trunc_start_offset up to the end of the list, then resize the list
// to trunc_start_offset.
//
// For each released FA block:
//   - If the block is not shareable: FreeSWABlock is called (if swa_cache_manager_ != nullptr),
//     then the block is reset and returned to free_cached_blocks_.
//   - If the block is shareable: the request is removed from active_requests /
//     inactive_requests according to cached_request->req_state; the block is inserted into
//     reusable_cached_blocks_ when no references remain.
//
// Does NOT touch swa_window_fa_block_ids or remote_cache_swa_fa_block_ids; callers are
// responsible for managing those lists before calling this method.
void PrefixCacheManager::TruncateRequestBlocks(PrefixCachedRequest* cached_request, size_t trunc_start_offset) {
  for (size_t i = trunc_start_offset; i < cached_request->cached_blocks.size(); ++i) {
    PrefixCachedBlock* cb = cached_request->cached_blocks[i];
    if (!cb->is_shareable) {
      // Non-shareable block: free its SWA block and return the FA block to the free pool.
      if (swa_cache_manager_ != nullptr) {
        swa_cache_manager_->FreeSWABlock(cb->block_id);
      }
      ResetCachedBlock(cb);
      free_cached_blocks_.push(cb);
      continue;
    }

    // Shareable block: remove this request's reference.
    auto& request =
        (cached_request->req_state == RequestState::kRunning || cached_request->req_state == RequestState::kFinished)
            ? cb->active_requests
            : cb->inactive_requests;
    request.erase(cached_request->req_id);

    // Make reusable if no request left.
    if (cb->active_requests.empty() && cb->inactive_requests.empty()) {
      reusable_cached_blocks_.insert(cb);
    }
  }
  cached_request->cached_blocks.resize(trunc_start_offset);
}

Status PrefixCacheManager::ConfirmRemoteCacheLoadingResult(int64_t req_id, const std::vector<int>& kvcached_token_ids,
                                                           size_t shareable_kvcache_token_num,
                                                           const std::vector<size_t>& matched_remote_swa_block_indexs) {
  auto it = cached_requests_.find(req_id);
  if (it == cached_requests_.end()) {
    return Status(RET_RUNTIME_FAILED, FormatStr("ConfirmRemoteCacheLoadingResult: req %d not found.", req_id));
  }
  PrefixCachedRequest* cached_request = it->second.get();

  if (!cached_request->is_remote_loading) {
    return Status(RET_RUNTIME_FAILED,
                  FormatStr("ConfirmRemoteCacheLoadingResult: req %d is not in remote_loading state.", req_id));
  }

  const size_t block_token_num = cache_manager_config_.block_token_num;
  const size_t shareable_block_num = shareable_kvcache_token_num / block_token_num;
  const size_t fa_prefix_block_num = cached_request->fa_prefix_block_num;

  // Pre-check: shareable_kvcache_token_num must not reduce the request's existing coverage.
  if (swa_cache_manager_ != nullptr) {
    // SWA enabled: must cover at least the rightmost token position in swa_window_fa_block_ids.
    const size_t swa_right_token_pos =
        (cached_request->swa_window_block_idxs.empty() ? 0 : (cached_request->swa_window_block_idxs.back() + 1)) *
        block_token_num;
    if (shareable_kvcache_token_num < swa_right_token_pos) {
      return Status(RET_RUNTIME_FAILED,
                    FormatStr("ConfirmRemoteCacheLoadingResult: req %d shareable_kvcache_token_num %zu < "
                              "swa_right_token_pos %zu.",
                              req_id, shareable_kvcache_token_num, swa_right_token_pos));
    }
  } else {
    // SWA disabled: must cover at least fa_prefix_block_num blocks.
    const size_t fa_max_token_pos = fa_prefix_block_num * block_token_num;
    if (shareable_kvcache_token_num < fa_max_token_pos) {
      return Status(RET_RUNTIME_FAILED,
                    FormatStr("ConfirmRemoteCacheLoadingResult: req %d shareable_kvcache_token_num %zu < "
                              "fa_max_token_pos %zu.",
                              req_id, shareable_kvcache_token_num, fa_max_token_pos));
    }
  }

  // Step 1 (no SWA): truncate cached_blocks to shareable_block_num, then call UpdateRequestTokens
  // only if new blocks were loaded beyond the original fa_prefix_block_num.
  if (swa_cache_manager_ == nullptr) {
    TruncateRequestBlocks(cached_request, shareable_block_num);

    cached_request->shared_block_num = fa_prefix_block_num;
    cached_request->is_remote_loading = false;
    cached_request->fa_prefix_block_num = 0;
    // G4a: Transition to kRunning only if blocks were allocated (kRemoteLoadingBlockAllocated).
    // If no blocks were allocated (kRemoteLoading), stay in kWaiting so the scheduler
    // knows to re-allocate before dispatching.
    cached_request->req_state =
        cached_request->is_remote_loading_block_allocated ? RequestState::kRunning : RequestState::kWaiting;
    cached_request->is_remote_loading_block_allocated = false;

    if (cached_request->cached_blocks.size() > fa_prefix_block_num) {
      std::vector<int> req_block_ids;
      bool block_merged = false;
      std::vector<int> swa_req_block_ids;
      std::vector<size_t> swa_block_idx_offsets;
      return UpdateRequestTokens(req_id, kvcached_token_ids, shareable_kvcache_token_num, req_block_ids, block_merged,
                                 swa_req_block_ids, swa_block_idx_offsets);
    }
    return Status();
  }

  // Step 2 (SWA enabled): SWA window block count is assumed to be 1.
  // matched_remote_swa_block_indexs.size() <= 1.
  //
  // Step 2.1: Adjust FA cache and SWA cache.
  //
  // Case 1 (size == 0): no remote SWA match.
  //   Free all remote SWA blocks, then truncate cached_blocks to swa_aligned_block_num
  //   (= swa_window_block_idxs.back() + 1) so that FA cache
  //   and SWA cache are aligned at the SWA window's right boundary.
  //
  // Case 2 (size == 1): remote SWA matched.
  //   k = matched_remote_swa_block_indexs[0].
  //   old_fa_block_id = remote_cache_swa_fa_block_ids[k] is the FA block that currently
  //   holds the remote-loaded SWA content.
  //   dst_fa_block_id = cached_blocks[shareable_block_num - 1]->block_id is the target FA block.
  //   Release the old SWA window (SetReclaimable) so that dst_fa_block_id has no bound SWA block,
  //   satisfying Transfer's precondition.
  //   If old_fa_block_id != dst_fa_block_id, Transfer the SWA block to dst_fa_block_id.
  //   Otherwise call SetSWABlockContentFixed directly.
  //   Free all unmatched remote SWA blocks.
  //   Rebuild the SWA window as [shareable_block_num - 1, shareable_block_num).
  //   Truncate cached_blocks to shareable_block_num.

  if (matched_remote_swa_block_indexs.empty()) {
    // Case 1: no remote SWA match.
    // G7: Free all remote SWA blocks (keyed by remote FA block ids) before clearing the list.
    // When is_remote_loading_block_allocated=true, these SWA blocks were allocated by
    // AllocateBlocksForRemoteCache and must be freed here to avoid a SWA block leak.
    if (swa_cache_manager_ != nullptr) {
      for (size_t fa_block_id : cached_request->remote_cache_swa_fa_block_ids) {
        swa_cache_manager_->FreeSWABlock(fa_block_id);
      }
    }
    cached_request->remote_cache_swa_fa_block_ids.clear();

    // Truncate FA cache to align with the SWA cache right boundary.
    // fa_prefix_block_num >= swa cache length, so we must truncate cached_blocks down to
    // swa_window_block_idxs.back() + 1 (the block count that covers exactly the SWA window's
    // rightmost FA block).
    // FreeSWABlock calls inside TruncateRequestBlocks are no-ops for already-freed SWA blocks.
    const size_t swa_aligned_block_num =
        cached_request->swa_window_block_idxs.empty() ? 0 : (cached_request->swa_window_block_idxs.back() + 1);
    KLLM_CHECK_WITH_INFO(swa_aligned_block_num == shareable_block_num,
                         FormatStr("swa_aligned_block_num %zu should be equal to shareable_block_num %zu.",
                                   swa_aligned_block_num, shareable_block_num));
    TruncateRequestBlocks(cached_request, swa_aligned_block_num);
  } else {
    // Case 2: remote SWA matched (exactly one match since window size == 1).
    const size_t k = matched_remote_swa_block_indexs[0];
    const size_t old_fa_block_id = cached_request->remote_cache_swa_fa_block_ids[k];
    const size_t dst_fa_block_id = cached_request->cached_blocks[shareable_block_num - 1]->block_id;

    // Release the old SWA window BEFORE Transfer so that dst_fa_block_id has no bound SWA block
    // (Transfer precondition: dst must have no associated SWA block).
    auto& ids = cached_request->swa_window_fa_block_ids;
    auto& idxs = cached_request->swa_window_block_idxs;
    for (size_t fa_id : ids) {
      swa_cache_manager_->SetReclaimable(req_id, fa_id);
    }
    ids.clear();
    idxs.clear();

    // Free all unmatched remote SWA blocks.
    const size_t remote_swa_count = cached_request->remote_cache_swa_fa_block_ids.size();
    for (size_t i = 0; i < remote_swa_count; ++i) {
      if (i != k) {
        swa_cache_manager_->FreeSWABlock(cached_request->remote_cache_swa_fa_block_ids[i]);
      }
    }
    cached_request->remote_cache_swa_fa_block_ids.clear();

    // Migrate the remote SWA block to the correct FA block.
    if (old_fa_block_id != dst_fa_block_id) {
      // Transfer re-keys the SWA block from the remote FA block to the prefix-tree FA block.
      swa_cache_manager_->Transfer(old_fa_block_id, dst_fa_block_id, shareable_block_num - 1);
    } else {
      // The remote FA block is already the correct prefix-tree FA block; just mark it fixed.
      swa_cache_manager_->SetSWABlockContentFixed(dst_fa_block_id);
    }

    // Rebuild the SWA window pointing to the new rightmost FA block.
    ids.push_back(dst_fa_block_id);
    idxs.push_back(shareable_block_num - 1);

    // Truncate cached_blocks to shareable_block_num.
    TruncateRequestBlocks(cached_request, shareable_block_num);
  }

  // Step 2.2: FA cache and SWA cache are now aligned.
  // All FA blocks and SWA blocks in cached_blocks are valid and usable.
  // Call UpdateRequestTokens only if new blocks were loaded beyond fa_prefix_block_num.
  // In Case 1 (no match), cached_blocks was truncated to swa_aligned_block_num <= fa_prefix_block_num,
  // so cached_blocks.size() == cached_request->cached_blocks.size() is the correct shared count.
  cached_request->shared_block_num =
      matched_remote_swa_block_indexs.empty() ? cached_request->cached_blocks.size() : shareable_block_num;
  cached_request->is_remote_loading = false;
  cached_request->fa_prefix_block_num = 0;
  // G4b: Same sub-state branching as G4a for the SWA-enabled path.
  cached_request->req_state =
      cached_request->is_remote_loading_block_allocated ? RequestState::kRunning : RequestState::kWaiting;
  cached_request->is_remote_loading_block_allocated = false;

  if (cached_request->cached_blocks.size() > fa_prefix_block_num) {
    std::vector<int> req_block_ids;
    bool block_merged = false;
    std::vector<int> swa_req_block_ids;
    std::vector<size_t> swa_block_idx_offsets;
    return UpdateRequestTokens(req_id, kvcached_token_ids, shareable_kvcache_token_num, req_block_ids, block_merged,
                               swa_req_block_ids, swa_block_idx_offsets);
  }
  return Status();
}

void PrefixCacheManager::DestroyFinishedRequest(int64_t req_id) {
  const auto it = cached_requests_.find(req_id);
  if (it == cached_requests_.end()) {
    KLLM_LOG_ERROR << "DestroyFinishedRequest req " << req_id << " not exists.";
    return;
  }

  PrefixCachedRequest& cached_request = *it->second.get();

  // G5: Sub-state branching for remote-loading cleanup.
  if (cached_request.is_remote_loading) {
    if (cached_request.is_remote_loading_block_allocated) {
      // kRemoteLoadingBlockAllocated: FA blocks were registered to req_id via
      // AllocateRequestBlocksInternal (active_requests).
      // TruncateRequestBlocks uses req_state to choose active_requests vs inactive_requests,
      // so we must set req_state=kRunning (matching the active_requests registration).
      // TruncateRequestBlocks will call FreeSWABlock for each non-shareable FA block it
      // releases, so no explicit FreeSWABlock/SetReclaimable loops are needed here.
      // Just clear the list references to avoid stale FA block IDs after truncation.
      cached_request.remote_cache_swa_fa_block_ids.clear();
      cached_request.swa_window_fa_block_ids.clear();
      cached_request.req_state = RequestState::kRunning;
      TruncateRequestBlocks(&cached_request, 0);
    } else {
      // kRemoteLoading: FA blocks were NOT registered to req_id (§4.0.3).
      // Drop data structures only; do NOT call TruncateRequestBlocks.
      if (swa_cache_manager_ != nullptr) {
        cached_request.remote_cache_swa_fa_block_ids.clear();
        for (size_t fa_block_id : cached_request.swa_window_fa_block_ids) {
          swa_cache_manager_->SetReclaimable(req_id, fa_block_id);
        }
        cached_request.swa_window_fa_block_ids.clear();
      }
      cached_request.cached_blocks.clear();
    }
    cached_request.is_remote_loading = false;
    cached_request.is_remote_loading_block_allocated = false;
    cached_request.fa_prefix_block_num = 0;
  } else {
    // Normal (non-remote-loading) state.
    // Release SWA window blocks: mark all SWA blocks still held by this request as reclaimable.
    if (swa_cache_manager_ != nullptr) {
      for (size_t fa_block_id : cached_request.swa_window_fa_block_ids) {
        swa_cache_manager_->SetReclaimable(req_id, fa_block_id);
      }
      cached_request.swa_window_fa_block_ids.clear();
    }

    // If not waiting, remove request from associated cache blocks and return non-shareable
    // blocks to the free pool.
    if (cached_request.req_state != RequestState::kWaiting) {
      TruncateRequestBlocks(&cached_request, 0);
    }
  }

  cached_requests_.erase(it);
  KLLM_LOG_DEBUG << "DestroyFinishedRequest req " << req_id << " removed from cached requests.";
}

Status PrefixCacheManager::MergeFilledCachedBlocks(PrefixCachedRequest* cached_request, size_t block_index,
                                                   PrefixCachedBlock* dst_cached_block,
                                                   PrefixCachedBlock* src_cached_block,
                                                   std::vector<int>& req_block_ids) {
  // No children, no requests, just update request which reference src cached block.
  dst_cached_block->active_requests[cached_request->req_id] = std::make_pair(block_index, cached_request);

  // Update request's block to new one.
  cached_request->cached_blocks[block_index] = dst_cached_block;

  // Update internal memory block ids of the request.
  req_block_ids[block_index] = dst_cached_block->memory_block_id;

  return Status();
}

Status PrefixCacheManager::AppendFilledCachedBlock(PrefixCachedRequest* cached_request, size_t block_index,
                                                   PrefixCachedBlock* cached_block, std::vector<int>& req_block_ids,
                                                   bool& block_merged) {
  PrefixCachedBlock* shadow_parent =
      (block_index == 0) ? root_cached_block_ : cached_request->cached_blocks[block_index - 1];

  // Try to find a brother block that have same data
  auto it = shadow_parent->children.find(cached_block->token_ids);
  if (it != shadow_parent->children.end()) {
    PrefixCachedBlock* dst_cached_block = it->second;
    // Found it, merge to existed block.
    MergeFilledCachedBlocks(cached_request, block_index, dst_cached_block, cached_block, req_block_ids);
    // The new FA block is being merged into an existing tree node; its SWA block
    // is no longer needed (the existing node's SWA block will be used instead).
    if (swa_cache_manager_ != nullptr) {
      // 先把当前 req 从源 SWA block 的 active_req_ids 里 erase, 否则下面的
      // FreeSWABlock 会因检测到非空 active_req_ids 而报 lifecycle warn。
      swa_cache_manager_->SetReclaimable(cached_request->req_id, cached_block->block_id);
      swa_cache_manager_->FreeSWABlock(cached_block->block_id);
      // Replace the old FA block id with the merged block's FA block id in
      // swa_window_fa_block_ids so that the window remains consistent.
      // Locate the entry by searching swa_window_block_idxs for block_index.
      auto& ids = cached_request->swa_window_fa_block_ids;
      const auto& idxs = cached_request->swa_window_block_idxs;
      for (size_t k = 0; k < idxs.size(); ++k) {
        if (idxs[k] == block_index) {
          // Replace in-place: the merged block's SWA block is now used by this request.
          ids[k] = dst_cached_block->block_id;
          swa_cache_manager_->ReuseSWABlock(cached_request->req_id, dst_cached_block->block_id);
          break;
        }
      }
      // If block_index is not found in idxs, the block has already slid out of the window.
    }
    // Reset it and move to free list.
    ResetCachedBlock(cached_block);
    free_cached_blocks_.push(cached_block);
    block_merged = true;
    return Status();
  }

  // Not found, mark it shareable, trace request info.
  cached_block->is_shareable = true;
  cached_block->active_requests[cached_request->req_id] = std::make_pair(block_index, cached_request);

  // Mark the associated SWA block as content-fixed (fully written, can be reused).
  if (swa_cache_manager_ != nullptr) {
    swa_cache_manager_->SetSWABlockContentFixed(cached_block->block_id);
  }

  // add to tree block.
  cached_block->parent = shadow_parent;
  cached_block->parent->children[cached_block->token_ids] = cached_block;
  InsertFlexibleCacheMap(cached_block);
  // Record timeline.
  AppendCachedBlockToTimedList(cached_block);

  // Not reusable now.
  reusable_cached_blocks_.erase(cached_block);

  return Status();
}

Status PrefixCacheManager::UpdateRequestTokens(int64_t req_id, const std::vector<int>& kvcached_token_ids,
                                               size_t shareable_kvcache_token_num, std::vector<int>& req_block_ids,
                                               bool& block_merged, std::vector<int>& swa_req_block_ids,
                                               std::vector<size_t>& swa_block_idx_offsets) {
  auto it = cached_requests_.find(req_id);
  if (it == cached_requests_.end()) {
    return Status(RET_RUNTIME_FAILED, FormatStr("Update request token error, req %d is not found.", req_id));
  }

  PrefixCachedRequest* cached_request = it->second.get();

  if (kvcached_token_ids.size() < shareable_kvcache_token_num) {
    return Status(RET_RUNTIME_FAILED,
                  FormatStr("Update request token error,  kvcached_token_ids.size=%d < shareable_kvcache_token_num=%d.",
                            kvcached_token_ids.size(), shareable_kvcache_token_num));
  }

  // Return if block is not full.
  const size_t block_token_num = cache_manager_config_.block_token_num;
  const size_t filled_block_num = shareable_kvcache_token_num / block_token_num;
  if (filled_block_num <= cached_request->shared_block_num) {
    // Even if no new blocks are filled, the SWA window may have advanced.
    // Fall through to the SWA window slide logic below.
  } else {
    if (cached_request->cached_blocks.size() < filled_block_num) {
      return Status(RET_RUNTIME_FAILED,
                    FormatStr("Update request token error,  req %d block size %d less than filled block num %d.",
                              req_id, cached_request->cached_blocks.size(), filled_block_num));
    }

    // Maybe there are more than one blocks should be process.
    for (size_t i = cached_request->shared_block_num; i < filled_block_num; ++i) {
      // Fill token ids
      memcpy(cached_request->cached_blocks[i]->token_ids.data(), kvcached_token_ids.data() + (i * block_token_num),
             block_token_num * sizeof(int));

      // Append new filled block to tree.
      AppendFilledCachedBlock(cached_request, i, cached_request->cached_blocks[i], req_block_ids, block_merged);
    }
    cached_request->shared_block_num = filled_block_num;
  }

  // SWA window sliding: release SWA blocks that have slid out of the window.
  if (swa_cache_manager_ != nullptr && cache_manager_config_.sliding_window_size > 0) {
    const size_t sliding_window_size = cache_manager_config_.sliding_window_size;
    size_t swa_left_block = 0;
    if (shareable_kvcache_token_num > sliding_window_size) {
      const size_t swa_left_token = shareable_kvcache_token_num - sliding_window_size;
      swa_left_block = swa_left_token / block_token_num;
    }

    // Release SWA blocks whose FA block index < swa_left_block.
    // swa_window_fa_block_ids is ordered by block index (oldest first);
    // swa_window_block_idxs[i] holds the FA global index for ids[i].
    auto& ids = cached_request->swa_window_fa_block_ids;
    auto& idxs = cached_request->swa_window_block_idxs;
    while (!ids.empty() && idxs.front() < swa_left_block) {
      swa_cache_manager_->SetReclaimable(req_id, ids.front());
      ids.erase(ids.begin());
      idxs.erase(idxs.begin());
    }

    // Rebuild swa_req_block_ids and swa_block_idx_offsets from the per-block index arrays.
    swa_req_block_ids.clear();
    swa_block_idx_offsets.clear();
    for (size_t i = 0; i < ids.size(); ++i) {
      int mem_id = swa_cache_manager_->GetSWAMemoryBlockId(ids[i]);
      if (mem_id >= 0) {
        swa_req_block_ids.push_back(mem_id);
        swa_block_idx_offsets.push_back(idxs[i]);
      }
    }
  }

  return Status();
}

void PrefixCacheManager::AppendCachedBlockToTimedList(PrefixCachedBlock* cached_block) {
  auto it = timed_cached_block_iters_.find(cached_block);
  if (it != timed_cached_block_iters_.end()) {
    timed_cached_blocks_.erase(it->second);
  }

  timed_cached_blocks_.push_back(cached_block);
  timed_cached_block_iters_[cached_block] = --timed_cached_blocks_.end();
}

void PrefixCacheManager::RemoveCachedBlockFromTimedList(PrefixCachedBlock* cached_block) {
  auto it = timed_cached_block_iters_.find(cached_block);
  if (it != timed_cached_block_iters_.end()) {
    timed_cached_blocks_.erase(it->second);
    timed_cached_block_iters_.erase(it);
  }
}

bool PrefixCacheManager::IsPrefixCachingEnabled() {
  return BaseCacheManager<PrefixCachedBlock, PrefixCachedRequest>::IsPrefixCachingEnabled();
}

}  // namespace ksana_llm
