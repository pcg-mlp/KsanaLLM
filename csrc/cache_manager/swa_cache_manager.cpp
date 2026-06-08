/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "cache_manager/swa_cache_manager.h"

#include <climits>

#include "utils/logger.h"
#include "utils/string_utils.h"

namespace ksana_llm {

SWACacheManager::SWACacheManager(std::shared_ptr<BlockAllocatorGroupInterface> swa_block_allocator_group,
                                 size_t sliding_window_size, size_t block_token_num,
                                 bool is_contiguous_swa_mode)
    : swa_block_allocator_group_(swa_block_allocator_group),
      sliding_window_size_(sliding_window_size),
      block_token_num_(block_token_num),
      is_contiguous_swa_mode_(is_contiguous_swa_mode) {}

int SWACacheManager::GetAllocatableSWACache(int64_t req_id, size_t cur_fa_block_num, size_t new_fa_block_num,
                                            size_t minimum_swa_block_num, bool disable_swa_block_num_adjust,
                                            std::vector<int>& fa_block_offset,
                                            const std::vector<size_t>& reserved_fa_block_ids) {
  fa_block_offset.clear();

  // Determine the required block count based on the current mode.
  const size_t needed = is_contiguous_swa_mode_ && !disable_swa_block_num_adjust ? (new_fa_block_num - cur_fa_block_num)
                                                                                 : minimum_swa_block_num;

  // Use GetAvailableBlockCount(reserved_fa_block_ids) so that reclaimable SWA blocks that
  // belong to the caller's current window (and will be re-acquired via AcquireSWABlock after
  // this check) are not counted as free capacity.  This avoids a race where the check passes
  // but AcquireSWABlock later shrinks the pool below what AllocateSWABlock needs.
  if (needed > 0 && GetAvailableBlockCount(reserved_fa_block_ids) < needed) {
    return -1;
  }

  if (is_contiguous_swa_mode_ && !disable_swa_block_num_adjust) {
    // Contiguous mode: bind an SWA block to every new FA block position.
    for (size_t i = cur_fa_block_num; i < new_fa_block_num; ++i) {
      fa_block_offset.push_back(static_cast<int>(i));
    }
  } else {
    // Non-contiguous mode: first bind the trailing minimum_swa_block_num positions,
    // i.e. [new_fa_block_num - minimum_swa_block_num, new_fa_block_num).
    // Future policy extensions may additionally allocate earlier positions when surplus
    // blocks are available.
    if (minimum_swa_block_num > 0) {
      const size_t tail_start = new_fa_block_num - minimum_swa_block_num;
      for (size_t i = tail_start; i < new_fa_block_num; ++i) {
        fa_block_offset.push_back(static_cast<int>(i));
      }
    }
    // TODO(future): extend to [cur_fa_block_num, tail_start) based on remaining availability.
  }

  return 0;
}

int SWACacheManager::AllocateSWABlock(int64_t req_id, size_t fa_block_id, size_t prefix_fa_block_num) {
  // If this FA block already has an associated SWA block, only allow replacement
  // when the existing block is reclaimable (no active requests; FA block reuse path
  // from reusable_cached_blocks_).  A block still held by active requests must not
  // be replaced, as doing so would corrupt those requests' SWA cache.
  auto existing_it = swa_blocks_.find(fa_block_id);
  if (existing_it != swa_blocks_.end()) {
    if (!existing_it->second->active_req_ids.empty()) {
      KLLM_LOG_WARNING << "AllocateSWABlock: fa_block_id=" << fa_block_id
                       << " already has a non-reclaimable SWA block; refusing to replace it";
      return -1;
    }
    FreeSWABlock(fa_block_id);
  }

  // Priority a: try to allocate from the free pool directly.
  std::vector<int> blocks;
  swa_block_allocator_group_->GetDeviceBlockAllocator(0)->AllocateBlocks(1, blocks);

  if (blocks.empty()) {
    // Priority b: evict a reclaimable SWA block to reclaim its memory.
    size_t evicted_fa_block_id = EvictOneSWABlock();
    if (evicted_fa_block_id == SIZE_MAX) {
      KLLM_LOG_WARNING << "AllocateSWABlock: no free SWA blocks and no reclaimable blocks available for fa_block_id="
                       << fa_block_id;
      return -1;
    }
    // After eviction the freed memory is back in the allocator pool; retry.
    swa_block_allocator_group_->GetDeviceBlockAllocator(0)->AllocateBlocks(1, blocks);
    if (blocks.empty()) {
      KLLM_LOG_WARNING << "AllocateSWABlock: allocation still failed after eviction for fa_block_id=" << fa_block_id;
      return -1;
    }
  }

  auto swa_block = std::make_unique<SWACachedBlock>();
  swa_block->fa_block_id = fa_block_id;
  swa_block->memory_block_id = blocks[0];
  swa_block->is_content_fixed = false;
  swa_block->hit_count = 0;
  swa_block->prefix_fa_block_num = prefix_fa_block_num;

  int memory_block_id = swa_block->memory_block_id;
  swa_blocks_[fa_block_id] = std::move(swa_block);

  // Register req_id as the initial holder of this block (allocation implies active use).
  // Use AcquireSWABlock directly — no hit-count increment for a freshly allocated block.
  AcquireSWABlock(req_id, fa_block_id);

  return memory_block_id;
}

void SWACacheManager::SetSWABlockContentFixed(size_t fa_block_id) {
  auto it = swa_blocks_.find(fa_block_id);
  if (it == swa_blocks_.end()) {
    return;
  }
  it->second->is_content_fixed = true;
}

void SWACacheManager::ReuseSWABlock(int64_t req_id, size_t fa_block_id) {
  // Register req_id as a new holder (removes block from reclaimable pool if needed).
  AcquireSWABlock(req_id, fa_block_id);
  // Increment hit count to reflect that a new request reused this prefix-cached block.
  IncrementHitCount(fa_block_id);
}

void SWACacheManager::SetReclaimable(int64_t req_id, size_t fa_block_id) {
  auto it = swa_blocks_.find(fa_block_id);
  if (it == swa_blocks_.end()) {
    return;
  }

  SWACachedBlock* block = it->second.get();
  bool was_reclaimable = block->active_req_ids.empty();

  // req_id stops using this block.
  block->active_req_ids.erase(req_id);

  bool now_reclaimable = block->active_req_ids.empty();

  if (now_reclaimable && !was_reclaimable) {
    // Transition to reclaimable: add to the appropriate reclaimable set.
    if (block->is_content_fixed) {
      InsertIntoLfuList(fa_block_id);
    } else {
      reclaimable_non_content_fixed_.insert(fa_block_id);
    }
  }
}

void SWACacheManager::AcquireSWABlock(int64_t req_id, size_t fa_block_id) {
  auto it = swa_blocks_.find(fa_block_id);
  if (it == swa_blocks_.end()) {
    return;
  }

  SWACachedBlock* block = it->second.get();
  bool was_reclaimable = block->active_req_ids.empty();

  // req_id starts using this block.
  block->active_req_ids.insert(req_id);

  if (was_reclaimable) {
    // Transition to non-reclaimable: remove from the reclaimable sets.
    if (block->is_content_fixed) {
      RemoveFromLfuList(fa_block_id);
    } else {
      reclaimable_non_content_fixed_.erase(fa_block_id);
    }
  }
}

void SWACacheManager::IncrementHitCount(size_t fa_block_id) {
  auto it = swa_blocks_.find(fa_block_id);
  if (it == swa_blocks_.end()) {
    return;
  }

  SWACachedBlock* block = it->second.get();

  // If the block is currently in the LFU list, remove it, increment, then re-insert
  // at the correct sorted position.
  bool in_lfu = (lfu_iters_.find(fa_block_id) != lfu_iters_.end());
  if (in_lfu) {
    RemoveFromLfuList(fa_block_id);
  }

  block->hit_count++;

  if (in_lfu) {
    InsertIntoLfuList(fa_block_id);
  }
}

void SWACacheManager::FreeSWABlock(size_t fa_block_id) {
  auto it = swa_blocks_.find(fa_block_id);
  if (it == swa_blocks_.end()) {
    return;
  }

  SWACachedBlock* block = it->second.get();

  // Warn if active requests still hold this block — indicates a lifecycle bug in the caller.
  if (!block->active_req_ids.empty()) {
    KLLM_LOG_WARNING << "FreeSWABlock: fa_block_id=" << fa_block_id << " still has " << block->active_req_ids.size()
                     << " active request(s); freeing anyway";
    block->active_req_ids.clear();
  }

  // Remove from reclaimable tracking structures (block is reclaimable when active_req_ids is empty).
  if (block->IsReclaimable()) {
    if (block->is_content_fixed) {
      RemoveFromLfuList(fa_block_id);
    } else {
      reclaimable_non_content_fixed_.erase(fa_block_id);
    }
  }

  // Return the memory block to the allocator.
  if (block->memory_block_id >= 0) {
    swa_block_allocator_group_->GetDeviceBlockAllocator(0)->FreeBlocks({block->memory_block_id});
  }

  swa_blocks_.erase(it);
}

bool SWACacheManager::IsSWABlockValid(size_t fa_block_id) const {
  auto it = swa_blocks_.find(fa_block_id);
  if (it == swa_blocks_.end()) {
    return false;
  }
  return it->second->is_content_fixed && it->second->memory_block_id >= 0;
}

int SWACacheManager::GetSWAMemoryBlockId(size_t fa_block_id) const {
  auto it = swa_blocks_.find(fa_block_id);
  if (it == swa_blocks_.end()) {
    return -1;
  }
  return it->second->memory_block_id;
}

size_t SWACacheManager::GetAvailableBlockCount() const {
  // Free pool: blocks not yet assigned to any FA block.
  size_t free_count = swa_block_allocator_group_->GetDeviceBlockAllocator(0)->GetFreeBlockNumber();
  // Reclaimable pool: blocks that can be evicted on demand.
  size_t reclaimable_count = reclaimable_non_content_fixed_.size() + reclaimable_content_fixed_lfu_.size();
  return free_count + reclaimable_count;
}

size_t SWACacheManager::GetAvailableBlockCount(const std::vector<size_t>& reserved_fa_block_ids) const {
  // Start from the baseline available count.
  size_t available = GetAvailableBlockCount();

  // Subtract reclaimable blocks that appear in reserved_fa_block_ids.
  // These blocks are currently counted as reclaimable (available) but will be
  // re-acquired by ReuseSWABlock inside AllocateRequestBlocksInternal, so they
  // must not be counted as free capacity for new SWA block allocations.
  for (size_t fa_id : reserved_fa_block_ids) {
    // Only subtract if the block is actually in one of the reclaimable pools.
    // A block that is already held by an active request is not in any reclaimable
    // pool and therefore was not counted in GetAvailableBlockCount() to begin with.
    if (reclaimable_non_content_fixed_.count(fa_id) > 0 || lfu_iters_.find(fa_id) != lfu_iters_.end()) {
      if (available > 0) {
        --available;
      }
    }
  }

  return available;
}

size_t SWACacheManager::EvictOneSWABlock() {
  // Priority: evict non-content-fixed reclaimable blocks first (they have no prefix cache value).
  if (!reclaimable_non_content_fixed_.empty()) {
    size_t fa_block_id = *reclaimable_non_content_fixed_.begin();
    FreeSWABlock(fa_block_id);
    return fa_block_id;
  }

  // Fall back to LFU eviction among content-fixed reclaimable blocks.
  // content-fixed block 持有可被后续请求复用的 prefix cache 数据, 驱逐意味着该缓存丢失。
  if (!reclaimable_content_fixed_lfu_.empty()) {
    size_t fa_block_id = reclaimable_content_fixed_lfu_.front();
    KLLM_LOG_DEBUG << "EvictOneSWABlock evicted content-fixed fa_block_id=" << fa_block_id
                   << " (prefix cache data dropped)";
    FreeSWABlock(fa_block_id);
    return fa_block_id;
  }

  return SIZE_MAX;
}

void SWACacheManager::InsertIntoLfuList(size_t fa_block_id) {
  auto it = swa_blocks_.find(fa_block_id);
  if (it == swa_blocks_.end()) {
    return;
  }
  SWACachedBlock* block = it->second.get();

  // Find the correct insertion position: sorted by (hit_count ASC, prefix_fa_block_num ASC).
  // Scan from the back to find the last element that is "less than or equal" to the new block,
  // then insert after it (i.e., find the first element strictly greater and insert before it).
  auto insert_pos = reclaimable_content_fixed_lfu_.begin();
  for (auto list_it = reclaimable_content_fixed_lfu_.begin(); list_it != reclaimable_content_fixed_lfu_.end();
       ++list_it) {
    auto& candidate = swa_blocks_.at(*list_it);
    if (candidate->hit_count < block->hit_count) {
      insert_pos = std::next(list_it);
      continue;
    }
    if (candidate->hit_count == block->hit_count && candidate->prefix_fa_block_num < block->prefix_fa_block_num) {
      insert_pos = std::next(list_it);
      continue;
    }
    break;
  }

  auto new_it = reclaimable_content_fixed_lfu_.insert(insert_pos, fa_block_id);
  lfu_iters_[fa_block_id] = new_it;
}

void SWACacheManager::RemoveFromLfuList(size_t fa_block_id) {
  auto map_it = lfu_iters_.find(fa_block_id);
  if (map_it == lfu_iters_.end()) {
    return;
  }
  reclaimable_content_fixed_lfu_.erase(map_it->second);
  lfu_iters_.erase(map_it);
}

bool SWACacheManager::Transfer(size_t src_fa_block_id, size_t dst_fa_block_id, size_t dst_prefix_fa_block_num) {
  // Precondition 1: source must exist.
  auto src_it = swa_blocks_.find(src_fa_block_id);
  if (src_it == swa_blocks_.end()) {
    KLLM_LOG_WARNING << "Transfer: src_fa_block_id=" << src_fa_block_id << " does not exist";
    return false;
  }

  SWACachedBlock* src_block = src_it->second.get();

  // Precondition 2: src block must have is_content_fixed=false.
  // Remote-loaded content is not yet committed to the prefix cache tree.
  if (src_block->is_content_fixed) {
    KLLM_LOG_WARNING << "Transfer: src_fa_block_id=" << src_fa_block_id
                     << " has is_content_fixed=true; only non-fixed remote blocks may be transferred";
    return false;
  }

  // Precondition 3: src block must be held by exactly one active request (no sharing).
  if (src_block->active_req_ids.size() != 1) {
    KLLM_LOG_WARNING << "Transfer: src_fa_block_id=" << src_fa_block_id << " has " << src_block->active_req_ids.size()
                     << " active request(s); exactly 1 is required for transfer";
    return false;
  }

  // Precondition 4: destination must NOT have any associated SWA block.
  if (swa_blocks_.find(dst_fa_block_id) != swa_blocks_.end()) {
    KLLM_LOG_WARNING << "Transfer: dst_fa_block_id=" << dst_fa_block_id
                     << " already has an associated SWA block; refusing to transfer";
    return false;
  }

  // Move the SWA block record from src to dst.
  std::unique_ptr<SWACachedBlock> block = std::move(src_it->second);
  swa_blocks_.erase(src_it);

  // Update the block's metadata to reflect the new FA block binding.
  block->fa_block_id = dst_fa_block_id;
  block->prefix_fa_block_num = dst_prefix_fa_block_num;

  swa_blocks_[dst_fa_block_id] = std::move(block);

  // The src block has is_content_fixed=false and exactly one active request,
  // so it is NOT in any reclaimable structure. No LFU/reclaimable bookkeeping needed.

  return true;
}

}  // namespace ksana_llm
