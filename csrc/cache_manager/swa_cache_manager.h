/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstdint>
#include <list>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "cache_manager/block_allocator/block_allocator_manager.h"

namespace ksana_llm {

// Describes the state of a single SWA (Sliding Window Attention) KV cache block
// that is bound to a specific FA (Full Attention) block in the prefix tree.
struct SWACachedBlock {
  // The FA block id (PrefixCachedBlock::block_id) this SWA block is bound to.
  size_t fa_block_id = 0;

  // The actual SWA device memory block id (from swa_block_allocator_group_).
  // -1 means no memory block is allocated.
  int memory_block_id = -1;

  // Whether the SWA KV cache content is fixed and can be reused by other requests.
  bool is_content_fixed = false;

  // Number of times a new request has hit the prefix cache entry that uses this SWA block.
  // Used for LFU eviction: blocks with lower hit_count are evicted first.
  size_t hit_count = 0;

  // Set of req_ids currently holding this SWA block in their SWA window.
  // The block is reclaimable only when this set is empty.
  // Replaces the old boolean is_reclaimable: reclaimable == active_req_ids.empty().
  std::unordered_set<int64_t> active_req_ids;

  // The number of FA blocks that precede this block in the FA block sequence
  // (i.e., the block_index of this block). Used as a secondary sort key in the
  // LFU eviction policy: when hit_count is equal, blocks with a smaller
  // prefix_fa_block_num are evicted first (they save less computation on reuse).
  size_t prefix_fa_block_num = 0;

  // Helper: returns true when no request is actively using this block.
  bool IsReclaimable() const { return active_req_ids.empty(); }
};

// Manages the binding between FA blocks (in the prefix tree) and SWA memory blocks.
// Implements an LFU-based eviction policy for reclaimable content-fixed SWA blocks.
//
// Thread-safety: all public methods must be called under the same lock as the
// enclosing PrefixCacheManager (i.e., the caller is responsible for locking).
class SWACacheManager {
 public:
  SWACacheManager(std::shared_ptr<BlockAllocatorGroupInterface> swa_block_allocator_group, size_t sliding_window_size,
                  size_t block_token_num);

  // Allocate a new SWA block for the given FA block and register req_id as its first holder.
  // If the FA block already has an associated SWA block that is reclaimable (no active
  // requests), the old SWA block is freed first and a new one is allocated.
  // Internally registers req_id as an active holder on success so the caller does not need
  // to call ReuseSWABlock separately.
  // prefix_fa_block_num: the block_index of this FA block in the FA block sequence,
  //                      used for eviction priority decisions.
  // Returns the allocated SWA memory_block_id, or -1 on failure.
  int AllocateSWABlock(int64_t req_id, size_t fa_block_id, size_t prefix_fa_block_num);

  // Mark the SWA block bound to fa_block_id as content-fixed (fully written and reusable).
  // No-op if the block does not exist.
  void SetSWABlockContentFixed(size_t fa_block_id);

  // Register req_id as a new holder of an existing content-fixed SWA block and increment
  // its prefix-cache hit count.  Called when a new request hits the prefix cache and reuses
  // an already-allocated SWA block (as opposed to AllocateSWABlock which creates a new one).
  // Removes the block from the reclaimable pool if it was reclaimable.
  // No-op if the block does not exist.
  void ReuseSWABlock(int64_t req_id, size_t fa_block_id);

  // Release req_id's hold on the SWA block bound to fa_block_id.
  // Removes req_id from active_req_ids; the block enters the reclaimable pool only when
  // active_req_ids becomes empty (i.e. no other request still holds it).
  // No-op if the block does not exist.
  void SetReclaimable(int64_t req_id, size_t fa_block_id);

  // Free the SWA block bound to fa_block_id and release its memory back to the allocator.
  // Logs a WARNING if active_req_ids is non-empty (lifecycle bug in caller), but still frees.
  // No-op if the block does not exist.
  void FreeSWABlock(size_t fa_block_id);

  // Move the SWA block currently bound to src_fa_block_id to dst_fa_block_id.
  // dst_prefix_fa_block_num is the new block_index for the destination FA block.
  //
  // Preconditions (enforced; returns false if violated):
  //   1. src_fa_block_id must have an associated SWA block.
  //   2. The src SWA block must have is_content_fixed=false (remote-loaded content not yet fixed).
  //   3. The src SWA block must be held by exactly one active request (no sharing).
  //   4. dst_fa_block_id must NOT have any associated SWA block (not even a reclaimable one).
  //
  // On success the SWA block is re-keyed from src to dst; active_req_ids, memory_block_id,
  // and all other fields are preserved except fa_block_id and prefix_fa_block_num.
  // Returns true on success, false on failure.
  bool Transfer(size_t src_fa_block_id, size_t dst_fa_block_id, size_t dst_prefix_fa_block_num);

  // Return true if a valid (allocated and content-fixed) SWA block exists for fa_block_id.
  bool IsSWABlockValid(size_t fa_block_id) const;

  // Return the SWA memory_block_id for fa_block_id, or -1 if not found.
  int GetSWAMemoryBlockId(size_t fa_block_id) const;

  // Return the total number of SWA blocks currently available for allocation:
  //   free pool (from swa_block_allocator_group_) + reclaimable pool.
  // Used by AllocateRequestBlocks as a pre-check before starting any allocation,
  // to avoid a half-allocated state where FA blocks are taken but SWA blocks run out.
  // This is a pure query: it does not trigger eviction or modify any state.
  size_t GetAvailableBlockCount() const;

  // Overload that excludes reclaimable SWA blocks whose FA block id appears in
  // reserved_fa_block_ids from the available count.
  //
  // Background: swa_window_fa_block_ids holds FA block ids whose SWA blocks are
  // currently reclaimable (no active request holds them yet) but will be
  // re-acquired by ReuseSWABlock inside AllocateRequestBlocksInternal.  Without
  // this exclusion, GetAvailableBlockCount() counts those blocks as "free to
  // allocate", causing the pre-check to overestimate availability.  When
  // ReuseSWABlock later removes them from the reclaimable pool the remaining
  // pool may be too small to satisfy the new swa_block_num allocation, leading
  // to an allocation failure after FA blocks have already been consumed.
  //
  // reserved_fa_block_ids: the set of FA block ids that will be re-acquired
  //   (via ReuseSWABlock) and must therefore NOT be counted as available.
  // This is a pure query: it does not trigger eviction or modify any state.
  size_t GetAvailableBlockCount(const std::vector<size_t>& reserved_fa_block_ids) const;

  // Increment the prefix-cache hit count for the SWA block bound to fa_block_id.
  // Re-inserts the block into the LFU list at the correct position if it is currently
  // reclaimable.  No-op if the block does not exist.
  // Called at query time (GetRequestPrefixBlockNumberInternal Phase 2) so that every
  // prefix-cache hit is counted regardless of whether AllocateRequestBlocks follows.
  void IncrementHitCount(size_t fa_block_id);

  // Register req_id as an active holder of the SWA block bound to fa_block_id.
  // Removes the block from the reclaimable pool if it was reclaimable.
  // No-op if the block does not exist.
  // Called by AllocateRequestBlocksInternal after the availability pre-check passes.
  void AcquireSWABlock(int64_t req_id, size_t fa_block_id);

 private:
  // Evict one reclaimable SWA block according to the LFU + prefix_fa_block_num policy.
  // Returns the fa_block_id of the evicted block, or SIZE_MAX if nothing was evicted.
  size_t EvictOneSWABlock();

  // Insert fa_block_id into the LFU list at the correct position (sorted by
  // hit_count ascending, then prefix_fa_block_num ascending).
  void InsertIntoLfuList(size_t fa_block_id);

  // Remove fa_block_id from the LFU list (if present).
  void RemoveFromLfuList(size_t fa_block_id);

  std::shared_ptr<BlockAllocatorGroupInterface> swa_block_allocator_group_;
  size_t sliding_window_size_;
  size_t block_token_num_;

  // fa_block_id -> SWACachedBlock
  std::unordered_map<size_t, std::unique_ptr<SWACachedBlock>> swa_blocks_;

  // Reclaimable non-content-fixed SWA blocks (highest eviction priority).
  // These blocks have is_content_fixed=false and active_req_ids is empty.
  std::unordered_set<size_t> reclaimable_non_content_fixed_;

  // Reclaimable content-fixed SWA blocks sorted by (hit_count ASC, prefix_fa_block_num ASC).
  // Front of the list = lowest value = first to be evicted.
  std::list<size_t> reclaimable_content_fixed_lfu_;

  // fa_block_id -> iterator into reclaimable_content_fixed_lfu_ for O(1) removal.
  std::unordered_map<size_t, std::list<size_t>::iterator> lfu_iters_;
};

}  // namespace ksana_llm
