/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "cache_manager/swa_cache_manager.h"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "test.h"
#include "cache_manager/prefix_cache_manager_test_helper.h"

using namespace ksana_llm;

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------

// Helper that builds a SWACacheManager backed by a FakedBlockAllocatorGroup.
// swa_block_num controls how many SWA memory blocks are pre-allocated.
class SWACacheManagerTest : public testing::Test {
 protected:
  // Build the manager with the given pool size and window configuration.
  void SetUp(size_t swa_block_num, size_t sliding_window_size = 32, size_t block_token_num = 16) {
    swa_block_num_ = swa_block_num;
    sliding_window_size_ = sliding_window_size;
    block_token_num_ = block_token_num;

    BlockAllocatorGroupConfig cfg;
    cfg.devices = {0};
    cfg.device_block_num = swa_block_num_;
    cfg.host_block_num = 0;
    cfg.block_size = block_token_num_ * sizeof(float);

    context_ = std::make_shared<Context>();
    memory_allocator_ = std::make_shared<FakedMemoryAllocator>();

    BlockAllocatorCreationFunc creation_fn = [](MemoryLocation location, size_t block_num, size_t block_size, int rank,
                                                std::shared_ptr<MemoryAllocatorInterface> memory_allocator,
                                                std::shared_ptr<Context> context)
        -> std::shared_ptr<BlockAllocatorInterface> {
      return std::make_shared<FakedBlockAllocator>(location, block_num, block_size, rank, memory_allocator, context);
    };

    allocator_group_ = std::make_shared<FakedBlockAllocatorGroup>(cfg, memory_allocator_, context_, creation_fn);

    manager_ = std::make_unique<SWACacheManager>(allocator_group_, sliding_window_size_, block_token_num_);
  }

  // Convenience: free block count in the underlying allocator.
  size_t FreeBlocks() const { return allocator_group_->GetDeviceBlockAllocator(0)->GetFreeBlockNumber(); }

  size_t swa_block_num_ = 0;
  size_t sliding_window_size_ = 32;
  size_t block_token_num_ = 16;

  std::shared_ptr<Context> context_;
  std::shared_ptr<MemoryAllocatorInterface> memory_allocator_;
  std::shared_ptr<BlockAllocatorGroupInterface> allocator_group_;
  std::unique_ptr<SWACacheManager> manager_;
};

// ===========================================================================
// AllocateSWABlock
// ===========================================================================

// Basic allocation: returns a valid memory_block_id and consumes one free block.
TEST_F(SWACacheManagerTest, AllocateSWABlock_Basic) {
  SetUp(/*swa_block_num=*/4);

  int mem_id = manager_->AllocateSWABlock(/*req_id=*/1001, /*fa_block_id=*/1, /*prefix_fa_block_num=*/0);
  EXPECT_GE(mem_id, 0);
  EXPECT_EQ(FreeBlocks(), 3u);
}

// Newly allocated block is NOT content-fixed.
TEST_F(SWACacheManagerTest, AllocateSWABlock_NotContentFixedAfterAlloc) {
  SetUp(/*swa_block_num=*/4);

  manager_->AllocateSWABlock(1001, 1, 0);
  // IsSWABlockValid requires is_content_fixed=true; must be false right after alloc.
  EXPECT_FALSE(manager_->IsSWABlockValid(1));
  // But the memory block id is valid.
  EXPECT_GE(manager_->GetSWAMemoryBlockId(1), 0);
}

// Re-allocating for the same fa_block_id is only allowed when the existing SWA
// block is reclaimable (FA block reuse path from reusable_cached_blocks_).
// The old reclaimable block is freed first, then a new one is allocated —
// net free-block count stays the same.
TEST_F(SWACacheManagerTest, AllocateSWABlock_ReplacesReclaimableExistingBlock) {
  SetUp(/*swa_block_num=*/4);

  int first_mem_id = manager_->AllocateSWABlock(1001, 1, 0);
  EXPECT_GE(first_mem_id, 0);
  EXPECT_EQ(FreeBlocks(), 3u);

  // Mark the block as reclaimable before re-allocating (simulates FA block reuse
  // after the block was returned to reusable_cached_blocks_).
  manager_->SetReclaimable(1001, 1);

  int second_mem_id = manager_->AllocateSWABlock(1002, 1, 0);
  EXPECT_GE(second_mem_id, 0);
  // Old block freed, new block taken: net count unchanged.
  EXPECT_EQ(FreeBlocks(), 3u);
  // New block is not content-fixed yet.
  EXPECT_FALSE(manager_->IsSWABlockValid(1));
}

// Re-allocating for the same fa_block_id when the existing SWA block is
// non-reclaimable (still actively used) must be refused (returns -1).
TEST_F(SWACacheManagerTest, AllocateSWABlock_RefusesNonReclaimableExistingBlock) {
  SetUp(/*swa_block_num=*/4);

  int first_mem_id = manager_->AllocateSWABlock(1001, 1, 0);
  EXPECT_GE(first_mem_id, 0);
  // Block is non-reclaimable by default after allocation (req 1001 holds it).

  // Attempting to re-allocate while the block is still in active use must fail.
  int result = manager_->AllocateSWABlock(1002, 1, 0);
  EXPECT_EQ(result, -1);

  // The original block must still be intact.
  EXPECT_EQ(manager_->GetSWAMemoryBlockId(1), first_mem_id);
  EXPECT_EQ(FreeBlocks(), 3u);
}

// When the free pool is empty and no reclaimable blocks exist, returns -1.
TEST_F(SWACacheManagerTest, AllocateSWABlock_FailsWhenPoolEmpty) {
  SetUp(/*swa_block_num=*/2);

  // Exhaust the pool.
  EXPECT_GE(manager_->AllocateSWABlock(1001, 1, 0), 0);
  EXPECT_GE(manager_->AllocateSWABlock(1001, 2, 0), 0);
  EXPECT_EQ(FreeBlocks(), 0u);

  // No reclaimable blocks → allocation must fail.
  int result = manager_->AllocateSWABlock(1001, 3, 0);
  EXPECT_EQ(result, -1);
}

// When the free pool is empty but a reclaimable non-content-fixed block exists,
// that block is evicted and the new allocation succeeds.
TEST_F(SWACacheManagerTest, AllocateSWABlock_EvictsReclaimableNonContentFixed) {
  SetUp(/*swa_block_num=*/1);

  // Allocate the only block for FA block 10.
  EXPECT_GE(manager_->AllocateSWABlock(1001, 10, 0), 0);
  EXPECT_EQ(FreeBlocks(), 0u);

  // Mark FA block 10 as reclaimable (non-content-fixed, since SetSWABlockContentFixed was not called).
  manager_->SetReclaimable(1001, 10);

  // Now allocate for FA block 20: should evict FA block 10 and succeed.
  int mem_id = manager_->AllocateSWABlock(1002, 20, 0);
  EXPECT_GE(mem_id, 0);

  // FA block 10 no longer has a valid SWA block.
  EXPECT_FALSE(manager_->IsSWABlockValid(10));
  EXPECT_EQ(manager_->GetSWAMemoryBlockId(10), -1);
}

// When the free pool is empty but a reclaimable content-fixed block exists,
// that block is evicted (LFU) and the new allocation succeeds.
TEST_F(SWACacheManagerTest, AllocateSWABlock_EvictsReclaimableContentFixed) {
  SetUp(/*swa_block_num=*/1);

  EXPECT_GE(manager_->AllocateSWABlock(1001, 5, 0), 0);
  manager_->SetSWABlockContentFixed(5);
  manager_->SetReclaimable(1001, 5);
  EXPECT_TRUE(manager_->IsSWABlockValid(5));

  // Pool is empty; evict FA block 5 to make room for FA block 6.
  int mem_id = manager_->AllocateSWABlock(1002, 6, 0);
  EXPECT_GE(mem_id, 0);
  EXPECT_FALSE(manager_->IsSWABlockValid(5));
}

// ===========================================================================
// SetSWABlockContentFixed / IsSWABlockValid
// ===========================================================================

// After SetSWABlockContentFixed, IsSWABlockValid returns true.
TEST_F(SWACacheManagerTest, SetSWABlockContentFixed_MakesBlockValid) {
  SetUp(/*swa_block_num=*/4);

  manager_->AllocateSWABlock(1001, 1, 0);
  EXPECT_FALSE(manager_->IsSWABlockValid(1));

  manager_->SetSWABlockContentFixed(1);
  EXPECT_TRUE(manager_->IsSWABlockValid(1));
}

// SetSWABlockContentFixed on a non-existent fa_block_id is a no-op (no crash).
TEST_F(SWACacheManagerTest, SetSWABlockContentFixed_NoopOnMissing) {
  SetUp(/*swa_block_num=*/4);
  EXPECT_NO_FATAL_FAILURE(manager_->SetSWABlockContentFixed(999));
}

// IsSWABlockValid returns false for an fa_block_id that was never allocated.
TEST_F(SWACacheManagerTest, IsSWABlockValid_FalseForUnknownId) {
  SetUp(/*swa_block_num=*/4);
  EXPECT_FALSE(manager_->IsSWABlockValid(42));
}

// ===========================================================================
// GetSWAMemoryBlockId
// ===========================================================================

// Returns -1 for an unknown fa_block_id.
TEST_F(SWACacheManagerTest, GetSWAMemoryBlockId_UnknownId) {
  SetUp(/*swa_block_num=*/4);
  EXPECT_EQ(manager_->GetSWAMemoryBlockId(99), -1);
}

// Returns the correct memory_block_id after allocation.
TEST_F(SWACacheManagerTest, GetSWAMemoryBlockId_KnownId) {
  SetUp(/*swa_block_num=*/4);
  int mem_id = manager_->AllocateSWABlock(1001, 7, 0);
  EXPECT_EQ(manager_->GetSWAMemoryBlockId(7), mem_id);
}

// ===========================================================================
// SetReclaimable
// ===========================================================================

// SetReclaimable on a non-content-fixed block adds it to the
// reclaimable_non_content_fixed_ set; GetAvailableBlockCount increases.
TEST_F(SWACacheManagerTest, SetReclaimable_NonContentFixedIncreasesAvailable) {
  SetUp(/*swa_block_num=*/4);

  manager_->AllocateSWABlock(1001, 1, 0);
  size_t avail_before = manager_->GetAvailableBlockCount();  // free pool = 3, reclaimable = 0

  manager_->SetReclaimable(1001, 1);
  size_t avail_after = manager_->GetAvailableBlockCount();  // free pool = 3, reclaimable = 1

  EXPECT_EQ(avail_after, avail_before + 1);
}

// ReuseSWABlock on a reclaimable block removes it from the reclaimable set
// (registers a new holder) and increments the hit count.
TEST_F(SWACacheManagerTest, ReuseSWABlock_RemovesFromReclaimable) {
  SetUp(/*swa_block_num=*/4);

  manager_->AllocateSWABlock(1001, 1, 0);
  manager_->SetReclaimable(1001, 1);
  size_t avail_reclaimable = manager_->GetAvailableBlockCount();

  manager_->ReuseSWABlock(1002, 1);
  size_t avail_active = manager_->GetAvailableBlockCount();

  EXPECT_EQ(avail_active, avail_reclaimable - 1);
}

// SetReclaimable on a non-existent fa_block_id is a no-op (no crash).
TEST_F(SWACacheManagerTest, SetReclaimable_NoopOnMissing) {
  SetUp(/*swa_block_num=*/4);
  EXPECT_NO_FATAL_FAILURE(manager_->SetReclaimable(1001, 999));
}

// Calling SetReclaimable twice with the same req_id does not double-count.
TEST_F(SWACacheManagerTest, SetReclaimable_IdempotentTrue) {
  SetUp(/*swa_block_num=*/4);

  manager_->AllocateSWABlock(1001, 1, 0);
  manager_->SetReclaimable(1001, 1);
  size_t avail_first = manager_->GetAvailableBlockCount();

  manager_->SetReclaimable(1001, 1);  // second call with same req_id — no-op
  EXPECT_EQ(manager_->GetAvailableBlockCount(), avail_first);
}

// ===========================================================================
// ReuseSWABlock / hit count (via LFU eviction order)
// ===========================================================================

// ReuseSWABlock on a non-existent fa_block_id is a no-op (no crash).
TEST_F(SWACacheManagerTest, ReuseSWABlock_NoopOnMissing) {
  SetUp(/*swa_block_num=*/4);
  EXPECT_NO_FATAL_FAILURE(manager_->ReuseSWABlock(1001, 999));
}

// ReuseSWABlock increments hit_count; a block reused more often is evicted later.
TEST_F(SWACacheManagerTest, ReuseSWABlock_AffectsLfuOrder) {
  // Pool of 2 blocks; both content-fixed and reclaimable.
  SetUp(/*swa_block_num=*/2);

  // FA block 10: hit_count stays at 0 (never reused after becoming reclaimable).
  manager_->AllocateSWABlock(1001, 10, 0);
  manager_->SetSWABlockContentFixed(10);
  manager_->SetReclaimable(1001, 10);

  // FA block 20: reused once → hit_count becomes 1.
  manager_->AllocateSWABlock(1001, 20, 0);
  manager_->SetSWABlockContentFixed(20);
  manager_->SetReclaimable(1001, 20);
  // Simulate a second request reusing FA block 20 from the prefix cache.
  manager_->ReuseSWABlock(1002, 20);
  // Release req 1002's hold so the block is reclaimable again.
  manager_->SetReclaimable(1002, 20);

  // Pool is now empty.  Allocating a new block must evict the LFU candidate,
  // which is FA block 10 (hit_count=0 < 1).
  int mem_id = manager_->AllocateSWABlock(1003, 30, 0);
  EXPECT_GE(mem_id, 0);

  // FA block 10 should have been evicted.
  EXPECT_FALSE(manager_->IsSWABlockValid(10));
  // FA block 20 should still be valid.
  EXPECT_TRUE(manager_->IsSWABlockValid(20));
}

// ===========================================================================
// FreeSWABlock
// ===========================================================================

// FreeSWABlock returns the memory block to the allocator.
TEST_F(SWACacheManagerTest, FreeSWABlock_ReturnsMemoryToPool) {
  SetUp(/*swa_block_num=*/4);

  manager_->AllocateSWABlock(1001, 1, 0);
  // Release the reference so the block is reclaimable before freeing.
  manager_->SetReclaimable(1001, 1);
  EXPECT_EQ(FreeBlocks(), 3u);

  manager_->FreeSWABlock(1);
  EXPECT_EQ(FreeBlocks(), 4u);

  // Block is no longer tracked.
  EXPECT_FALSE(manager_->IsSWABlockValid(1));
  EXPECT_EQ(manager_->GetSWAMemoryBlockId(1), -1);
}

// FreeSWABlock on a reclaimable content-fixed block removes it from the LFU list.
TEST_F(SWACacheManagerTest, FreeSWABlock_RemovesFromLfuList) {
  SetUp(/*swa_block_num=*/4);

  manager_->AllocateSWABlock(1001, 5, 0);
  manager_->SetSWABlockContentFixed(5);
  manager_->SetReclaimable(1001, 5);

  size_t avail_before = manager_->GetAvailableBlockCount();
  manager_->FreeSWABlock(5);
  // Free pool +1, reclaimable -1 → net change = 0.
  EXPECT_EQ(manager_->GetAvailableBlockCount(), avail_before);
}

// FreeSWABlock on a non-existent fa_block_id is a no-op (no crash).
TEST_F(SWACacheManagerTest, FreeSWABlock_NoopOnMissing) {
  SetUp(/*swa_block_num=*/4);
  EXPECT_NO_FATAL_FAILURE(manager_->FreeSWABlock(999));
}

// ===========================================================================
// GetAvailableBlockCount
// ===========================================================================

// Initially equals the total pool size (all blocks free, none reclaimable).
TEST_F(SWACacheManagerTest, GetAvailableBlockCount_InitiallyEqualsTotalPool) {
  SetUp(/*swa_block_num=*/8);
  EXPECT_EQ(manager_->GetAvailableBlockCount(), 8u);
}

// Decreases by 1 for each non-reclaimable allocated block.
TEST_F(SWACacheManagerTest, GetAvailableBlockCount_DecreasesOnAlloc) {
  SetUp(/*swa_block_num=*/8);

  manager_->AllocateSWABlock(1001, 1, 0);
  EXPECT_EQ(manager_->GetAvailableBlockCount(), 7u);

  manager_->AllocateSWABlock(1001, 2, 0);
  EXPECT_EQ(manager_->GetAvailableBlockCount(), 6u);
}

// Increases by 1 when a block transitions to reclaimable.
TEST_F(SWACacheManagerTest, GetAvailableBlockCount_IncreasesOnReclaimable) {
  SetUp(/*swa_block_num=*/4);

  manager_->AllocateSWABlock(1001, 1, 0);
  manager_->AllocateSWABlock(1001, 2, 0);
  EXPECT_EQ(manager_->GetAvailableBlockCount(), 2u);

  manager_->SetReclaimable(1001, 1);
  EXPECT_EQ(manager_->GetAvailableBlockCount(), 3u);

  manager_->SetReclaimable(1001, 2);
  EXPECT_EQ(manager_->GetAvailableBlockCount(), 4u);
}

// Stays constant when a reclaimable block is evicted to satisfy a new alloc:
// free pool -1 (new alloc), reclaimable -1 (evicted), free pool +1 (evicted
// block returned) → net: reclaimable -1, free pool unchanged → total -1 + 1 = 0.
TEST_F(SWACacheManagerTest, GetAvailableBlockCount_StableAfterEviction) {
  SetUp(/*swa_block_num=*/1);

  manager_->AllocateSWABlock(1001, 10, 0);
  manager_->SetReclaimable(1001, 10);
  size_t avail = manager_->GetAvailableBlockCount();  // free=1, reclaimable=1 → 2? No:
  // free pool = 0 (block 10 is allocated), reclaimable = 1 → total = 1.
  EXPECT_EQ(avail, 1u);

  // Allocate for FA block 20: evicts FA block 10, then allocates.
  manager_->AllocateSWABlock(1002, 20, 0);
  // After eviction: free pool = 0 (block returned then immediately taken), reclaimable = 0.
  EXPECT_EQ(manager_->GetAvailableBlockCount(), 0u);
}

// ===========================================================================
// LFU eviction order: (hit_count ASC, prefix_fa_block_num ASC)
// ===========================================================================

// Three content-fixed reclaimable blocks with different hit_counts.
// Eviction order must be: lowest hit_count first.
TEST_F(SWACacheManagerTest, LfuEviction_ByHitCountAscending) {
  SetUp(/*swa_block_num=*/3);

  // FA block 1: hit_count=2 (reused twice after becoming reclaimable)
  manager_->AllocateSWABlock(1001, 1, 0);
  manager_->SetSWABlockContentFixed(1);
  manager_->SetReclaimable(1001, 1);
  manager_->ReuseSWABlock(1002, 1);
  manager_->SetReclaimable(1002, 1);
  manager_->ReuseSWABlock(1003, 1);
  manager_->SetReclaimable(1003, 1);

  // FA block 2: hit_count=0
  manager_->AllocateSWABlock(1001, 2, 0);
  manager_->SetSWABlockContentFixed(2);
  manager_->SetReclaimable(1001, 2);

  // FA block 3: hit_count=1 (reused once)
  manager_->AllocateSWABlock(1001, 3, 0);
  manager_->SetSWABlockContentFixed(3);
  manager_->SetReclaimable(1001, 3);
  manager_->ReuseSWABlock(1002, 3);
  manager_->SetReclaimable(1002, 3);

  // Pool empty; first eviction should remove FA block 2 (hit_count=0).
  manager_->AllocateSWABlock(1002, 10, 0);
  EXPECT_FALSE(manager_->IsSWABlockValid(2));
  EXPECT_TRUE(manager_->IsSWABlockValid(1));
  EXPECT_TRUE(manager_->IsSWABlockValid(3));

  // Second eviction should remove FA block 3 (hit_count=1).
  manager_->AllocateSWABlock(1002, 11, 0);
  EXPECT_FALSE(manager_->IsSWABlockValid(3));
  EXPECT_TRUE(manager_->IsSWABlockValid(1));

  // Third eviction should remove FA block 1 (hit_count=2).
  manager_->AllocateSWABlock(1002, 12, 0);
  EXPECT_FALSE(manager_->IsSWABlockValid(1));
}

// Two content-fixed reclaimable blocks with equal hit_count but different
// prefix_fa_block_num.  The one with smaller prefix_fa_block_num is evicted first.
TEST_F(SWACacheManagerTest, LfuEviction_ByPrefixFaBlockNumAscendingOnTie) {
  SetUp(/*swa_block_num=*/2);

  // FA block 5: hit_count=1, prefix_fa_block_num=3 (deeper in tree → higher value)
  manager_->AllocateSWABlock(1001, 5, /*prefix_fa_block_num=*/3);
  manager_->SetSWABlockContentFixed(5);
  manager_->SetReclaimable(1001, 5);
  manager_->ReuseSWABlock(1002, 5);
  manager_->SetReclaimable(1002, 5);

  // FA block 6: hit_count=1, prefix_fa_block_num=1 (shallower → lower value)
  manager_->AllocateSWABlock(1001, 6, /*prefix_fa_block_num=*/1);
  manager_->SetSWABlockContentFixed(6);
  manager_->SetReclaimable(1001, 6);
  manager_->ReuseSWABlock(1002, 6);
  manager_->SetReclaimable(1002, 6);

  // Pool empty; eviction should remove FA block 6 (prefix_fa_block_num=1 < 3).
  manager_->AllocateSWABlock(1002, 20, 0);
  EXPECT_FALSE(manager_->IsSWABlockValid(6));
  EXPECT_TRUE(manager_->IsSWABlockValid(5));
}

// Non-content-fixed reclaimable blocks are evicted before content-fixed ones.
TEST_F(SWACacheManagerTest, LfuEviction_NonContentFixedBeforeContentFixedReclaimable) {
  SetUp(/*swa_block_num=*/2);

  // FA block 1: content-fixed, reclaimable, hit_count=0
  manager_->AllocateSWABlock(1001, 1, 0);
  manager_->SetSWABlockContentFixed(1);
  manager_->SetReclaimable(1001, 1);

  // FA block 2: non-content-fixed, reclaimable
  manager_->AllocateSWABlock(1001, 2, 0);
  manager_->SetReclaimable(1001, 2);

  // Pool empty; FA block 2 (non-content-fixed) must be evicted first.
  manager_->AllocateSWABlock(1002, 10, 0);
  EXPECT_FALSE(manager_->IsSWABlockValid(2));
  EXPECT_TRUE(manager_->IsSWABlockValid(1));
}

// ===========================================================================
// State machine: full lifecycle
// ===========================================================================

// Simulate the full lifecycle of a single FA block:
//   AllocateSWABlock (req A) → SetSWABlockContentFixed → SetReclaimable(req A) →
//   ReuseSWABlock(req B) [new request hits prefix cache] → SetReclaimable(req B) [request ends] →
//   FreeSWABlock [FA block evicted from prefix tree]
TEST_F(SWACacheManagerTest, FullLifecycle_SingleBlock) {
  SetUp(/*swa_block_num=*/4);
  const int64_t req_a = 1001;
  const int64_t req_b = 1002;

  // 1. Allocate by req A: ALLOCATED state (req A holds it).
  int mem_id = manager_->AllocateSWABlock(req_a, 1, 0);
  EXPECT_GE(mem_id, 0);
  EXPECT_FALSE(manager_->IsSWABlockValid(1));  // not content-fixed yet

  // 2. Fill complete: CONTENT_FIXED state.
  manager_->SetSWABlockContentFixed(1);
  EXPECT_TRUE(manager_->IsSWABlockValid(1));

  // 3. Request A ends: RECLAIMABLE state (no more holders).
  manager_->SetReclaimable(req_a, 1);
  EXPECT_EQ(manager_->GetAvailableBlockCount(), 4u);  // free=3, reclaimable=1

  // 4. Request B hits prefix cache: IN_USE state (req B holds it, hit_count incremented).
  manager_->ReuseSWABlock(req_b, 1);
  EXPECT_EQ(manager_->GetAvailableBlockCount(), 3u);  // free=3, reclaimable=0

  // 5. Request B ends: RECLAIMABLE again.
  manager_->SetReclaimable(req_b, 1);
  EXPECT_EQ(manager_->GetAvailableBlockCount(), 4u);

  // 6. FA block evicted from prefix tree: FREED.
  manager_->FreeSWABlock(1);
  EXPECT_FALSE(manager_->IsSWABlockValid(1));
  EXPECT_EQ(FreeBlocks(), 4u);  // memory returned to pool
}

// Simulate FA block reuse: a reclaimable FA block is taken from
// reusable_cached_blocks_ and AllocateSWABlock is called again.
// The old SWA block must be freed and a new one allocated.
TEST_F(SWACacheManagerTest, FullLifecycle_FaBlockReuse) {
  SetUp(/*swa_block_num=*/4);
  const int64_t req_a = 1001;
  const int64_t req_b = 1002;

  // First use of FA block 7 by req A.
  manager_->AllocateSWABlock(req_a, 7, 0);
  manager_->SetSWABlockContentFixed(7);
  manager_->SetReclaimable(req_a, 7);
  int old_mem_id = manager_->GetSWAMemoryBlockId(7);
  EXPECT_GE(old_mem_id, 0);

  // FA block 7 is reused (content overwritten) by req B: AllocateSWABlock frees old block first.
  int new_mem_id = manager_->AllocateSWABlock(req_b, 7, 1);
  EXPECT_GE(new_mem_id, 0);

  // New block is not content-fixed yet.
  EXPECT_FALSE(manager_->IsSWABlockValid(7));
  // Free pool: old block returned, new block taken → net unchanged.
  EXPECT_EQ(FreeBlocks(), 3u);
}

// ===========================================================================
// Multi-request sharing: reference counting correctness
// ===========================================================================

// Two requests share the same SWA block.
// The block must NOT become reclaimable until BOTH requests release it.
TEST_F(SWACacheManagerTest, MultiReqSharing_BlockNotReclaimableUntilAllRelease) {
  SetUp(/*swa_block_num=*/4);
  const int64_t req_a = 2001;
  const int64_t req_b = 2002;

  // Req A allocates and fixes the block.
  manager_->AllocateSWABlock(req_a, 1, 0);
  manager_->SetSWABlockContentFixed(1);
  size_t avail_after_alloc = manager_->GetAvailableBlockCount();  // free=3, reclaimable=0

  // Req B also starts using the same block (hits prefix cache).
  manager_->ReuseSWABlock(req_b, 1);
  EXPECT_EQ(manager_->GetAvailableBlockCount(), avail_after_alloc);  // still 3

  // Req A ends: block still held by req B → must NOT enter reclaimable pool.
  manager_->SetReclaimable(req_a, 1);
  EXPECT_EQ(manager_->GetAvailableBlockCount(), avail_after_alloc);  // still 3

  // Req B ends: now no holders → block enters reclaimable pool.
  manager_->SetReclaimable(req_b, 1);
  EXPECT_EQ(manager_->GetAvailableBlockCount(), avail_after_alloc + 1);  // 4
}

// Same req_id calling SetReclaimable twice is idempotent (no double-count).
TEST_F(SWACacheManagerTest, MultiReqSharing_SameReqIdIdempotentRelease) {
  SetUp(/*swa_block_num=*/4);
  const int64_t req_a = 2001;

  manager_->AllocateSWABlock(req_a, 1, 0);
  size_t avail_before = manager_->GetAvailableBlockCount();

  manager_->SetReclaimable(req_a, 1);
  size_t avail_after_first = manager_->GetAvailableBlockCount();
  EXPECT_EQ(avail_after_first, avail_before + 1);

  // Second release with same req_id: no-op (already removed from set).
  manager_->SetReclaimable(req_a, 1);
  EXPECT_EQ(manager_->GetAvailableBlockCount(), avail_after_first);
}

// Three requests share a block; block becomes reclaimable only after all three release.
TEST_F(SWACacheManagerTest, MultiReqSharing_ThreeRequestsSequentialRelease) {
  SetUp(/*swa_block_num=*/4);
  const int64_t req_a = 3001;
  const int64_t req_b = 3002;
  const int64_t req_c = 3003;

  // Req A allocates; req B and req C also acquire the block via prefix cache hit.
  manager_->AllocateSWABlock(req_a, 1, 0);
  manager_->SetSWABlockContentFixed(1);
  manager_->ReuseSWABlock(req_b, 1);
  manager_->ReuseSWABlock(req_c, 1);

  size_t avail_base = manager_->GetAvailableBlockCount();  // free=3, reclaimable=0

  // Release one by one; block must not be reclaimable until the last release.
  manager_->SetReclaimable(req_a, 1);
  EXPECT_EQ(manager_->GetAvailableBlockCount(), avail_base);  // req B and C still hold it

  manager_->SetReclaimable(req_b, 1);
  EXPECT_EQ(manager_->GetAvailableBlockCount(), avail_base);  // req C still holds it

  manager_->SetReclaimable(req_c, 1);
  EXPECT_EQ(manager_->GetAvailableBlockCount(), avail_base + 1);  // now reclaimable
}

// ===========================================================================
// Multiple FA blocks: independent tracking
// ===========================================================================

// Allocating blocks for multiple FA block ids tracks them independently.
TEST_F(SWACacheManagerTest, MultipleBlocks_IndependentTracking) {
  SetUp(/*swa_block_num=*/8);

  for (size_t fa_id = 1; fa_id <= 5; ++fa_id) {
    manager_->AllocateSWABlock(1001, fa_id, fa_id - 1);
    manager_->SetSWABlockContentFixed(fa_id);
  }

  EXPECT_EQ(FreeBlocks(), 3u);

  for (size_t fa_id = 1; fa_id <= 5; ++fa_id) {
    EXPECT_TRUE(manager_->IsSWABlockValid(fa_id));
    EXPECT_GE(manager_->GetSWAMemoryBlockId(fa_id), 0);
  }

  // Release req 1001's hold on FA block 3 before freeing.
  manager_->SetReclaimable(1001, 3);
  manager_->FreeSWABlock(3);
  EXPECT_FALSE(manager_->IsSWABlockValid(3));
  EXPECT_TRUE(manager_->IsSWABlockValid(1));
  EXPECT_TRUE(manager_->IsSWABlockValid(2));
  EXPECT_TRUE(manager_->IsSWABlockValid(4));
  EXPECT_TRUE(manager_->IsSWABlockValid(5));
  EXPECT_EQ(FreeBlocks(), 4u);
}

// Verify the exact arithmetic: free + reclaimable_non_content_fixed + reclaimable_content_fixed.
TEST_F(SWACacheManagerTest, GetAvailableBlockCount_CombinedPools) {
  SetUp(/*swa_block_num=*/6);

  // Allocate 4 blocks; free pool = 2.
  manager_->AllocateSWABlock(1001, 1, 0);
  manager_->AllocateSWABlock(1001, 2, 0);
  manager_->AllocateSWABlock(1001, 3, 0);
  manager_->AllocateSWABlock(1001, 4, 0);

  // FA block 1: content-fixed + reclaimable → reclaimable_content_fixed.
  manager_->SetSWABlockContentFixed(1);
  manager_->SetReclaimable(1001, 1);

  // FA block 2: non-content-fixed + reclaimable → reclaimable_non_content_fixed.
  manager_->SetReclaimable(1001, 2);

  // FA blocks 3 and 4: active (not reclaimable).

  // available = free(2) + reclaimable_content_fixed(1) + reclaimable_non_content_fixed(1) = 4.
  EXPECT_EQ(manager_->GetAvailableBlockCount(), 4u);
}

// ===========================================================================
// GetAvailableBlockCount(reserved_fa_block_ids): excludes reserved reclaimable blocks
// ===========================================================================

// Empty reserved list: result equals the no-arg overload.
TEST_F(SWACacheManagerTest, GetAvailableBlockCount_Reserved_EmptyListEqualsBaseline) {
  SetUp(/*swa_block_num=*/4);

  manager_->AllocateSWABlock(1001, 1, 0);
  manager_->SetSWABlockContentFixed(1);
  manager_->SetReclaimable(1001, 1);

  std::vector<size_t> empty;
  EXPECT_EQ(manager_->GetAvailableBlockCount(empty), manager_->GetAvailableBlockCount());
}

// A reclaimable non-content-fixed block in reserved_fa_block_ids is excluded.
TEST_F(SWACacheManagerTest, GetAvailableBlockCount_Reserved_ExcludesReclaimableNonContentFixed) {
  SetUp(/*swa_block_num=*/4);

  // FA block 10: non-content-fixed, reclaimable.
  manager_->AllocateSWABlock(1001, 10, 0);
  manager_->SetReclaimable(1001, 10);

  // Baseline: free(3) + reclaimable_non_content_fixed(1) = 4.
  EXPECT_EQ(manager_->GetAvailableBlockCount(), 4u);

  // With FA block 10 reserved: the reclaimable block is excluded → 3.
  std::vector<size_t> reserved = {10};
  EXPECT_EQ(manager_->GetAvailableBlockCount(reserved), 3u);
}

// A reclaimable content-fixed block in reserved_fa_block_ids is excluded.
TEST_F(SWACacheManagerTest, GetAvailableBlockCount_Reserved_ExcludesReclaimableContentFixed) {
  SetUp(/*swa_block_num=*/4);

  // FA block 20: content-fixed, reclaimable.
  manager_->AllocateSWABlock(1001, 20, 0);
  manager_->SetSWABlockContentFixed(20);
  manager_->SetReclaimable(1001, 20);

  // Baseline: free(3) + reclaimable_content_fixed(1) = 4.
  EXPECT_EQ(manager_->GetAvailableBlockCount(), 4u);

  // With FA block 20 reserved: the reclaimable block is excluded → 3.
  std::vector<size_t> reserved = {20};
  EXPECT_EQ(manager_->GetAvailableBlockCount(reserved), 3u);
}

// An active (non-reclaimable) block in reserved_fa_block_ids is NOT subtracted
// because it was never counted in the baseline.
TEST_F(SWACacheManagerTest, GetAvailableBlockCount_Reserved_ActiveBlockNotSubtracted) {
  SetUp(/*swa_block_num=*/4);

  // FA block 5: allocated but still active (req 1001 holds it).
  manager_->AllocateSWABlock(1001, 5, 0);

  // Baseline: free(3), reclaimable(0) = 3.
  EXPECT_EQ(manager_->GetAvailableBlockCount(), 3u);

  // FA block 5 is active, not reclaimable → not counted in baseline → no change.
  std::vector<size_t> reserved = {5};
  EXPECT_EQ(manager_->GetAvailableBlockCount(reserved), 3u);
}

// An unknown fa_block_id in reserved_fa_block_ids is silently ignored.
TEST_F(SWACacheManagerTest, GetAvailableBlockCount_Reserved_UnknownIdIgnored) {
  SetUp(/*swa_block_num=*/4);

  std::vector<size_t> reserved = {999};
  EXPECT_EQ(manager_->GetAvailableBlockCount(reserved), manager_->GetAvailableBlockCount());
}

// Multiple reserved blocks: each reclaimable one is subtracted once.
TEST_F(SWACacheManagerTest, GetAvailableBlockCount_Reserved_MultipleReservedBlocks) {
  SetUp(/*swa_block_num=*/6);

  // Allocate 4 blocks; free pool = 2.
  manager_->AllocateSWABlock(1001, 1, 0);
  manager_->AllocateSWABlock(1001, 2, 0);
  manager_->AllocateSWABlock(1001, 3, 0);
  manager_->AllocateSWABlock(1001, 4, 0);

  // FA block 1: content-fixed + reclaimable.
  manager_->SetSWABlockContentFixed(1);
  manager_->SetReclaimable(1001, 1);

  // FA block 2: non-content-fixed + reclaimable.
  manager_->SetReclaimable(1001, 2);

  // FA blocks 3 and 4: active.

  // Baseline: free(2) + reclaimable(2) = 4.
  EXPECT_EQ(manager_->GetAvailableBlockCount(), 4u);

  // Reserve both reclaimable blocks: available = 4 - 2 = 2 (only free pool).
  std::vector<size_t> reserved = {1, 2};
  EXPECT_EQ(manager_->GetAvailableBlockCount(reserved), 2u);

  // Reserve only one reclaimable block: available = 4 - 1 = 3.
  std::vector<size_t> reserved_one = {1};
  EXPECT_EQ(manager_->GetAvailableBlockCount(reserved_one), 3u);
}

// Mirrors the bug scenario: swa_window_fa_block_ids contains reclaimable blocks
// that will be re-acquired by ReuseSWABlock.  The pre-check must use the
// reserved overload to avoid overestimating availability.
//
// Setup: pool of 2 SWA blocks.
//   - FA block 10: content-fixed, reclaimable (in swa_window_fa_block_ids).
//   - FA block 20: content-fixed, reclaimable (in swa_window_fa_block_ids).
//   - swa_block_num = 1 (one new SWA block needed for a new FA block).
//
// Without the fix: GetAvailableBlockCount() = 2 ≥ 1 → pre-check passes.
//   But after ReuseSWABlock(10) and ReuseSWABlock(20), both reclaimable blocks
//   are consumed, leaving 0 blocks for the new allocation → failure.
//
// With the fix: GetAvailableBlockCount({10, 20}) = 0 < 1 → pre-check correctly
//   rejects the allocation before any state is modified.
TEST_F(SWACacheManagerTest, GetAvailableBlockCount_Reserved_BugScenario_SwaWindowBlocks) {
  SetUp(/*swa_block_num=*/2);

  // FA block 10: content-fixed, reclaimable (simulates a block in swa_window_fa_block_ids).
  manager_->AllocateSWABlock(1001, 10, 0);
  manager_->SetSWABlockContentFixed(10);
  manager_->SetReclaimable(1001, 10);

  // FA block 20: content-fixed, reclaimable (simulates another block in swa_window_fa_block_ids).
  manager_->AllocateSWABlock(1001, 20, 1);
  manager_->SetSWABlockContentFixed(20);
  manager_->SetReclaimable(1001, 20);

  // Pool is now exhausted: free(0) + reclaimable(2) = 2.
  EXPECT_EQ(manager_->GetAvailableBlockCount(), 2u);

  // Without the fix: the pre-check would see 2 available and allow swa_block_num=1.
  // With the fix: both reclaimable blocks are reserved → available = 0 < 1 → reject.
  std::vector<size_t> swa_window_fa_block_ids = {10, 20};
  EXPECT_EQ(manager_->GetAvailableBlockCount(swa_window_fa_block_ids), 0u);

  // Verify that if there is one extra free block the check correctly passes.
  // Simulate: one additional free block exists (pool size 3, two allocated+reclaimable).
  SetUp(/*swa_block_num=*/3);

  manager_->AllocateSWABlock(1001, 10, 0);
  manager_->SetSWABlockContentFixed(10);
  manager_->SetReclaimable(1001, 10);

  manager_->AllocateSWABlock(1001, 20, 1);
  manager_->SetSWABlockContentFixed(20);
  manager_->SetReclaimable(1001, 20);

  // free(1) + reclaimable(2) = 3; reserved = {10, 20} → available = 1 ≥ 1 → passes.
  EXPECT_EQ(manager_->GetAvailableBlockCount(swa_window_fa_block_ids), 1u);
}
