/* Copyright 2026 Tencent Inc. All rights reserved.

==============================================================================*/
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>

#include "cache_manager/block_allocator/block_allocator_manager.h"
#include "utils/status.h"

namespace ksana_llm {

// Describes a single WORKING Mamba state block.
struct MambaCachedBlock {
  // The Mamba device memory block ID (range [1, N]). -1 means no memory block is allocated.
  int memory_block_id = -1;
  // WORKING-block owner.
  int64_t owner_req_id = -1;
};

// Manages the WORKING Mamba state blocks (one per running request). Allocator
// block ids are translated to Mamba slot ids so slot 0 remains the NULL sentinel.
// Mamba + prefix-cache is rejected by CacheManagerFactory; working-block
// management only.
class MambaCacheManager {
 public:
  explicit MambaCacheManager(std::shared_ptr<BlockAllocatorGroupInterface> mamba_block_allocator_group);
  MambaCacheManager(const MambaCacheManager&) = delete;
  MambaCacheManager& operator=(const MambaCacheManager&) = delete;

  // ---- Request WORKING block ----

  // Allocate a block for req_id (or return its existing block) and write the
  // working block id to mamba_kv_cache_block_id (-1 on failure).
  Status AllocateRequestMambaBlock(int64_t req_id, int& mamba_kv_cache_block_id);

  // Return req_id's block to the allocator. Unknown requests are ignored.
  Status FreeRequestMambaBlock(int64_t req_id);

  // The WORKING block id bound to req_id, or -1 when req_id has no block.
  int GetRequestMambaBlockId(int64_t req_id) const;

 private:
  Status AllocatePhysicalBlock(int& mamba_kv_cache_block_id);
  Status FreePhysicalBlock(int mamba_kv_cache_block_id);

  std::shared_ptr<BlockAllocatorGroupInterface> mamba_block_allocator_group_;

  // req_id → WORKING block.
  std::unordered_map<int64_t, std::unique_ptr<MambaCachedBlock>> req_to_mamba_block_;
};

}  // namespace ksana_llm
