/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "cache_manager/cache_manager_interface.h"
#include "configure/environment.h"

namespace ksana_llm {

// Create CacheManager instance.
class CacheManagerFactory {
 public:
  // Create a cache manager instance.
  // swa_block_allocator_group is optional; when non-null and sliding_window_size > 0,
  // DirectCacheManager will use it for SWA KV cache block allocation.
  // mamba_block_allocator_group is the Mamba/GDN pool, optional.
  static std::shared_ptr<CacheManagerInterface> CreateCacheManager(
      const CacheManagerConfig& cache_manager_config,
      std::shared_ptr<BlockAllocatorGroupInterface> block_allocator_group,
      std::shared_ptr<BlockAllocatorGroupInterface> swa_block_allocator_group = nullptr,
      std::shared_ptr<BlockAllocatorGroupInterface> mamba_block_allocator_group = nullptr);
};

}  // namespace ksana_llm
