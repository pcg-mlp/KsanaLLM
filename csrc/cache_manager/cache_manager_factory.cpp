/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "cache_manager/cache_manager_factory.h"

#include "cache_manager/direct_cache_manager.h"
#include "cache_manager/prefix_cache_manager.h"
#include "utils/logger.h"

namespace ksana_llm {

std::shared_ptr<CacheManagerInterface> CacheManagerFactory::CreateCacheManager(
    const CacheManagerConfig& cache_manager_config, std::shared_ptr<BlockAllocatorGroupInterface> block_allocator_group,
    std::shared_ptr<BlockAllocatorGroupInterface> swa_block_allocator_group,
    std::shared_ptr<BlockAllocatorGroupInterface> mamba_block_allocator_group) {
  // TODO(lioshen): delete this when support mamba cache with prefix caching
  if (cache_manager_config.enable_prefix_caching && mamba_block_allocator_group != nullptr) {
    KLLM_THROW("Mamba cache does not support prefix caching.");
  }

  if (cache_manager_config.enable_prefix_caching) {
    return std::make_shared<PrefixCacheManager>(cache_manager_config, block_allocator_group, swa_block_allocator_group);
  } else {
    return std::make_shared<DirectCacheManager>(cache_manager_config, block_allocator_group, swa_block_allocator_group,
                                                mamba_block_allocator_group);
  }
}

}  // namespace ksana_llm
