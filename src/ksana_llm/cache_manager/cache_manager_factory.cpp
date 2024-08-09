/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/cache_manager/cache_manager_factory.h"

#include "ksana_llm/cache_manager/direct_cache_manager.h"
#include "ksana_llm/cache_manager/prefix_cache_manager.h"

namespace ksana_llm {

std::shared_ptr<CacheManagerInterface> CacheManagerFactory::CreateCacheManager(
    const CacheManagerConfig& cache_manager_config) {
  if (cache_manager_config.enable_preifx_caching) {
    return std::make_shared<PrefixCacheManager>(cache_manager_config);
  } else {
    return std::make_shared<DirectCacheManager>(cache_manager_config);
  }
}

}  // namespace ksana_llm
