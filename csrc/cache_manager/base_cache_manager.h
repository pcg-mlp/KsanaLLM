/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "cache_manager/block_allocator/block_allocator_manager.h"
#include "cache_manager/cache_manager_interface.h"
#include "configure/environment.h"
#include "utils/logger.h"
#include "utils/memory_utils.h"
#include "utils/status.h"
#include "utils/string_utils.h"

namespace ksana_llm {

template <class CachedBlockType, class CachedRequestType>
class BaseCacheManager {
 public:
  explicit BaseCacheManager(const CacheManagerConfig& cache_manager_config,
                            std::shared_ptr<BlockAllocatorGroupInterface> block_allocator_group) {
    cache_manager_config_ = cache_manager_config;
    block_allocator_group_ = block_allocator_group;
    block_device_num_ = block_allocator_group_->GetBlockAllocatorDevices().size();
  }

  ~BaseCacheManager() = default;

  std::shared_ptr<BlockAllocatorGroupInterface> GetBlockAllocatorGroup() const { return block_allocator_group_; }

  // The value is from block mamanger.
  size_t GetHostFreeBlockNumber() { return block_allocator_group_->GetHostBlockAllocator()->GetFreeBlockNumber(); }

  size_t GetDeviceUsedBlockNumber() { return block_allocator_group_->GetDeviceBlockAllocator()->GetUsedBlockNumber(); }

 protected:
  // Create a new cached request instance.
  CachedRequestType* CreateCachedRequest(int64_t req_id) {
    // New created request is in waiting state, and have no cached blocks.PrefixCacheManager
    auto cached_request = std::make_unique<CachedRequestType>();
    cached_request->req_id = req_id;

    // Get a raw pointer from the unique pointer of CachedRequestType without ownership.
    CachedRequestType* cached_request_ptr = cached_request.get();
    // Append to request list.
    cached_requests_.emplace(req_id, std::move(cached_request));

    return cached_request_ptr;
  }

  bool IsPrefixCachingEnabled() { return cache_manager_config_.enable_prefix_caching; }

 protected:
  CacheManagerConfig cache_manager_config_;

  // The block allocators for this cache manager instance.
  std::shared_ptr<BlockAllocatorGroupInterface> block_allocator_group_ = nullptr;

  // The devices of current cache manager.
  size_t block_device_num_;

  // The cached blocks that have no computed data, and could be reused freely.
  std::queue<CachedBlockType*> free_cached_blocks_;

  // All requests, the key is request id.
  std::unordered_map<int64_t, std::unique_ptr<CachedRequestType>> cached_requests_;
};

}  // namespace ksana_llm
