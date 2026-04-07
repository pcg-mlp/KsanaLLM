/* Copyright 2026 Tencent Inc.  All rights reserved.
==============================================================================*/

#include "utils/host_memory_pool.h"

#include <fmt/format.h>

#include "device/device_utils.h"
#include "utils/logger.h"

namespace ksana_llm {

// 单例实现（与 NvlsMcastMemory::GetInstance() 风格一致）
HostMemoryPool* HostMemoryPool::GetInstance() {
  static HostMemoryPool instance;
  return &instance;
}

void HostMemoryPool::Initialize(size_t pool_size_gb) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (initialized_.load(std::memory_order_relaxed)) {
    KLLM_LOG_WARNING << "HostMemoryPool already initialized, skipping.";
    return;
  }

  size_t pool_size = 0;
  if (pool_size_gb > 0) {
    pool_size = pool_size_gb * 1024 * 1024 * 1024;
    KLLM_LOG_INFO << fmt::format("HostMemoryPool: using config host_pool_size_gb={} GB", pool_size_gb);
  } else {
    KLLM_LOG_INFO << "HostMemoryPool: host_pool_size_gb is 0, memory pool disabled";
    return;
  }

  pool_size_ = (pool_size + kHostMemoryAlignment - 1) & ~(kHostMemoryAlignment - 1);

  HostAlloc(&pool_base_, pool_size_);

  KLLM_CHECK_WITH_INFO(pool_base_ != nullptr, "HostMemoryPool: failed to allocate host memory pool");

  const uintptr_t base_addr = reinterpret_cast<uintptr_t>(pool_base_);
  if (base_addr % kHostMemoryAlignment != 0) {
    KLLM_LOG_WARNING << fmt::format(
        "HostMemoryPool: pool_base_ is not aligned to {} bytes, this may affect performance.", kHostMemoryAlignment);
  }

  used_size_.store(0, std::memory_order_relaxed);

  free_blocks_.insert({pool_size_, pool_base_});
  free_blocks_by_addr_[pool_base_] = pool_size_;

  initialized_.store(true, std::memory_order_release);

  KLLM_LOG_INFO << fmt::format("HostMemoryPool initialized, size: {} bytes ({} MB)", pool_size_,
                               pool_size_ / 1024 / 1024);
}

HostMemoryPool::~HostMemoryPool() { Destroy(); }

void HostMemoryPool::Destroy() {
  std::lock_guard<std::mutex> lock(mutex_);

  if (!initialized_.load(std::memory_order_relaxed)) {
    return;
  }

  // 检查是否有未释放的内存块
  if (!allocated_blocks_.empty()) {
    KLLM_LOG_WARNING << fmt::format("HostMemoryPool: destroying with {} unreleased blocks, total {} bytes",
                                    allocated_blocks_.size(), used_size_.load());
  }

  if (pool_base_ != nullptr) {
    HostFree(pool_base_);
    pool_base_ = nullptr;
  }

  allocated_blocks_.clear();
  free_blocks_.clear();
  free_blocks_by_addr_.clear();
  pool_size_ = 0;
  used_size_.store(0, std::memory_order_relaxed);
  initialized_.store(false, std::memory_order_release);

  KLLM_LOG_INFO << "HostMemoryPool destroyed.";
}

void* HostMemoryPool::Allocate(size_t size, size_t alignment) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (!initialized_.load(std::memory_order_relaxed)) {
    KLLM_LOG_ERROR << "HostMemoryPool not initialized.";
    return nullptr;
  }

  if (size == 0) {
    return nullptr;
  }

  // 对齐请求大小（只对齐 size，不对齐地址）
  // 因为 cudaHostAlloc 返回的地址已经是页对齐的（4KB），远大于 256 字节
  // 所以只需保证分配的 size 是对齐的，后续分配的地址自然也是对齐的
  size_t aligned_size = (size + alignment - 1) & ~(alignment - 1);

  // Best-fit 查找：使用 multimap 的 lower_bound 直接按 size 查找
  // 这比使用 set<pair> 更简洁，避免了 nullptr 作为第二个元素的问题
  auto it = free_blocks_.lower_bound(aligned_size);
  if (it == free_blocks_.end()) {
    // 内存池空间不足，返回 nullptr
    // 调用方（Tensor::AcquireImpl）会将 location 改为 LOCATION_HOST 并直接分配
    KLLM_LOG_WARNING << fmt::format(
        "HostMemoryPool exhausted, requested: {}, available: {}, will fallback to direct allocation.", aligned_size,
        pool_size_ - used_size_.load());
    return nullptr;
  }

  size_t block_size = it->first;
  void* block_ptr = it->second;

  // 从空闲列表移除
  free_blocks_.erase(it);
  free_blocks_by_addr_.erase(block_ptr);

  // 直接使用 block_ptr，无需再对齐地址
  // 因为初始的 pool_base_ 已经是对齐的，且所有分配的 size 都是对齐的
  // 所以所有空闲块的起始地址都是对齐的

  // 如果有剩余空间，放回空闲列表
  size_t remaining = block_size - aligned_size;
  if (remaining > 0) {
    void* remaining_ptr = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(block_ptr) + aligned_size);
    free_blocks_.insert({remaining, remaining_ptr});
    free_blocks_by_addr_[remaining_ptr] = remaining;
  }

  // 记录分配
  allocated_blocks_[block_ptr] = aligned_size;
  used_size_.fetch_add(aligned_size, std::memory_order_relaxed);

  return block_ptr;
}

void HostMemoryPool::Free(void* ptr) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (!initialized_.load(std::memory_order_relaxed)) {
    return;
  }

  if (ptr == nullptr) {
    return;
  }

  auto it = allocated_blocks_.find(ptr);
  if (it == allocated_blocks_.end()) {
    KLLM_LOG_ERROR << "Attempting to free unknown pointer from HostMemoryPool.";
    return;
  }

  size_t size = it->second;
  allocated_blocks_.erase(it);
  used_size_.fetch_sub(size, std::memory_order_relaxed);

  // 尝试与相邻空闲块合并，减少碎片
  TryCoalesce(ptr, size);
}

// Try to coalesce adjacent free blocks.
void HostMemoryPool::TryCoalesce(void* ptr, size_t size) {
  uintptr_t current_addr = reinterpret_cast<uintptr_t>(ptr);
  uintptr_t current_end = current_addr + size;

  void* merged_ptr = ptr;
  size_t merged_size = size;

  // 尝试与前一个空闲块合并
  // 查找地址小于当前块的最大空闲块
  auto prev_it = free_blocks_by_addr_.lower_bound(ptr);
  if (prev_it != free_blocks_by_addr_.begin()) {
    --prev_it;
    void* prev_ptr = prev_it->first;
    size_t prev_size = prev_it->second;
    uintptr_t prev_addr = reinterpret_cast<uintptr_t>(prev_ptr);
    uintptr_t prev_end = prev_addr + prev_size;

    // 检查前一个块是否与当前块相邻
    if (prev_end == current_addr) {
      // 从空闲列表中移除前一个块
      // 注意：multimap 可能有多个相同 size 的条目，需要找到正确的那个
      auto range = free_blocks_.equal_range(prev_size);
      for (auto range_it = range.first; range_it != range.second; ++range_it) {
        if (range_it->second == prev_ptr) {
          free_blocks_.erase(range_it);
          break;
        }
      }
      free_blocks_by_addr_.erase(prev_it);

      // 合并
      merged_ptr = prev_ptr;
      merged_size += prev_size;
      current_addr = prev_addr;
      current_end = current_addr + merged_size;  // 更新 current_end 用于后续合并

      KLLM_LOG_DEBUG << fmt::format("HostMemoryPool: coalesced with previous block, prev_addr={}, prev_size={}",
                                    prev_ptr, prev_size);
    }
  }

  // 尝试与后一个空闲块合并
  auto next_it = free_blocks_by_addr_.lower_bound(reinterpret_cast<void*>(current_end));
  if (next_it != free_blocks_by_addr_.end()) {
    void* next_ptr = next_it->first;
    size_t next_size = next_it->second;
    uintptr_t next_addr = reinterpret_cast<uintptr_t>(next_ptr);

    // 检查后一个块是否与当前块相邻
    if (current_end == next_addr) {
      // 从空闲列表中移除后一个块
      auto range = free_blocks_.equal_range(next_size);
      for (auto range_it = range.first; range_it != range.second; ++range_it) {
        if (range_it->second == next_ptr) {
          free_blocks_.erase(range_it);
          break;
        }
      }
      free_blocks_by_addr_.erase(next_it);

      // 合并
      merged_size += next_size;

      KLLM_LOG_DEBUG << fmt::format("HostMemoryPool: coalesced with next block, next_addr={}, next_size={}", next_ptr,
                                    next_size);
    }
  }

  // 将合并后的块加入空闲列表
  free_blocks_.insert({merged_size, merged_ptr});
  free_blocks_by_addr_[merged_ptr] = merged_size;

  if (merged_size > size) {
    KLLM_LOG_DEBUG << fmt::format("HostMemoryPool: coalesced block from {} to {} bytes", size, merged_size);
  }
}

void HostMemoryPool::Reset() {
  std::lock_guard<std::mutex> lock(mutex_);

  if (!initialized_.load(std::memory_order_relaxed)) {
    return;
  }

  allocated_blocks_.clear();
  free_blocks_.clear();
  free_blocks_by_addr_.clear();
  free_blocks_.insert({pool_size_, pool_base_});
  free_blocks_by_addr_[pool_base_] = pool_size_;
  used_size_.store(0, std::memory_order_relaxed);

  KLLM_LOG_INFO << "HostMemoryPool reset.";
}

// Thread-safe getters.
size_t HostMemoryPool::GetPoolSize() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return pool_size_;
}

size_t HostMemoryPool::GetUsedSize() const {
  // used_size_ 是原子变量，可以直接读取
  return used_size_.load(std::memory_order_relaxed);
}

size_t HostMemoryPool::GetFreeSize() const {
  // pool_size_ 在初始化后不变，used_size_ 是原子变量，无需加锁
  return pool_size_ - used_size_.load(std::memory_order_relaxed);
}

size_t HostMemoryPool::GetAllocatedBlockCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return allocated_blocks_.size();
}

size_t HostMemoryPool::GetFreeBlockCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return free_blocks_.size();
}

// Print memory pool stats.
void HostMemoryPool::PrintStats(bool detailed) const {
  std::lock_guard<std::mutex> lock(mutex_);

  if (!initialized_.load(std::memory_order_relaxed)) {
    KLLM_LOG_INFO << "HostMemoryPool: not initialized";
    return;
  }

  size_t used = used_size_.load(std::memory_order_relaxed);
  double usage_percent = pool_size_ > 0 ? (100.0 * used / pool_size_) : 0.0;

  if (!detailed) {
    KLLM_LOG_INFO << fmt::format(
        "HostMemoryPool stats: pool_size={} ({} MB), used={} ({} MB), free={} ({} MB), usage={:.1f}%, "
        "allocated_blocks={}, free_blocks={}",
        pool_size_, pool_size_ / 1024 / 1024, used, used / 1024 / 1024, pool_size_ - used,
        (pool_size_ - used) / 1024 / 1024, usage_percent, allocated_blocks_.size(), free_blocks_.size());
  } else {
    // 计算碎片率：1 - (最大空闲块 / 总空闲空间)
    size_t largest_free_block = 0;
    for (const auto& [size, ptr] : free_blocks_) {
      if (size > largest_free_block) {
        largest_free_block = size;
      }
    }
    double fragmentation =
        (pool_size_ - used) > 0 ? (1.0 - static_cast<double>(largest_free_block) / (pool_size_ - used)) * 100.0 : 0.0;

    KLLM_LOG_INFO << fmt::format(
        "HostMemoryPool detailed stats:\n"
        "  - pool_size: {} ({} MB)\n"
        "  - used: {} ({} MB)\n"
        "  - free: {} ({} MB)\n"
        "  - usage: {:.1f}%\n"
        "  - allocated_blocks: {}\n"
        "  - free_blocks: {}\n"
        "  - largest_free_block: {} ({} MB)\n"
        "  - fragmentation: {:.1f}%",
        pool_size_, pool_size_ / 1024 / 1024, used, used / 1024 / 1024, pool_size_ - used,
        (pool_size_ - used) / 1024 / 1024, usage_percent, allocated_blocks_.size(), free_blocks_.size(),
        largest_free_block, largest_free_block / 1024 / 1024, fragmentation);
  }
}

}  // namespace ksana_llm