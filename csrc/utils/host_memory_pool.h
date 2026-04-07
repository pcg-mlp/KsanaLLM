/* Copyright 2026 Tencent Inc.  All rights reserved.
==============================================================================*/
#pragma once

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <mutex>
#include <stdexcept>

namespace ksana_llm {

constexpr size_t kHostMemoryAlignment = 256;

constexpr size_t kDefaultHostPoolSize = 0;

// Host memory pool allocated by cudaHostAlloc/HostAlloc once.
// Notes:
// - cudaHostAlloc usually returns page-aligned address (4KB), which is larger than 256 bytes.
// - We only align the requested size to kHostMemoryAlignment, and do not realign the address.
// - Use std::multimap for best-fit allocation.
class HostMemoryPool {
 public:
  static HostMemoryPool* GetInstance();

  void Initialize(size_t pool_size_gb = 0);

  bool IsInitialized() const { return initialized_.load(std::memory_order_acquire); }

  void* Allocate(size_t size, size_t alignment = kHostMemoryAlignment);

  void Free(void* ptr);

  void Reset();

  void Destroy();

  size_t GetPoolSize() const;
  size_t GetUsedSize() const;
  size_t GetFreeSize() const;
  size_t GetAllocatedBlockCount() const;
  size_t GetFreeBlockCount() const;

  // 打印统计信息，detailed=true 时包含碎片率等详细信息
  void PrintStats(bool detailed = false) const;

 private:
  // 私有构造函数（Singleton 模式）
  HostMemoryPool() = default;
  ~HostMemoryPool();

  // 禁止拷贝
  HostMemoryPool(const HostMemoryPool&) = delete;
  HostMemoryPool& operator=(const HostMemoryPool&) = delete;

 private:
  std::atomic<bool> initialized_{false};
  void* pool_base_ = nullptr;
  size_t pool_size_ = 0;
  std::atomic<size_t> used_size_{0};

  mutable std::mutex mutex_;

  std::map<void*, size_t> allocated_blocks_;

  std::multimap<size_t, void*> free_blocks_;

  std::map<void*, size_t> free_blocks_by_addr_;

  void TryCoalesce(void* ptr, size_t size);
};

}  // namespace ksana_llm