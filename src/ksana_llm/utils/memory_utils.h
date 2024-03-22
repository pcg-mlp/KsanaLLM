/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <algorithm>
#include <memory>
#include <queue>
#include <vector>

#include "ksana_llm/block_manager/block_manager.h"
#include "ksana_llm/utils/singleton.h"

namespace ksana_llm {

// Set a global block manager
void SetBlockManager(BlockManager* block_manager);

// Get the global block manager
BlockManager* GetBlockManager();

// Get block pointer.
template <typename T>
std::vector<T*> GetBlockPtrs(const std::vector<int>& blocks) {
  std::vector<void*> addrs;
  GetBlockManager()->GetBlockPtrs(blocks, addrs);
  std::vector<T*> results(addrs.size());
  std::transform(addrs.begin(), addrs.end(), results.begin(), [](void* p) { return reinterpret_cast<T*>(p); });
  return results;
}

// Get block pointer.
template <typename T>
T* GetContiguousPtr(int block_id) {
  void* addr;
  GetBlockManager()->GetContiguousPtr(block_id, addr);
  return reinterpret_cast<T*>(addr);
}

// Get free & total memory in bytes of current selected device.
Status GetDeviceMemoryInfo(MemoryDevice device, size_t* free, size_t* total);

// Get free & total host memory in bytes.
Status GetHostMemoryInfo(size_t* free, size_t* total);

}  // namespace ksana_llm
