/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <algorithm>
#include <memory>
#include <queue>
#include <vector>

#include "ksana_llm/block_manager/block_manager_interface.h"

namespace ksana_llm {

static int64_t DivRoundUp(int64_t dividend, int64_t divisor) { return (dividend + divisor - 1) / divisor; }

static int64_t DivRoundDown(int64_t dividend, int64_t divisor) { return dividend / divisor; }

// Set a global block manager
void SetBlockManager(BlockManagerInterface* block_manager);

// Get the global block manager
BlockManagerInterface* GetBlockManager();

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

// Get host block pointer.
template <typename T>
T* GetHostContiguousPtr(int block_id) {
  void* addr;
  GetBlockManager()->GetHostContiguousPtr(block_id, addr);
  return reinterpret_cast<T*>(addr);
}

// Get free & total memory in bytes of current selected device.
Status GetDeviceMemoryInfo(MemoryDevice device, size_t* free, size_t* total);

// Get free & total host memory in bytes.
Status GetHostMemoryInfo(size_t* free, size_t* total);

// Get workspace of size.
// It maintain a global memory block, and reallocated if size is not enough.
void GetWorkSpaceImpl(size_t size, void** ws_addr);

// Define a function to create kernel workspace.
typedef void (*WorkSpaceFunc)(size_t, void**);

// Get the workspace function.
WorkSpaceFunc GetWorkSpaceFunc();

}  // namespace ksana_llm
