/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <algorithm>
#include <memory>
#include <queue>
#include <vector>

#include "numerous_llm/block_manager/block_manager.h"
#include "numerous_llm/utils/singleton.h"

namespace numerous_llm {

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

}  // namespace numerous_llm
