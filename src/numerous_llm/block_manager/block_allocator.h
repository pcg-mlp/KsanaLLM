/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cuda_runtime.h>

#include "numerous_llm/block_manager/memory_block.h"
#include "numerous_llm/utils/nvidia/cuda_utils.h"
#include "numerous_llm/utils/environment.h"
#include "numerous_llm/utils/status.h"

namespace numerous_llm {

// The block allocator maintains a list of free blocks, and allocate a block
// when requested. When a block is free, its reference count is decremented. If
// the reference count becomes zero, the block is addes back to the free list.
class BlockAllocator {
 public:
  BlockAllocator(const AllocatorConfig &allocator_config);
  ~BlockAllocator();
  // Aalloc a block from allocator.
  Status Allocate(int64_t block_num, std::vector<int>& blocks);

  // Free a block to allocator.
  Status Free(std::vector<int>& blocks);

  // 根据给定的block_ids，获取对应的内存指针，存储在addrs中
  Status GetBlockPtrs(const std::vector<int>& blocks, std::vector<void*>& addrs);

  int64_t GetFreeBlockNumber() {
    return free_map_.size();
  }
 private:
  std::mutex mutex_;
  std::unordered_map<int64_t, MemoryBlock> free_map_;
  std::unordered_map<int64_t, MemoryBlock> used_map_;
  int block_num_;
  AllocatorConfig allocator_config_;
};

}  // namespace numerous_llm
