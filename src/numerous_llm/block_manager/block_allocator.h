/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "numerous_llm/block_manager/memory_block.h"
#include "numerous_llm/utils/environment.h"
#include "numerous_llm/utils/status.h"

namespace numerous_llm {

// The block allocator maintains a list of free blocks, and allocate a block
// when requested. When a block is free, its reference count is decremented. If
// the reference count becomes zero, the block is addes back to the free list.
class BlockAllocator {
 public:
  BlockAllocator(const AllocatorConfig &allocator_config);

  // Aalloc a block from allocator.
  Status Allocate(MemoryBlock &block);

  // Free a block to allocator.
  Status Free(MemoryBlock &block);

 private:
  std::vector<MemoryBlock> free_blocks_;
};

}  // namespace numerous_llm
