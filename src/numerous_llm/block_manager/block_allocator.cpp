/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/block_manager/block_allocator.h"

namespace numerous_llm {

BlockAllocator::BlockAllocator(const AllocatorConfig &allocator_config) {}

Status BlockAllocator::Allocate(PhysicalBlock &block) {
  if (free_blocks_.empty()) {
    return Status(RET_OUT_OF_MEMORY,
                  "Out of memory, no free blocks available.");
  }

  block = free_blocks_.back();
  free_blocks_.pop_back();
  block.ref_count = 1;
  return Status();
}

Status BlockAllocator::Free(PhysicalBlock &block) {
  if (block.ref_count == 0) {
    return Status(RET_SEGMENT_FAULT,
                  "Double free error, block id " + block.block_index);
  }

  block.ref_count -= 1;
  if (block.ref_count == 0) {
    free_blocks_.push_back(block);
  }
  return Status();
}

} // namespace numerous_llm
