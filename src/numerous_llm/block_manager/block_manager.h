/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "numerous_llm/block_manager/memory_block.h"
#include "numerous_llm/utils/environment.h"
#include "numerous_llm/utils/status.h"

namespace numerous_llm {

class BlockManager {
 public:
  BlockManager(const BlockManagerConfig& block_manager_config);

  // Get block pointer.
  Status GetBlockPtrs(const std::vector<int>& blocks, std::vector<void*>& addrs);

  Status Allocate(size_t block_size, size_t block_num, std::vector<int>& blocks);

  Status Free(std::vector<int>& blocks);

  Status SwapIn(std::vector<int>& blocks);

  Status SwapOut(std::vector<int>& blocks);

  // Get free block number in specific device.
  size_t GetFreeBlockNumber(MemoryDevice device);
};

}  // namespace numerous_llm
