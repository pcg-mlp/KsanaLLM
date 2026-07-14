/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <mutex>
#include <unordered_set>

#include "cache_manager/block_allocator/block_allocator_interface.h"

namespace ksana_llm {

class EngineBlockAllocator : public BlockAllocatorInterface {
 public:
  explicit EngineBlockAllocator(size_t block_num);

  virtual ~EngineBlockAllocator();

  virtual void PreAllocateBlocks() override;

  // Free all blocks.
  virtual void Clear() override;

  // Allocate blocked memory.
  virtual Status AllocateBlocks(size_t block_num, std::vector<int>& blocks) override;

  // Free blocked memory.
  virtual Status FreeBlocks(const std::vector<int>& blocks) override;

  // Get memory address of blocked memory.
  virtual Status GetBlockPtrs(const std::vector<int>& blocks, std::vector<void*>& addrs) override;

  // Used for ATB mode, all blocks is part of a whole flatten memory space
  virtual void* GetBlocksBasePtr() override;

  // Used for ATB mode, return the first allocated block id
  virtual int GetBlocksBaseId() override;

  // Get number of free blocked memory.
  virtual size_t GetFreeBlockNumber() override;

  // Get number of used blocked memory.
  virtual size_t GetUsedBlockNumber() override;

 private:
  size_t block_num_;

  // block id to address
  std::unordered_set<int> free_blocks_;
  std::unordered_set<int> used_blocks_;

  // Make thread-safe.
  std::mutex mutex_;
};

}  // namespace ksana_llm
