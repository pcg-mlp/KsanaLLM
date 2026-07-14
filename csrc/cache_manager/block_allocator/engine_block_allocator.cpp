/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "cache_manager/block_allocator/engine_block_allocator.h"

#include "fmt/core.h"
#include "utils/logger.h"
#include "utils/string_utils.h"

namespace ksana_llm {

EngineBlockAllocator::EngineBlockAllocator(size_t block_num) : block_num_(block_num) {}

EngineBlockAllocator::~EngineBlockAllocator() { Clear(); }

void EngineBlockAllocator::PreAllocateBlocks() {
  free_blocks_.reserve(block_num_);
  for (size_t block_id = 0; block_id < block_num_; ++block_id) {
    free_blocks_.insert(block_id);
  }
}

void EngineBlockAllocator::Clear() {
  std::unique_lock<std::mutex> lock(mutex_);
  free_blocks_.clear();
  used_blocks_.clear();
}

Status EngineBlockAllocator::AllocateBlocks(size_t block_num, std::vector<int>& blocks) {
  std::unique_lock<std::mutex> lock(mutex_);

  if (block_num > free_blocks_.size()) {
    return Status(RET_DEVICE_MEM_ALLOCATE_FAILED,
                  FormatStr("No more free blocks, expect %d, free %d", block_num, free_blocks_.size()));
  }

  blocks.clear();
  blocks.reserve(block_num);
  auto it = free_blocks_.begin();
  while (block_num--) {
    used_blocks_.insert(*it);
    blocks.push_back(*it);
    it = free_blocks_.erase(it);
  }
  return Status();
}

Status EngineBlockAllocator::FreeBlocks(const std::vector<int>& blocks) {
  std::unique_lock<std::mutex> lock(mutex_);

  for (auto block_id : blocks) {
    auto it = used_blocks_.find(block_id);
    if (it != used_blocks_.end()) {
      free_blocks_.insert(*it);
      used_blocks_.erase(it);
    } else {
      return Status(RET_DEVICE_MEM_FREE_FAILED, fmt::format("Double free error, block id {}", block_id));
    }
  }
  return Status();
}

Status EngineBlockAllocator::GetBlockPtrs(const std::vector<int>& blocks, std::vector<void*>& addrs) {
  KLLM_THROW("EngineBlockAllocator::GetBlockPtrs is not implemented.");
}

void* EngineBlockAllocator::GetBlocksBasePtr() {
  KLLM_THROW("EngineBlockAllocator::GetBlocksBasePtr is not implemented.");
}

int EngineBlockAllocator::GetBlocksBaseId() { KLLM_THROW("EngineBlockAllocator::GetBlocksBaseId is not implemented."); }

size_t EngineBlockAllocator::GetFreeBlockNumber() {
  std::unique_lock<std::mutex> lock(mutex_);
  return free_blocks_.size();
}

size_t EngineBlockAllocator::GetUsedBlockNumber() {
  std::unique_lock<std::mutex> lock(mutex_);
  return used_blocks_.size();
}

}  // namespace ksana_llm
