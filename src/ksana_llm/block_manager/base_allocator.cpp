/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/block_manager/base_allocator.h"
#include "ksana_llm/utils/environment.h"

namespace ksana_llm {

IdGenerator BaseAllocator::id_generator_;

BaseAllocator::BaseAllocator(const AllocatorConfig& allocator_config, std::shared_ptr<Context> context)
    : allocator_config_(allocator_config), context_(context) {}

void BaseAllocator::PreAllocateBlocks() {
  void* memory_ptr;
  for (int64_t i = 0; i < allocator_config_.blocks_num; ++i) {
    AllocateMemory(&memory_ptr, allocator_config_.block_size);

    int block_id = id_generator_.Gen();
    MemoryBlock block = {block_id, allocator_config_.block_size, 0, allocator_config_.device, memory_ptr};
    free_blocks_.insert({block_id, block});
  }
}

Status BaseAllocator::ResetPreAllocatedBlocks(size_t blocks_num) {
  allocator_config_.blocks_num = blocks_num;
  PreAllocateBlocks();
  return Status();
}

Status BaseAllocator::AllocateBlocks(size_t block_num, std::vector<int>& blocks) {
  std::unique_lock<std::mutex> lock(block_mutex_);

  if (block_num > free_blocks_.size()) {
    return Status(RET_ALLOCATE_FAIL, "No more free blocks.");
  }

  blocks.clear();
  blocks.reserve(block_num);
  auto it = free_blocks_.begin();
  while (block_num--) {
    used_blocks_.insert(*it);
    blocks.push_back(it->first);
    it = free_blocks_.erase(it);
  }
  return Status();
}

Status BaseAllocator::FreeBlocks(const std::vector<int>& blocks) {
  std::unique_lock<std::mutex> lock(block_mutex_);

  for (auto block_id : blocks) {
    auto it = used_blocks_.find(block_id);
    if (it != used_blocks_.end()) {
      free_blocks_.insert(*it);
      used_blocks_.erase(it);
    } else {
      return Status(RET_FREE_FAIL, fmt::format("Double free error, block id {}", block_id));
    }
  }
  return Status();
}

Status BaseAllocator::AllocateContiguous(size_t size, int& block_id) {
  std::unique_lock<std::mutex> lock(contiguous_mutex_);

  void* memory_ptr;
  AllocateMemory(&memory_ptr, size);

  block_id = id_generator_.Gen();
  MemoryBlock block = {block_id, allocator_config_.block_size, 0, allocator_config_.device, memory_ptr};
  used_contiguous_.insert({block_id, block});

  return Status();
}

Status BaseAllocator::FreeContiguous(int block_id) {
  std::unique_lock<std::mutex> lock(contiguous_mutex_);

  auto it = used_contiguous_.find(block_id);
  if (it != used_contiguous_.end()) {
    FreeMemory(it->second.address);
    used_contiguous_.erase(it);
  } else {
    return Status(RET_FREE_FAIL, fmt::format("Double free error, block id {}", block_id));
  }

  return Status();
}

Status BaseAllocator::GetBlockPtrs(const std::vector<int>& blocks, std::vector<void*>& addrs) {
  std::unique_lock<std::mutex> lock(block_mutex_);

  addrs.clear();
  for (auto block_id : blocks) {
    auto it = used_blocks_.find(block_id);
    if (it != used_blocks_.end()) {
      addrs.push_back(it->second.address);
      continue;
    }
    NLLM_LOG_ERROR << "Get block id " << block_id << " address error on device " << allocator_config_.device;
    return Status(RET_SEGMENT_FAULT, "Get block address error.");
  }
  return Status();
}

Status BaseAllocator::GetContiguousPtr(int block_id, void*& addr) {
  std::unique_lock<std::mutex> lock(contiguous_mutex_);

  auto it = used_contiguous_.find(block_id);
  if (it != used_contiguous_.end()) {
    addr = it->second.address;
    return Status();
  }
  return Status(RET_SEGMENT_FAULT, "Get contiguous address error.");
}

size_t BaseAllocator::GetFreeBlockNumber() {
  std::unique_lock<std::mutex> lock(block_mutex_);
  return free_blocks_.size();
}

size_t BaseAllocator::GetUsedBlockNumber() {
  std::unique_lock<std::mutex> lock(block_mutex_);
  return used_blocks_.size();
}

}  // namespace ksana_llm
