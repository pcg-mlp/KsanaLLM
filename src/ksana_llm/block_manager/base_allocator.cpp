/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/block_manager/base_allocator.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/string_utils.h"

namespace ksana_llm {

IdGenerator BaseAllocator::id_generator_;

BaseAllocator::BaseAllocator(const AllocatorConfig& allocator_config, std::shared_ptr<Context> context)
    : allocator_config_(allocator_config), context_(context) {}

void BaseAllocator::PreAllocateBlocks() {
  void* memory_ptr = nullptr;
  bool is_continuous_mode = false;
  void* base_mem_ptr = nullptr;
#ifdef ENABLE_ACL_ATB
  if (allocator_config_.device == MEMORY_DEVICE) {
    is_continuous_mode = true;
    // NOTE(karlluo): allocator_config_.block_size shape: 2 x layer_num x block_size x head_dim x head_size x 2 x
    // sizeof(DTYPE)
    AllocateMemory(&base_mem_ptr, allocator_config_.blocks_num * allocator_config_.block_size);
    blocks_base_ptr = base_mem_ptr;
  }
#endif
  for (int64_t i = 0; i < allocator_config_.blocks_num; ++i) {
    int block_id = id_generator_.Gen();
    if (is_continuous_mode) {
      memory_ptr = base_mem_ptr + i * allocator_config_.block_size;
      if (i == 0) {
        block_base_id = block_id;
      }
    } else {
      AllocateMemory(&memory_ptr, allocator_config_.block_size);
    }
    MemoryBlock block = {block_id, allocator_config_.block_size, 0, allocator_config_.device, memory_ptr};
    free_blocks_.insert({block_id, block});
  }
}

Status BaseAllocator::ResetPreAllocatedBlocks(size_t blocks_num) {
  allocator_config_.blocks_num = blocks_num;
  PreAllocateBlocks();
  return Status();
}

bool BaseAllocator::IsContiguousUsed(const int block_id) {
  std::unique_lock<std::mutex> lock(block_mutex_);

  auto it = used_blocks_.find(block_id);
  return it != used_blocks_.end();
}

Status BaseAllocator::AllocateBlocks(size_t block_num, std::vector<int>& blocks) {
  std::unique_lock<std::mutex> lock(block_mutex_);

  if (block_num > free_blocks_.size()) {
    return Status(RET_ALLOCATE_FAIL,
                  FormatStr("No more free blocks, expect %d, free %d", block_num, free_blocks_.size()));
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
  MemoryBlock block = {block_id, size, 0, allocator_config_.device, memory_ptr};
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
    KLLM_LOG_ERROR << "Get block id " << block_id << " address error on device " << allocator_config_.device;
    return Status(RET_SEGMENT_FAULT, FormatStr("Get block address error, block id ", block_id));
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
  return Status(RET_SEGMENT_FAULT, FormatStr("Get contiguous address error, block id %d.", block_id));
}

size_t BaseAllocator::GetFreeBlockNumber() {
  std::unique_lock<std::mutex> lock(block_mutex_);
  return free_blocks_.size();
}

size_t BaseAllocator::GetUsedBlockNumber() {
  std::unique_lock<std::mutex> lock(block_mutex_);
  return used_blocks_.size();
}

void* BaseAllocator::GetBlocksBasePtr() { return blocks_base_ptr; }

const AllocatorConfig& BaseAllocator::GetAllocatorConfig() { return allocator_config_; }

int BaseAllocator::GetBlocksBaseId() {return block_base_id;}

}  // namespace ksana_llm
