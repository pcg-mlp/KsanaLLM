/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "cache_manager/block_allocator/executor_block_allocator.h"

#include "configure/context.h"
#include "device/device_types.h"
#include "device/device_utils.h"
#include "fmt/core.h"
#include "utils/logger.h"
#include "utils/string_utils.h"

namespace ksana_llm {

ExecutorBlockAllocator::ExecutorBlockAllocator(MemoryLocation location, size_t block_num, size_t block_size, int rank,
                                               std::shared_ptr<MemoryAllocatorInterface> memory_allocator,
                                               std::shared_ptr<Context> context)
    : location_(location),
      block_num_(block_num),
      block_size_(block_size),
      rank_(rank),
      memory_allocator_(memory_allocator),
      context_(context) {
  using namespace std::placeholders;
  if (location_ == MemoryLocation::LOCATION_HOST) {
    malloc_fn_ = std::bind(&MemoryAllocatorInterface::HostAlloc, memory_allocator.get(), _1, _2);
    free_fn_ = std::bind(&MemoryAllocatorInterface::HostFree, memory_allocator.get(), _1);
  } else if (location_ == MemoryLocation::LOCATION_DEVICE) {
    malloc_fn_ = std::bind(&MemoryAllocatorInterface::MallocAsync, memory_allocator.get(), _1, _2,
                           context_->device->GetMemoryManageStream());
    free_fn_ = std::bind(&MemoryAllocatorInterface::FreeAsync, memory_allocator.get(), _1,
                         context_->device->GetMemoryManageStream());
  } else {
    KLLM_THROW("The MemoryLocation is not supported.");
  }
}

ExecutorBlockAllocator::~ExecutorBlockAllocator() { Clear(); }

void ExecutorBlockAllocator::PreAllocateBlocks() {
  bool use_continuous_memory = false;
  void* base_mem_ptr = nullptr;

  if (location_ == MemoryLocation::LOCATION_DEVICE) {
    SetDevice(rank_);
  }

#if defined(ENABLE_ACL) || defined(ENABLE_FLASH_ATTN_WITH_CACHE)
  if (location_ == MemoryLocation::LOCATION_DEVICE) {
    use_continuous_memory = true;
    size_t dev_bytes = (block_num_ + 1) * block_size_;
    malloc_fn_(reinterpret_cast<void**>(&base_mem_ptr), dev_bytes);
    blocks_base_ptr_ = base_mem_ptr;
  }
#endif

  // NOTE: Make sure block ids on all worker nodes have same id range.
  allocated_blocks_.reserve(block_num_);
  void* memory_ptr = nullptr;
  for (size_t block_id = 0; block_id < block_num_; ++block_id) {
    if (use_continuous_memory) {
      memory_ptr = reinterpret_cast<uint8_t*>(base_mem_ptr) + block_id * block_size_;
    } else {
      malloc_fn_(&memory_ptr, block_size_);
    }
    allocated_blocks_.emplace(block_id, memory_ptr);
  }
}

void ExecutorBlockAllocator::Clear() {
  if (location_ == MemoryLocation::LOCATION_DEVICE) {
    SetDevice(rank_);
  }

#if defined(ENABLE_ACL_ATB) || defined(ENABLE_FLASH_ATTN_WITH_CACHE)
  if (location_ == MemoryLocation::LOCATION_DEVICE) {
    std::unique_lock<std::mutex> lock(mutex_);

    if (blocks_base_ptr_ != nullptr) {
      free_fn_(blocks_base_ptr_);
      blocks_base_ptr_ = nullptr;
    }
    allocated_blocks_.clear();
    return;
  }
#endif

  {
    auto clear_fn = [&](std::unordered_map<int, void*>& blocks) -> void {
      for (auto it = blocks.begin(); it != blocks.end();) {
        free_fn_(it->second);
        it = blocks.erase(it);
      }
    };

    std::unique_lock<std::mutex> lock(mutex_);
    clear_fn(allocated_blocks_);
  }
}

Status ExecutorBlockAllocator::AllocateBlocks(size_t block_num, std::vector<int>& blocks) {
  KLLM_THROW("ExecutorBlockAllocator::AllocateBlocks is not implemented.");
}

Status ExecutorBlockAllocator::FreeBlocks(const std::vector<int>& blocks) {
  KLLM_THROW("ExecutorBlockAllocator::FreeBlocks is not implemented.");
}

Status ExecutorBlockAllocator::GetBlockPtrs(const std::vector<int>& blocks, std::vector<void*>& addrs) {
  std::unique_lock<std::mutex> lock(mutex_);

  addrs.clear();
  addrs.reserve(blocks.size());
  for (auto block_id : blocks) {
    auto it = allocated_blocks_.find(block_id);
    if (it != allocated_blocks_.end()) {
      addrs.push_back(it->second);
      continue;
    }

    KLLM_LOG_ERROR << "Get block id " << block_id << " address error on device " << rank_;
    return Status(RET_SEGMENT_FAULT, FormatStr("Get block address error, block id {}", block_id));
  }
  return Status();
}

void* ExecutorBlockAllocator::GetBlocksBasePtr() { return blocks_base_ptr_; }

int ExecutorBlockAllocator::GetBlocksBaseId() { return blocks_base_id_; }

size_t ExecutorBlockAllocator::GetFreeBlockNumber() {
  KLLM_THROW("ExecutorBlockAllocator::GetFreeBlockNumber is not implemented.");
}

size_t ExecutorBlockAllocator::GetUsedBlockNumber() {
  KLLM_THROW("ExecutorBlockAllocator::GetUsedBlockNumber is not implemented.");
}

}  // namespace ksana_llm
