/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/block_manager/block_allocator.h"
#include "numerous_llm/utils/memory_utils.h"

namespace numerous_llm {

BlockAllocator::BlockAllocator(const AllocatorConfig &allocator_config) {
  allocator_config_ = allocator_config;
  // 定义一个 void 指针，用于存储内存分配的结果
  void* memory;

  // 为每个 CPU 内存块分配内存
  for (int64_t i = 0; i< allocator_config.blocks_num; ++i) {
    switch (allocator_config_.device){
        case MEMORY_CPU_PINNED:
          CUDA_CHECK(cudaHostAlloc(&memory, allocator_config_.block_size, cudaHostAllocDefault));
          break;

        case MEMORY_GPU:
          CUDA_CHECK(cudaMalloc(&memory, allocator_config_.block_size));
          break;
        
        default:
          break;
    }
    int64_t block_id = Singleton<UniqueIDGenerator>::GetInstance()->GetUniqueID();
    free_map_.insert({block_id, {block_id, allocator_config_.block_size, 1 , allocator_config_.device, memory}});
  }
}

BlockAllocator::~BlockAllocator() {
  switch (allocator_config_.device){
    case MEMORY_CPU_PINNED:
      for (auto& block_pair : free_map_) {
          CUDA_CHECK(cudaFreeHost(block_pair.second.address));
      }
      for (auto& block_pair : used_map_) {
          CUDA_CHECK(cudaFreeHost(block_pair.second.address));
      }
      break;

    case MEMORY_GPU:
      for (auto& block_pair : free_map_) {
          CUDA_CHECK(cudaFree(block_pair.second.address));
      }
      for (auto& block_pair : used_map_) {
          CUDA_CHECK(cudaFree(block_pair.second.address));
      }
      break;
    
    default:
      break;
  }
}

// 根据给定的block_ids，获取对应的内存指针，存储在addrs中
Status BlockAllocator::GetBlockPtrs(const std::vector<int>& blocks, std::vector<void*>& addrs) {
  addrs.clear();
  for (auto block_id : blocks) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      auto it = used_map_.find(block_id);
      if (it != used_map_.end()) {
        addrs.push_back(it->second.address);
        continue;
      }
    }
    {
      std::unique_lock<std::mutex> lock(contiguous_memory_mutex_);
      auto it = used_contiguous_memory_map_.find(block_id);
      if (it != used_contiguous_memory_map_.end()) {
        addrs.push_back(it->second.address);
        continue;
      }
    }
    return Status(RET_SEGMENT_FAULT);
  }
  return Status();
}

Status BlockAllocator::Allocate(int64_t block_num, std::vector<int>& blocks) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (block_num > free_map_.size()) {
      return Status(RET_ALLOCATE_FAIL);
  }
  blocks.clear();
  auto it = free_map_.begin();
  while (block_num--) {
      if(it != free_map_.end()) {
          it->second.ref_count++;
          used_map_.insert(*it);
          blocks.push_back(it->first);
          it = free_map_.erase(it);
      }
  }
  return Status();
}

Status BlockAllocator::Free(std::vector<int>& blocks) {
  std::unique_lock<std::mutex> lock(mutex_);
  for (auto block_id : blocks) {
      auto it = used_map_.find(block_id);
      if (it != used_map_.end()) {
          if (--it->second.ref_count == 1) {
              free_map_.insert(*it);
              used_map_.erase(it);
          }
      } else {
          return Status(RET_FREE_FAIL, fmt::format("Double free error, block id {}" , block_id));
      }
  }
  return Status();
}

// 分配指定大小的设备存储空间
Status BlockAllocator::AllocateContiguous(int64_t size, int& block_id) {
  std::unique_lock<std::mutex> lock(contiguous_memory_mutex_);
  // 定义一个 void 指针，用于存储内存分配的结果
  void* memory;
  CUDA_CHECK(cudaMalloc(&memory, size));
  block_id = Singleton<UniqueIDGenerator>::GetInstance()->GetUniqueID();
  used_contiguous_memory_map_.insert({block_id, {block_id, size, 1, MEMORY_GPU, memory}});
  return Status();
}

// 释放连续设备存储
Status BlockAllocator::FreeContiguous(int block_id) {
  std::unique_lock<std::mutex> lock(contiguous_memory_mutex_);
  auto it = used_contiguous_memory_map_.find(block_id);
  if (it != used_contiguous_memory_map_.end()) {
    if (--it->second.ref_count == 0) {
      CUDA_CHECK(cudaFree(it->second.address));
      used_contiguous_memory_map_.erase(it);
      Singleton<UniqueIDGenerator>::GetInstance()->RecycleID(block_id);
    }
  } else {
    return Status(RET_FREE_FAIL, fmt::format("Double free error, block id {}", block_id));
  }

  return Status();
}

}  // namespace numerous_llm
