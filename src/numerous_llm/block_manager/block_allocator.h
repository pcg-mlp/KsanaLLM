/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cuda_runtime.h>

#include "numerous_llm/block_manager/memory_block.h"
#include "numerous_llm/utils/environment.h"
#include "numerous_llm/utils/nvidia/cuda_utils.h"
#include "numerous_llm/utils/status.h"

namespace numerous_llm {
// The block allocator maintains a list of free blocks, and allocate a block
// when requested. When a block is free, its reference count is decremented. If
// the reference count becomes zero, the block is addes back to the free list.
class BlockAllocator {
 public:
  explicit BlockAllocator(const AllocatorConfig& allocator_config);
  ~BlockAllocator();
  // Aalloc a block from allocator.
  Status Allocate(int64_t block_num, std::vector<int>& blocks);

  // Free a block to allocator.
  Status Free(std::vector<int>& blocks);

  // 分配指定大小的设备存储空间
  // 参数：size - 要分配的设备存储空间大小
  // 参数：block_id - 设备存储块 ID
  // 返回值：Status 对象，表示操作的成功或失败
  Status AllocateContiguous(int64_t size, int& block_id);

  // 释放连续设备存储
  // 参数：block_id - 设备存储块 ID
  // 返回值：Status 对象，表示操作的成功或失败
  Status FreeContiguous(int block_id);

  // 根据给定的block_ids，获取对应的内存指针，存储在addrs中
  static Status GetBlockPtrs(const std::vector<int>& blocks, std::vector<void*>& addrs);

  int64_t GetFreeBlockNumber() { return free_map_.size(); }

 private:
  // 使用静态成员变量, 管理全部 MmemoryBlock 数据块, 实现不区分卡查询 block_ids 对应地址信息功能
  // TODO(zakwang): 需要重新整理内部数据结构及结构间关系;
  //                需要解决类析构时静态成员变量先于类实例被析构, 所引发的内存泄露问题
  static std::mutex mutex_;
  static std::mutex contiguous_memory_mutex_;
  std::unordered_map<int64_t, MemoryBlock> free_map_;
  static std::unordered_map<int64_t, MemoryBlock> used_map_;
  static std::unordered_map<int64_t, MemoryBlock> used_contiguous_memory_map_;
  int block_num_;
  AllocatorConfig allocator_config_;
};

}  // namespace numerous_llm
