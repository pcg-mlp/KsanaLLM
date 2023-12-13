/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cuda_runtime.h>
#include <unordered_map>
#include <vector>

#include "numerous_llm/block_manager/memory_block.h"
#include "numerous_llm/block_manager/block_allocator.h"
#include "numerous_llm/utils/environment.h"
#include "numerous_llm/utils/status.h"
#include "numerous_llm/utils/nvidia/cuda_utils.h"


namespace numerous_llm {
// 类：BlockManager
// 用于管理显存、内存块的分配、回收、迁移等操作
class BlockManager {
public:
  // 构造函数
  BlockManager();

  // 构造函数
  // 参数：block_manager_config - BlockManager 的配置信息
  BlockManager(const BlockManagerConfig &block_manager_config);

  // 析构函数
  ~BlockManager();

  // 获取配置
  BlockManagerConfig GetBlockManagerConfig();

  // 获取显存块的指针
  // 参数：blocks - 显存块 ID 列表
  // 参数：addrs - 返回的显存块指针列表
  // 返回值：Status 对象，表示操作的成功或失败
  Status GetGpuBlockPtrs(const std::vector<int>& blocks, std::vector<void*>& addrs);

  // 分配显存块
  // 参数：block_num - 要分配的显存块数量
  // 参数：blocks - 返回的显存块 ID 列表
  // 返回值：Status 对象，表示操作的成功或失败
  Status AllocateGpuBlocks(int64_t block_num, std::vector<int>& blocks);

  // 分配指定大小的显存空间
  // 参数：gpu_memory - 返回分配的显存空间指针
  // 参数：size - 要分配的显存空间大小
  // 返回值：Status 对象，表示操作的成功或失败
  Status Allocate(void*& gpu_memory, int64_t size);

  // 释放显存块
  // 参数：blocks - 要释放的显存块 ID 列表
  // 返回值：Status 对象，表示操作的成功或失败
  Status FreeGpuBlocks(std::vector<int>& blocks);

  // 释放连续显存
  // 参数：gpu_memory - 分配的显存空间指针
  // 返回值：Status 对象，表示操作的成功或失败
  Status Free(void* gpu_memory);

  // 将显存块从 GPU 交换到 CPU
  // 参数：gpu_blocks - 要交换的显存块 ID 列表
  // 参数：stream - 用于执行交换操作的 CUDA 流
  // 返回值：Status 对象，表示操作的成功或失败
  Status SwapGpuToCpu(std::vector<int>& gpu_blocks, cudaStream_t stream);

  // 将数据块从 CPU 交换到 GPU
  // 参数：gpu_blocks - 要交换的显存块 ID 列表
  // 参数：stream - 用于执行交换操作的 CUDA 流
  // 返回值：Status 对象，表示操作的成功或失败
  Status SwapCpuToGpu(std::vector<int>& gpu_blocks, cudaStream_t stream);

  // 获取指定设备类型的空闲块数量
  // 参数：device - 设备类型（CPU 或 GPU）
  // 返回值：空闲块数量
  int64_t GetFreeBlockNumber(MemoryDevice device);

private:
  std::mutex contiguous_memory_mutex_;
  std::mutex swap_mutex_;
  BlockAllocator gpu_allocator;
  BlockAllocator cpu_allocator;
  std::unordered_map<void*, int64_t> used_contiguous_memory_map_;
  std::unordered_map<int64_t, int64_t> swap_map_;
  int64_t block_size_;
};
} // namespace numerous_llm
