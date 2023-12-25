/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <unordered_map>
#include <vector>

#include "3rdparty/LLM_kernels/csrc/kernels/nvidia/paged_attention/dtype_float16.cuh"
#include "numerous_llm/block_manager/block_allocator.h"
#include "numerous_llm/block_manager/memory_block.h"
#include "numerous_llm/runtime/context.h"
#include "numerous_llm/utils/environment.h"
#include "numerous_llm/utils/logger.h"
#include "numerous_llm/utils/nvidia/cuda_utils.h"
#include "numerous_llm/utils/status.h"

namespace numerous_llm {

// 类：DeviceBlockManager
// 用于管理设备存储、内存块的分配、回收、迁移等操作
class DeviceBlockManager {
 public:
  // 构造函数

  explicit DeviceBlockManager(int device_id = 0);

  // 构造函数
  // 参数：block_manager_config - DeviceBlockManager 的配置信息
  explicit DeviceBlockManager(const BlockManagerConfig& block_manager_config, int device_id = 0);

  // 析构函数
  ~DeviceBlockManager();

  // 获取配置
  BlockManagerConfig GetBlockManagerConfig();

  // 获取设备存储块的指针
  // 参数：blocks - 设备存储块 ID 列表
  // 参数：addrs - 返回的设备存储块指针列表
  // 返回值：Status 对象，表示操作的成功或失败
  Status GetBlockPtrs(const std::vector<int>& blocks, std::vector<void*>& addrs);

  // 分配设备存储块
  // 参数：block_num - 要分配的设备存储块数量
  // 参数：blocks - 返回的设备存储块 ID 列表
  // 返回值：Status 对象，表示操作的成功或失败
  Status AllocateBlocks(int64_t block_num, std::vector<int>& blocks);

  // 分配指定大小的设备存储空间
  // 参数：size - 要分配的设备存储空间大小
  // 参数：block_id - 设备存储块 ID
  // 返回值：Status 对象，表示操作的成功或失败
  Status AllocateContiguous(int64_t size, int& block_id);

  // 释放设备存储块
  // 参数：blocks - 要释放的设备存储块 ID 列表
  // 返回值：Status 对象，表示操作的成功或失败
  Status FreeBlocks(const std::vector<int>& blocks);

  // 释放连续设备存储
  // 参数：block_id - 设备存储块 ID
  // 返回值：Status 对象，表示操作的成功或失败
  Status FreeContiguous(int block_id);

  // 将设备存储块换入
  // 参数：device_blocks - 要交换的设备存储块 ID 列表
  // 参数：stream - 用于执行交换操作的 CUDA 流
  // 返回值：Status 对象，表示操作的成功或失败
  Status SwapIn(std::vector<int>& device_blocks, cudaStream_t stream);

  // 将数据块换出
  // 参数：device_blocks - 要交换的设备存储块 ID 列表
  // 参数：stream - 用于执行交换操作的 CUDA 流
  // 返回值：Status 对象，表示操作的成功或失败
  Status SwapOut(std::vector<int>& device_blocks, cudaStream_t stream);

  // 获取指定设备类型的空闲块数量
  // 参数：device - 设备类型
  // 返回值：空闲块数量
  int64_t GetFreeBlockNumber(MemoryDevice device);

  // 获取设备id
  // 返回值：device id
  int GetDeviceId() {
    int device_id;
    CUDA_CHECK(cudaGetDevice(&device_id));
    return device_id;
  }

 private:
  std::mutex swap_mutex_;
  BlockAllocator device_allocator;
  BlockAllocator cpu_allocator;
  std::unordered_map<int64_t, int64_t> swap_map_;
  int64_t block_size_;
  int device_id_;
};

// The block manager used to manager multiple devices.
class BlockManager {
 public:
  BlockManager(const BlockManagerConfig& block_manager_config, std::shared_ptr<Context> context)
      : block_manager_config_(block_manager_config), context_(context) {
    for (int worker_id = 0; worker_id < context_->GetTensorParallelSize(); ++worker_id) {
      dev_block_managers_.push_back(std::make_shared<DeviceBlockManager>(block_manager_config, worker_id));
    }
  }

  ~BlockManager() {}

  // This function maybe called concurrently from different threads.
  // DO NOT store the device id in variable.
  void SetDeviceId(int device_id) { CUDA_CHECK(cudaSetDevice(device_id)); }

  // This function maybe called concurrently from different threads.
  int GetDeviceId() {
    int device_id;
    CUDA_CHECK(cudaGetDevice(&device_id));
    return device_id;
  }

  Status GetBlockPtrs(const std::vector<int>& blocks, std::vector<void*>& addrs) {
    return GetDeviceBlockManager()->GetBlockPtrs(blocks, addrs);
  }

  Status AllocateBlocks(int64_t block_num, std::vector<int>& blocks) {
    return GetDeviceBlockManager()->AllocateBlocks(block_num, blocks);
  }

  Status AllocateContiguous(int64_t size, int& block_id) {
    return GetDeviceBlockManager()->AllocateContiguous(size, block_id);
  }

  Status FreeBlocks(const std::vector<int>& blocks) { return GetDeviceBlockManager()->FreeBlocks(blocks); }

  Status FreeContiguous(int block_id) { return GetDeviceBlockManager()->FreeContiguous(block_id); }

  Status SwapIn(std::vector<int>& device_blocks) {
    int device_id = GetDeviceId();
    return GetDeviceBlockManager(device_id)->SwapIn(device_blocks, context_->h2d_streams_[device_id]);
  }

  Status SwapOut(std::vector<int>& device_blocks) {
    int device_id = GetDeviceId();
    return GetDeviceBlockManager(device_id)->SwapOut(device_blocks, context_->d2h_streams_[device_id]);
  }

  // Get free block number for current selected device.
  int64_t GetFreeBlockNumber(MemoryDevice device = MemoryDevice::MEMORY_GPU) {
    return GetDeviceBlockManager()->GetFreeBlockNumber(device);
  }

  // Get the size in bytes for one block.
  size_t GetBlockSize() const { return block_manager_config_.device_allocator_config.block_size; }

  // get the token number for one block.
  size_t GetBlockTokenNum() const { return block_manager_config_.device_allocator_config.block_token_num; }

 private:
  // Get device block manager on current selected device.
  std::shared_ptr<DeviceBlockManager> GetDeviceBlockManager() {
    int device_id = GetDeviceId();
    return dev_block_managers_[device_id];
  }

  std::shared_ptr<DeviceBlockManager> GetDeviceBlockManager(int device_id) { return dev_block_managers_[device_id]; }

 private:
  // Every deivce has its own manager.
  std::vector<std::shared_ptr<DeviceBlockManager>> dev_block_managers_;

  BlockManagerConfig block_manager_config_;

  // The global context.
  std::shared_ptr<Context> context_ = nullptr;
};

}  // namespace numerous_llm
