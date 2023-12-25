/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/block_manager/block_manager.h"
#include "numerous_llm/utils/logger.h"
#include "numerous_llm/utils/singleton.h"

namespace numerous_llm {

// 构造函数，根据device id, Singleton Instance的 BlockManagerConfig 配置 DeviceBlockManager
DeviceBlockManager::DeviceBlockManager(int device_id) : DeviceBlockManager(GetBlockManagerConfig(), device_id) {}

// 获取配置
BlockManagerConfig DeviceBlockManager::GetBlockManagerConfig() {
  BlockManagerConfig block_manager_config;
  Singleton<Environment>::GetInstance()->GetBlockManagerConfig(block_manager_config);
  return block_manager_config;
}

// 构造函数，根据给定的 BlockManagerConfig 配置 DeviceBlockManager
DeviceBlockManager::DeviceBlockManager(const BlockManagerConfig& block_manager_config, int device_id)
    : device_allocator(block_manager_config.device_allocator_config),
      cpu_allocator(block_manager_config.cpu_allocator_config),
      device_id_(device_id) {
  block_size_ = block_manager_config.device_allocator_config.block_size;
  NLLM_LOG_INFO << "DeviceBlockManager Init Success";
}

// 析构函数，释放DeviceBlockManager分配的所有内存
DeviceBlockManager::~DeviceBlockManager() {}

// 根据给定的block_ids，获取对应的内存指针，存储在addrs中
Status DeviceBlockManager::GetBlockPtrs(const std::vector<int>& blocks, std::vector<void*>& addrs) {
  return device_allocator.GetBlockPtrs(blocks, addrs);
}

// 分配block_num个块，将分配成功的块的id存储在blocks中
Status DeviceBlockManager::AllocateBlocks(int64_t block_num, std::vector<int>& blocks) {
  return device_allocator.Allocate(block_num, blocks);
}

// 分配指定大小的设备存储空间
Status DeviceBlockManager::AllocateContiguous(int64_t size, int& block_id) {
  return device_allocator.AllocateContiguous(size, block_id);
}

// 释放给定的blocks，将它们从used_device_block_map_移动到free_device_block_map_
Status DeviceBlockManager::FreeBlocks(const std::vector<int>& blocks) { return device_allocator.Free(blocks); }

// 释放连续设备存储
Status DeviceBlockManager::FreeContiguous(int block_id) { return device_allocator.FreeContiguous(block_id); }

Status DeviceBlockManager::SwapIn(std::vector<int>& device_blocks, cudaStream_t stream) {
  std::unique_lock<std::mutex> lock(swap_mutex_);
  std::vector<void*> device_addrs;
  STATUS_CHECK_RETURN(GetBlockPtrs(device_blocks, device_addrs));
  std::vector<int> cpu_blocks;
  std::vector<void*> cpu_addrs;
  STATUS_CHECK_RETURN(cpu_allocator.Allocate(device_blocks.size(), cpu_blocks));
  STATUS_CHECK_RETURN(cpu_allocator.GetBlockPtrs(cpu_blocks, cpu_addrs));
  for (int i = 0; i < device_blocks.size(); i++) {
    swap_map_[device_blocks[i]] = cpu_blocks[i];
    CUDA_CHECK(cudaMemcpyAsync(cpu_addrs[i], device_addrs[i], block_size_, cudaMemcpyDeviceToHost, stream));
  }
  return Status();
}

Status DeviceBlockManager::SwapOut(std::vector<int>& device_blocks, cudaStream_t stream) {
  std::unique_lock<std::mutex> lock(swap_mutex_);
  std::vector<void*> device_addrs;
  STATUS_CHECK_RETURN(GetBlockPtrs(device_blocks, device_addrs));
  std::vector<int> cpu_blocks(1);
  std::vector<void*> cpu_addrs(1);
  for (int i = 0; i < device_blocks.size(); i++) {
    auto it = swap_map_.find(device_blocks[i]);
    if (it != swap_map_.end()) {
      cpu_blocks[0] = it->second;
      STATUS_CHECK_RETURN(cpu_allocator.GetBlockPtrs(cpu_blocks, cpu_addrs));
      CUDA_CHECK(cudaMemcpyAsync(device_addrs[i], cpu_addrs[0], block_size_, cudaMemcpyHostToDevice, stream));
      STATUS_CHECK_RETURN(cpu_allocator.Free(cpu_blocks));
      swap_map_.erase(it);
    } else {
      return Status(RET_SEGMENT_FAULT);
    }
  }
  return Status();
}

// 函数：获取指定设备类型的空闲内存块数量
// 参数：device - 设备类型
// 返回值：空闲内存块数量
int64_t DeviceBlockManager::GetFreeBlockNumber(MemoryDevice device) {
  switch (device) {
    case MEMORY_CPU_PINNED:
      return cpu_allocator.GetFreeBlockNumber();

    case MEMORY_GPU:
      return device_allocator.GetFreeBlockNumber();

    default:
      return 0;
  }
}

}  // namespace numerous_llm
