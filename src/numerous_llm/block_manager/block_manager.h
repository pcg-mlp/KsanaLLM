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
// 用于管理设备存储、内存块的分配、回收、迁移等操作
class BlockManager {
public:
  // 构造函数

  explicit BlockManager(int device_id = 0);


  // 构造函数
  // 参数：block_manager_config - BlockManager 的配置信息
  explicit BlockManager(const BlockManagerConfig &block_manager_config, int device_id = 0);

  // 析构函数
  ~BlockManager();

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
  Status FreeBlocks(std::vector<int>& blocks);

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
    return device_id_;
  }

  // Get the size in bytes for one block.
  size_t GetBlockSize() const {
    // TODO
    return 65536;
  }

private:
  std::mutex swap_mutex_;
  BlockAllocator device_allocator;
  BlockAllocator cpu_allocator;
  std::unordered_map<int64_t, int64_t> swap_map_;
  int64_t block_size_;
  int device_id_;
};

// 定义一个模板类 DeviceSelect，用于根据设备 ID 选择并执行特定操作。
template<typename T>
class DeviceSelect {
public:
    // 构造函数，初始化device_ptr_vec_
    DeviceSelect() {
      int device_number = GetDeviceNumber();
      for (int i = 0; i < device_number ;i++) {
        device_ptr_vec_.push_back(std::make_shared<T>(i));
      }
    }

    // 执行函数，根据设备 ID 执行给定的函数和参数。
    template <typename Func, typename... Args>
    decltype(auto) Execute(int device_id, Func&& func, Args&&... args) {
        // 检查设备 ID 是否有效。
        if (device_id < 0 || device_id >= device_ptr_vec_.size()) {
            throw std::out_of_range("Invalid device_id.");
        }
        // 多卡的通用操作在这里执行
        CUDA_CHECK(cudaSetDevice(device_id));
        // 获取设备对应的实例，并执行给定的函数和参数。
        auto& ins = *device_ptr_vec_[device_id].get();
        return (ins.*func)(std::forward<Args>(args)...);
    }

private:
    // 用于存储设备实例。
    std::vector<std::shared_ptr<T>> device_ptr_vec_;
};

#define DEVICE_EXECUTE(device_id, class, func, ...)  \
    Singleton<DeviceSelect<class>>::GetInstance()->Execute(device_id, &class::func, ##__VA_ARGS__)

} // namespace numerous_llm
