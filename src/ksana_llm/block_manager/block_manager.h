/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <vector>

#include "ksana_llm/block_manager/device_allocator.h"
#include "ksana_llm/block_manager/host_allocator.h"
#include "ksana_llm/runtime/context.h"
#include "ksana_llm/utils/environment.h"

namespace ksana_llm {

// The memory pool management.
class BlockManager {
 public:
  BlockManager(const BlockManagerConfig& block_manager_config, std::shared_ptr<Context> context);

  ~BlockManager() {}

  // This function maybe called concurrently from different threads.
  // DO NOT store the device id in variable.
  void SetDeviceId(int device_id);

  // This function maybe called concurrently from different threads.
  int GetDeviceId();

  // Allocate blocked memory on device.
  Status AllocateBlocks(int64_t block_num, std::vector<int>& blocks);

  // Allocate contiguous memory on device.
  Status AllocateContiguous(int64_t size, int& block_id);

  // Free blocked memory on device.
  Status FreeBlocks(const std::vector<int>& blocks);

  // Free contiguous memory on device.
  Status FreeContiguous(int block_id);

  // Get memory addresses of blocked memory on device.
  Status GetBlockPtrs(const std::vector<int>& blocks, std::vector<void*>& addrs);

  // Get memory address of contiguous memory on device.
  Status GetContiguousPtr(int block_id, void*& addr);

  // Get number of free blocked memory on device.
  int GetFreeBlockNumber();

  // Get number of used blocked memory on device.
  int GetUsedBlockNumber();

  // Allocate blocked memory on host.
  Status AllocateHostBlocks(int64_t block_num, std::vector<int>& blocks);

  // Allocate contiguous memory on host.
  Status AllocateHostContiguous(int64_t size, int& block_id);

  // Free blocked memory on host.
  Status FreeHostBlocks(const std::vector<int>& blocks);

  // Free contiguous memory on host.
  Status FreeHostContiguous(int block_id);

  // Get memory addresses of blocked memory on host.
  Status GetHostBlockPtrs(const std::vector<int>& blocks, std::vector<void*>& addrs);

  // Get memory address of contiguous memory on host.
  Status GetHostContiguousPtr(int block_id, void*& addr);

  // Get number of free blocked memory on host.
  int GetHostFreeBlockNumber();

  // Get number of used blocked memory on host.
  int GetHostUsedBlockNumber();

  // Swap out blocks from device to host,
  // it could be swapped in later and keep block id not changed.
  Status SwapOut(const std::vector<int>& device_blocks, std::vector<int>& host_blocks);

  // Swap in blocks from host to device.
  Status SwapIn(const std::vector<int>& host_blocks, std::vector<int>& device_blocks);

  // Drop the swapped blocks on host, and the block ids could be resued.
  Status SwapDrop(const std::vector<int>& host_blocks);

  // Get the size in bytes for one block.
  size_t GetBlockSize() const;

  // get the token number for one block.
  size_t GetBlockTokenNum() const;

 private:
  // Get the device allocator for current selected device.
  std::shared_ptr<DeviceAllocator>& GetDeviceAllocator();

  // Get the global host allocator.
  std::shared_ptr<HostAllocator>& GetHostAllocator();

 private:
  BlockManagerConfig block_manager_config_;

  // The global context.
  std::shared_ptr<Context> context_ = nullptr;

  // The allocator used to manage cpu memory.
  std::shared_ptr<HostAllocator> host_allocator_;

  // The allocators used to manage device memory, one allocator for one device.
  std::vector<std::shared_ptr<DeviceAllocator>> device_allocators_;
};

}  // namespace ksana_llm
