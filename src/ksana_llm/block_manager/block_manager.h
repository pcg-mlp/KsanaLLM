/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <vector>

#include "ksana_llm/block_manager/block_manager_interface.h"

#include "ksana_llm/block_manager/device_allocator.h"
#include "ksana_llm/block_manager/host_allocator.h"

namespace ksana_llm {

// The memory pool management.
class BlockManager : public BlockManagerInterface {
 public:
  BlockManager(const BlockManagerConfig& block_manager_config, std::shared_ptr<Context> context);

  ~BlockManager() {}

  // Preallocate blocks.
  Status PreAllocateBlocks();

  // Reset the preallocated blocks for device & host.
  Status ResetPreAllocatedBlocks();

  // This function maybe called concurrently from different threads.
  // DO NOT store the device id in variable.
  void SetDeviceId(int device_id);

  // This function maybe called concurrently from different threads.
  int GetDeviceId();

  // Allocate blocked memory on device.
  Status AllocateBlocks(int64_t block_num, std::vector<int>& blocks);

  // The data type of the memory block allocated.
  DataType GetDtype();

  // Allocate contiguous memory on device.
  Status AllocateContiguous(int64_t size, int& block_id);

  // Free blocked memory on device.
  Status FreeBlocks(const std::vector<int>& blocks);

  // Free contiguous memory on device.
  Status FreeContiguous(int block_id);

  // Check contiguous memory is in used.
  bool IsContiguousUsed(const int block_id);

  // Get memory addresses of blocked memory on device.
  Status GetBlockPtrs(const std::vector<int>& blocks, std::vector<void*>& addrs);

  // Get memory address of contiguous memory on device.
  Status GetContiguousPtr(int block_id, void*& addr);

  // Get number of free blocked memory on device.
  size_t GetDeviceFreeBlockNumber();

  // Get number of used blocked memory on device.
  size_t GetDeviceUsedBlockNumber();

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
  size_t GetHostFreeBlockNumber();

  // Get number of used blocked memory on host.
  size_t GetHostUsedBlockNumber();

  // The swap out/in for single block, the device block has been allocated on current device.
  // Do not free memory after swapness, the caller will do that.
  Status SwapOut(int host_block_id, int device_block_id);
  Status SwapIn(int device_block_id, int host_block_id);

  // Drop the swapped blocks on host, and the block ids could be resued.
  Status SwapDrop(const std::vector<int>& host_blocks);

  // Get the size in bytes for one block.
  size_t GetBlockSize() const;

  // Get the token number for one block.
  size_t GetBlockTokenNum() const;

  // Get block manager config
  const BlockManagerConfig& GetBlockManagerConfig() const;

  // Get block base ptr
  void* GetBlockBasePtr();

  // Get block manager's related allocator's config
  const AllocatorConfig& GetAllocatorConfig();

  // Get the first allocated block id from continuous blocks memory space
  int GetBlocksBaseId();

  // Get each device workspace buffer block id and size
  WorkspaceMeta& GetWorkspaceMeta();

  // Update block magager config
  Status UpdateConfig(const BlockManagerConfig& update_block_manager_config);

 private:
  // Calculate the block number.
  Status CalculateBlockNumber(size_t& device_blocks_num, size_t& host_block_num);

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

  // The prefix cache blocks on different device
  std::vector<std::vector<int>> prefix_cache_blocks_;

  // The prefix cache blocks number
  size_t prefix_cache_block_num_{0ul};

  // The target prefix cache token
  std::vector<int> prefix_cache_tokens_;

  // Each device's workspace buffer meta
  std::vector<WorkspaceMeta> workspace_metas_;
};

}  // namespace ksana_llm
