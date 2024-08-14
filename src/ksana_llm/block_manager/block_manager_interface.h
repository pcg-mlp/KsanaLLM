/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <vector>

#include "ksana_llm/utils/context.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

// The memory pool management.
class BlockManagerInterface {
 public:
  virtual ~BlockManagerInterface() {}

  // Preallocate blocks.
  virtual Status PreAllocateBlocks() = 0;

  // Reset the preallocated blocks for device & host.
  virtual Status ResetPreAllocatedBlocks() = 0;

  // This function maybe called concurrently from different threads.
  // DO NOT store the device id in variable.
  virtual void SetDeviceId(int device_id) = 0;

  // This function maybe called concurrently from different threads.
  virtual int GetDeviceId() = 0;

  // The data type of the memory block allocated.
  virtual DataType GetDtype() = 0;

  // Allocate blocked memory on device.
  virtual Status AllocateBlocks(int64_t block_num, std::vector<int>& blocks) = 0;

  // Allocate contiguous memory on device.
  virtual Status AllocateContiguous(int64_t size, int& block_id) = 0;

  // Free blocked memory on device.
  virtual Status FreeBlocks(const std::vector<int>& blocks) = 0;

  // Free contiguous memory on device.
  virtual Status FreeContiguous(int block_id) = 0;

  // Check contiguous memory is in used.
  virtual bool IsContiguousUsed(const int block_id) = 0;

  // Get memory addresses of blocked memory on device.
  virtual Status GetBlockPtrs(const std::vector<int>& blocks, std::vector<void*>& addrs) = 0;

  // Get memory address of contiguous memory on device.
  virtual Status GetContiguousPtr(int block_id, void*& addr) = 0;

  // Get number of free blocked memory on device.
  virtual size_t GetDeviceFreeBlockNumber() = 0;

  // Get number of used blocked memory on device.
  virtual size_t GetDeviceUsedBlockNumber() = 0;

  // Allocate blocked memory on host.
  virtual Status AllocateHostBlocks(int64_t block_num, std::vector<int>& blocks) = 0;

  // Allocate contiguous memory on host.
  virtual Status AllocateHostContiguous(int64_t size, int& block_id) = 0;

  // Free blocked memory on host.
  virtual Status FreeHostBlocks(const std::vector<int>& blocks) = 0;

  // Free contiguous memory on host.
  virtual Status FreeHostContiguous(int block_id) = 0;

  // Get memory addresses of blocked memory on host.
  virtual Status GetHostBlockPtrs(const std::vector<int>& blocks, std::vector<void*>& addrs) = 0;

  // Get memory address of contiguous memory on host.
  virtual Status GetHostContiguousPtr(int block_id, void*& addr) = 0;

  // Get number of free blocked memory on host.
  virtual size_t GetHostFreeBlockNumber() = 0;

  // Get number of used blocked memory on host.
  virtual size_t GetHostUsedBlockNumber() = 0;

  // The swap out/in for single block, the device block has been allocated on current device.
  // Do not free memory after swapness, the caller will do that.
  virtual Status SwapOut(int host_block_id, int device_block_id) { return Status(); }
  virtual Status SwapIn(int device_block_id, int host_block_id) { return Status(); }

  // Drop the swapped blocks on host, and the block ids could be resued.
  virtual Status SwapDrop(const std::vector<int>& host_blocks) = 0;

  // Get the size in bytes for one block.
  virtual size_t GetBlockSize() const = 0;

  // Get the token number for one block.
  virtual size_t GetBlockTokenNum() const = 0;

  // Get block manager config
  virtual const BlockManagerConfig& GetBlockManagerConfig() const = 0;

  // Get block base ptr
  virtual void* GetBlockBasePtr() = 0;

  // Get block manager's related allocator's config.
  virtual const AllocatorConfig& GetAllocatorConfig() = 0;

  // Get the first allocated block id
  virtual int GetBlocksBaseId() = 0;
};

}  // namespace ksana_llm
