/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <vector>

#include "ksana_llm/utils/context.h"
#include "ksana_llm/utils/environment.h"

namespace ksana_llm {

// The memory pool management.
class BlockManagerInterface {
 public:
  virtual ~BlockManagerInterface() {}

  // Preallocate blocks.
  virtual Status PreAllocateBlocks() = 0;

  // Reset the preallocated blocks for device & hosg.
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

  // Swap out blocks from device to host,
  // it could be swapped in later and keep block id not changed.
  virtual Status SwapOut(const std::vector<int>& device_blocks, std::vector<int>& host_blocks,
                         const int host_block_num_to_add) = 0;

  // Swap in blocks from host to device.
  virtual Status SwapIn(const std::vector<int>& host_blocks, std::vector<int>& device_blocks) = 0;

  // Drop the swapped blocks on host, and the block ids could be resued.
  virtual Status SwapDrop(const std::vector<int>& host_blocks) = 0;

  // Get the size in bytes for one block.
  virtual size_t GetBlockSize() const = 0;

  // Get the token number for one block.
  virtual size_t GetBlockTokenNum() const = 0;

  // Prepare blocks for prefix cache
  virtual Status PreparePrefixCacheBlocks() = 0;

  // Get the prefix cache tokens numbers
  virtual int GetPrefixCacheTokensNumber() const = 0;

  // Get the prefix cache blocks numbers
  virtual size_t GetPrefixCacheBlocksNumber() const = 0;

  // Check the input token is valid for prefix cache
  virtual bool CheckReqIsValidForPrefixCache(const std::vector<int>& input_tokens) = 0;

  // Fill prefix kv cache to input blocks vector
  virtual Status FillPrefixCacheBlocks(std::vector<std::vector<int>>& kv_cache_blocks) = 0;

  // Get block manager config
  virtual const BlockManagerConfig& GetBlockManagerConfig() const = 0;
};

}  // namespace ksana_llm
