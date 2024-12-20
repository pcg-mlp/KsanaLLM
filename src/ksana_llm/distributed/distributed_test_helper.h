/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <thread>
#include "ksana_llm/block_manager/block_manager_interface.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

class FakedTestBlockManager : public BlockManagerInterface {
 public:
  FakedTestBlockManager() {}

  ~FakedTestBlockManager() {}

  void SetBlockNumber(size_t device_blocks_num, size_t host_block_num) {
    std::unique_lock<std::mutex> lock(mutex_);

    std::thread::id thread_id = std::this_thread::get_id();
    thread_block_nums_[thread_id] = std::make_pair(device_blocks_num, host_block_num);
  }

  // Get calculated block number.
  Status GetBlockNumber(size_t& device_blocks_num, size_t& host_block_num) {
    std::unique_lock<std::mutex> lock(mutex_);

    std::thread::id thread_id = std::this_thread::get_id();
    device_blocks_num = thread_block_nums_[thread_id].first;
    host_block_num = thread_block_nums_[thread_id].second;
    return Status();
  }

  Status PreAllocateBlocks() { return Status(); }
  Status ResetPreAllocatedBlocks() { return Status(); }
  void SetDeviceId(int device_id) {}
  int GetDeviceId() { return 0; }
  DataType GetDtype() { return DataType::TYPE_FP16; }
  Status AllocateBlocks(int64_t block_num, std::vector<int>& blocks) { return Status(); }
  Status AllocateContiguous(int64_t size, int& block_id) { return Status(); }
  Status FreeBlocks(const std::vector<int>& blocks) { return Status(); }
  Status FreeContiguous(int block_id) { return Status(); }
  bool IsContiguousUsed(const int block_id) { return false; }
  Status GetBlockPtrs(const std::vector<int>& blocks, std::vector<void*>& addrs) { return Status(); }
  Status GetContiguousPtr(int block_id, void*& addr) { return Status(); }
  size_t GetDeviceFreeBlockNumber() { return 0; }
  size_t GetDeviceUsedBlockNumber() { return 0; }
  Status AllocateHostBlocks(int64_t block_num, std::vector<int>& blocks) { return Status(); }
  Status AllocateHostContiguous(int64_t size, int& block_id) { return Status(); }
  Status FreeHostBlocks(const std::vector<int>& blocks) { return Status(); }
  Status FreeHostContiguous(int block_id) { return Status(); }
  Status GetHostBlockPtrs(const std::vector<int>& blocks, std::vector<void*>& addrs) { return Status(); }
  Status GetHostContiguousPtr(int block_id, void*& addr) { return Status(); }
  size_t GetHostFreeBlockNumber() { return 0; }
  size_t GetHostUsedBlockNumber() { return 0; }
  Status SwapOut(int host_block_id, int device_block_id) { return Status(); }
  Status SwapIn(int device_block_id, int host_block_id) { return Status(); }
  Status SwapDrop(const std::vector<int>& host_blocks) { return Status(); }
  size_t GetBlockSize() const { return 0; }
  size_t GetBlockTokenNum() const { return 0; }
  const BlockManagerConfig& GetBlockManagerConfig() const { return block_manager_config_; }
  void* GetBlockBasePtr() { return nullptr; }
  const AllocatorConfig& GetAllocatorConfig() { return allocator_config_; }
  int GetBlocksBaseId() { return 0; }
  WorkspaceMeta& GetWorkspaceMeta() { return workspace_metas_; }

 private:
  BlockManagerConfig block_manager_config_;
  AllocatorConfig allocator_config_;
  WorkspaceMeta workspace_metas_;

  std::unordered_map<std::thread::id, std::pair<size_t, size_t>> thread_block_nums_;
  std::mutex mutex_;
};

}  // namespace ksana_llm
