/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <list>
#include <unordered_map>
#include <vector>

#include "ksana_llm/block_manager/block_manager_interface.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

// A faked block manger, used for test only.
class FakedBlockManager : public BlockManagerInterface {
 public:
  FakedBlockManager(const BlockManagerConfig& block_manager_config, size_t device_num) {
    block_manager_config_ = block_manager_config;

    device_num_ = device_num;
    workspace_metas_.resize(device_num_);
    cur_device_id_ = 0;
  }

  ~FakedBlockManager() {}

  // Preallocate blocks.
  Status PreAllocateBlocks() {
    for (size_t i = 0; i < block_manager_config_.host_allocator_config.blocks_num; ++i) {
      free_host_blocks_.push_back(i);
    }

    for (size_t i = 0; i < device_num_; ++i) {
      for (size_t j = 0; j < block_manager_config_.device_allocator_config.blocks_num; ++j) {
        free_device_blocks_[i].push_back(j);
      }
    }

    return Status();
  }

  // Reset the preallocated blocks for device & host.
  Status ResetPreAllocatedBlocks() {
    KLLM_CHECK_WITH_INFO(false, "Not implemented");
    return Status();
  }

  // This function maybe called concurrently from different threads.
  // DO NOT store the device id in variable.
  void SetDeviceId(int device_id) { cur_device_id_ = device_id; }

  // This function maybe called concurrently from different threads.
  int GetDeviceId() { return cur_device_id_; }

  DataType GetDtype() {
    KLLM_CHECK_WITH_INFO(false, "Not implemented");
    return TYPE_INVALID;
  }

  // Allocate blocked memory on devicen
  Status AllocateBlocks(int64_t block_num, std::vector<int>& blocks) {
    blocks.clear();
    size_t needed_block_num = block_num;
    while (needed_block_num > 0 && !free_device_blocks_[cur_device_id_].empty()) {
      int block_id = free_device_blocks_[cur_device_id_].front();
      blocks.push_back(block_id);

      free_device_blocks_[cur_device_id_].pop_front();
      used_device_blocks_[cur_device_id_].push_back(block_id);

      --needed_block_num;
    }

    return needed_block_num == 0 ? Status() : Status(RET_OUT_OF_MEMORY, "No more device blocks.");
  }

  // Allocate contiguous memory on device.
  Status AllocateContiguous(int64_t size, int& block_id) {
    KLLM_CHECK_WITH_INFO(false, "Not implemented");
    return Status();
  }

  // Free blocked memory on device.
  Status FreeBlocks(const std::vector<int>& blocks) {
    KLLM_CHECK_WITH_INFO(false, "Not implemented");
    return Status();
  }

  // Free contiguous memory on device.
  Status FreeContiguous(int block_id) {
    KLLM_CHECK_WITH_INFO(false, "Not implemented");
    return Status();
  }

  // Check contiguous memory is in used.
  bool IsContiguousUsed(const int block_id) {
    KLLM_CHECK_WITH_INFO(false, "Not implemented");
    return false;
  }

  // Get memory addresses of blocked memory on device.
  Status GetBlockPtrs(const std::vector<int>& blocks, std::vector<void*>& addrs) {
    KLLM_CHECK_WITH_INFO(false, "Not implemented");
    return Status();
  }

  // Get memory address of contiguous memory on device.
  Status GetContiguousPtr(int block_id, void*& addr) {
    KLLM_CHECK_WITH_INFO(false, "Not implemented");
    return Status();
  }

  // Get number of free blocked memory on device.
  size_t GetDeviceFreeBlockNumber() { return free_device_blocks_[cur_device_id_].size(); }

  // Get number of used blocked memory on device.
  size_t GetDeviceUsedBlockNumber() { return used_device_blocks_[cur_device_id_].size(); }

  // Allocate blocked memory on host.
  Status AllocateHostBlocks(int64_t block_num, std::vector<int>& blocks) {
    blocks.clear();
    size_t needed_block_num = block_num;
    while (needed_block_num > 0 && !free_host_blocks_.empty()) {
      int block_id = free_host_blocks_.front();
      blocks.push_back(block_id);

      free_host_blocks_.pop_front();
      used_host_blocks_.push_back(block_id);

      --needed_block_num;
    }

    return needed_block_num == 0 ? Status() : Status(RET_OUT_OF_MEMORY, "No more device blocks.");
  }

  // Allocate contiguous memory on host.
  Status AllocateHostContiguous(int64_t size, int& block_id) {
    KLLM_CHECK_WITH_INFO(false, "Not implemented");
    return Status();
  }

  // Free blocked memory on host.
  Status FreeHostBlocks(const std::vector<int>& blocks) {
    for (int block_id : blocks) {
      auto it = std::find(used_host_blocks_.begin(), used_host_blocks_.end(), block_id);
      if (it != used_host_blocks_.end()) {
        used_host_blocks_.erase(it);
      }
      free_host_blocks_.push_back(block_id);
    }

    return Status();
  }

  // Free contiguous memory on host.
  Status FreeHostContiguous(int block_id) {
    KLLM_CHECK_WITH_INFO(false, "Not implemented");
    return Status();
  }

  // Get memory addresses of blocked memory on host.
  Status GetHostBlockPtrs(const std::vector<int>& blocks, std::vector<void*>& addrs) {
    KLLM_CHECK_WITH_INFO(false, "Not implemented");
    return Status();
  }

  // Get memory address of contiguous memory on host.
  Status GetHostContiguousPtr(int block_id, void*& addr) {
    KLLM_CHECK_WITH_INFO(false, "Not implemented");
    return Status();
  }

  // Get number of free blocked memory on host.
  size_t GetHostFreeBlockNumber() { return free_host_blocks_.size(); }

  // Get number of used blocked memory on host.
  size_t GetHostUsedBlockNumber() { return used_host_blocks_.size(); }

  // Swap out blocks from device to host,
  Status SwapOut(const std::vector<int>& device_blocks, std::vector<int>& host_blocks,
                 const int host_block_num_to_add) {
    KLLM_CHECK_WITH_INFO(false, "Not implemented");
    return Status();
  }

  // Swap in blocks from host to device.
  Status SwapIn(const std::vector<int>& host_blocks, std::vector<int>& device_blocks) {
    KLLM_CHECK_WITH_INFO(false, "Not implemented");
    return Status();
  }

  // The swap out/in for single block.
  Status SwapOut(int host_block_id, int device_block_id) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return Status();
  }

  // The swap out/in for single block.
  Status SwapIn(int device_block_id, int host_block_id) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return Status();
  }

  // Drop the swapped blocks on host, and the block ids could be resued.
  Status SwapDrop(const std::vector<int>& host_blocks) {
    FreeHostBlocks(host_blocks);
    return Status();
  }

  // Get the size in bytes for one block.
  size_t GetBlockSize() const {
    KLLM_CHECK_WITH_INFO(false, "Not implemented");
    return 0;
  }

  // Get the token number for one block.
  size_t GetBlockTokenNum() const { return block_manager_config_.device_allocator_config.block_token_num; }

  // Prepare blocks for prefix cache
  Status PreparePrefixCacheBlocks() {
    KLLM_CHECK_WITH_INFO(false, "Not implemented");
    return Status();
  }

  // Get the prefix cache tokens numbers
  int GetPrefixCacheTokensNumber() const {
    KLLM_CHECK_WITH_INFO(false, "Not implemented");
    return 0;
  }

  // Get the prefix cache blocks numbers
  size_t GetPrefixCacheBlocksNumber() const {
    KLLM_CHECK_WITH_INFO(false, "Not implemented");
    return 0;
  }

  // Check the input token is valid for prefix cache
  bool CheckReqIsValidForPrefixCache(const std::vector<int>& input_tokens) {
    KLLM_CHECK_WITH_INFO(false, "Not implemented");
    return false;
  }

  // Fill prefix kv cache to input blocks vector
  Status FillPrefixCacheBlocks(std::vector<std::vector<int>>& kv_cache_blocks) {
    KLLM_CHECK_WITH_INFO(false, "Not implemented");
    return Status();
  }

  // Get block manager config
  const BlockManagerConfig& GetBlockManagerConfig() const { return block_manager_config_; }

  void* GetBlockBasePtr() { return nullptr; }

  const AllocatorConfig& GetAllocatorConfig() { return block_manager_config_.device_allocator_config; }

  int GetBlocksBaseId() { return 0; }

  WorkspaceMeta& GetWorkspaceMeta() {return workspace_metas_[0];}

 private:
  BlockManagerConfig block_manager_config_;

  std::vector<WorkspaceMeta> workspace_metas_;
  size_t device_num_;
  int cur_device_id_ = 0;

  std::unordered_map<int, std::list<int>> free_device_blocks_;
  std::unordered_map<int, std::list<int>> used_device_blocks_;

  std::list<int> free_host_blocks_;
  std::list<int> used_host_blocks_;
};

// A faked token generator.
class FakedTokenGenerator {
 public:
  // Generate some tokens by pair of seed and length.
  void GeneratePromptTokens(const std::vector<std::pair<int, int>>& seeds, std::vector<int>& token_ids) {
    for (auto& pair : seeds) {
      std::srand(pair.first);
      for (int i = 0; i < pair.second; ++i) {
        token_ids.push_back(std::rand() % vocab_size);
      }
    }
  }

  // Generate a random size.
  int GenerateRandomInteger(int min, int max) {
    std::srand(std::time(nullptr));
    return min + (std::rand() % (max - min));
  }

  // Generate a new token.
  void GenerateOneToken(size_t token_num, std::vector<int>& token_ids) {
    for (size_t i = 0; i < token_num; ++i) {
      token_ids.push_back(GenerateRandomInteger(0, vocab_size));
    }
  }

 private:
  size_t vocab_size = 32000;
};

}  // namespace ksana_llm
