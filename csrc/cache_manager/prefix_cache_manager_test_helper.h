/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <algorithm>
#include <list>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "cache_manager/block_allocator/block_allocator_interface.h"
#include "cache_manager/block_allocator/block_allocator_manager.h"
#include "configure/context.h"
#include "device/device_types.h"
#include "utils/memory_allocator_interface.h"
#include "utils/ret_code.h"
#include "utils/status.h"

namespace ksana_llm {

class FakedMemoryAllocator : public MemoryAllocatorInterface {
 public:
  virtual ~FakedMemoryAllocator() {}

  virtual void SetDevice(int device_id) override {}
  virtual void GetDevice(int* device_id) override {}

  virtual void Malloc(void** dev_ptr, size_t size) override {}
  virtual void MallocAsync(void** dev_ptr, size_t size, Stream stream) override {}

  virtual void MemsetAsync(void* dev_ptr, int value, size_t count, Stream stream) override {}
  virtual void Memset(void* dev_ptr, int value, size_t count) override {}

  virtual void MemcpyAsync(void* dst, const void* src, size_t count, enum MemcpyKind kind, Stream stream) override {}
  virtual void Memcpy(void* dst, const void* src, size_t count, enum MemcpyKind kind) override {}

  virtual void Free(void* dev_ptr) override {}
  virtual void FreeAsync(void* dev_ptr, Stream stream) override {}

  virtual void HostAlloc(void** host_ptr, size_t size) override {}

  virtual void HostFree(void* host_ptr) override {}
};

class FakedBlockAllocator : public BlockAllocatorInterface {
 public:
  FakedBlockAllocator(MemoryLocation location, size_t block_num, size_t block_size, int rank = 0,
                      std::shared_ptr<MemoryAllocatorInterface> memory_allocator = nullptr,
                      std::shared_ptr<Context> context = nullptr) {
    location_ = location;
    block_num_ = block_num;
    block_size_ = block_size;
    rank_ = rank;
    memory_allocator_ = memory_allocator;
    context_ = context;
  }

  virtual ~FakedBlockAllocator() {}

  virtual void PreAllocateBlocks() override {
    std::lock_guard<std::mutex> guard(mutex_);
    for (size_t i = 0; i < block_num_; ++i) {
      free_blocks_.push_back(i);
    }
  }

  virtual void Clear() override {
    std::lock_guard<std::mutex> guard(mutex_);
    free_blocks_.clear();
    used_blocks_.clear();
  }

  virtual Status AllocateBlocks(size_t block_num, std::vector<int>& blocks) override {
    std::lock_guard<std::mutex> guard(mutex_);
    blocks.clear();
    size_t needed_block_num = block_num;
    while (needed_block_num > 0 && !free_blocks_.empty()) {
      int block_id = free_blocks_.front();
      blocks.push_back(block_id);

      free_blocks_.pop_front();
      used_blocks_.push_back(block_id);

      --needed_block_num;
    }

    RetCode ret_code = (location_ == LOCATION_HOST) ? RET_OUT_OF_HOST_MEMORY : RET_OUT_OF_DEVICE_MEMORY;
    return (needed_block_num == 0) ? Status() : Status(ret_code, "No more blocks.");
  }

  virtual Status FreeBlocks(const std::vector<int>& blocks) override {
    std::lock_guard<std::mutex> guard(mutex_);
    for (int block_id : blocks) {
      auto it = std::find(used_blocks_.begin(), used_blocks_.end(), block_id);
      if (it != used_blocks_.end()) {
        used_blocks_.erase(it);
      }
      free_blocks_.push_back(block_id);
    }

    return Status();
  }

  virtual Status GetBlockPtrs(const std::vector<int>& blocks, std::vector<void*>& addrs) override {
    addrs.resize(blocks.size(), nullptr);
    return Status();
  }

  virtual void* GetBlocksBasePtr() override { return nullptr; }

  virtual int GetBlocksBaseId() override { return 0; }

  virtual size_t GetFreeBlockNumber() override {
    std::lock_guard<std::mutex> guard(mutex_);
    return free_blocks_.size();
  }

  virtual size_t GetUsedBlockNumber() override {
    std::lock_guard<std::mutex> guard(mutex_);
    return used_blocks_.size();
  }

 private:
  MemoryLocation location_ = MemoryLocation::LOCATION_HOST;

  size_t block_num_;
  size_t block_size_;

  int rank_ = -1;

  std::shared_ptr<MemoryAllocatorInterface> memory_allocator_ = nullptr;

  // The global context.
  std::shared_ptr<Context> context_ = nullptr;

  std::list<int> free_blocks_;
  std::list<int> used_blocks_;
  std::mutex mutex_;
};

class FakedBlockAllocatorGroup : public BlockAllocatorGroupInterface {
 public:
  FakedBlockAllocatorGroup(const BlockAllocatorGroupConfig& block_allocator_group_config,
                           std::shared_ptr<MemoryAllocatorInterface> memory_allocator, std::shared_ptr<Context> context,
                           BlockAllocatorCreationFunc block_allocator_creation_fn)
      : block_allocator_group_config_(block_allocator_group_config),
        memory_allocator_(memory_allocator),
        context_(context),
        block_allocator_creation_fn_(block_allocator_creation_fn) {
    Initialize();
  }

  virtual ~FakedBlockAllocatorGroup() {}

  virtual std::vector<int> GetBlockAllocatorDevices() const override { return block_allocator_group_config_.devices; }

  virtual std::shared_ptr<BlockAllocatorInterface> GetHostBlockAllocator() const override { return host_allocator_; }

  virtual std::shared_ptr<BlockAllocatorInterface> GetDeviceBlockAllocator(int index = 0) const override {
    size_t ull_idx = static_cast<size_t>(index);
    if (ull_idx >= block_allocator_group_config_.devices.size()) {
      return nullptr;
    }
    int device_id = block_allocator_group_config_.devices[ull_idx];
    auto device_it = dev_allocators_.find(device_id);
    if (device_it == dev_allocators_.end()) {
      return nullptr;
    }
    return device_it->second;
  }

  virtual Status SwapIn(int index, int device_block_id, int host_block_id) override { return Status(); }

  virtual Status SwapOut(int index, int host_block_id, int device_block_id) override { return Status(); }

  virtual Status BatchSwapOut(int index, const std::vector<int>& host_block_ids,
                              const std::vector<int>& device_block_ids) override {
    if (device_block_ids.size() != host_block_ids.size()) {
      return Status(RET_RUNTIME_FAILED, "BatchSwapOut error, device and host block count mismatch.");
    }
    return Status();
  }

  virtual Status BatchSwapIn(int index, const std::vector<int>& device_block_ids,
                             const std::vector<int>& host_block_ids) override {
    if (device_block_ids.size() != host_block_ids.size()) {
      return Status(RET_RUNTIME_FAILED, "BatchSwapIn error, device and host block count mismatch.");
    }
    return Status();
  }

  virtual const BlockAllocatorGroupConfig& GetBlockAllocatorGroupConfig() const override {
    return block_allocator_group_config_;
  }

 private:
  void Initialize() {
    host_allocator_ =
        block_allocator_creation_fn_(MemoryLocation::LOCATION_HOST, block_allocator_group_config_.host_block_num,
                                     block_allocator_group_config_.block_size,
                                     /*rank=*/0, memory_allocator_, context_);
    host_allocator_->PreAllocateBlocks();
    for (int device_id : block_allocator_group_config_.devices) {
      dev_allocators_[device_id] = block_allocator_creation_fn_(
          MemoryLocation::LOCATION_DEVICE, block_allocator_group_config_.device_block_num,
          block_allocator_group_config_.block_size, device_id, memory_allocator_, context_);
      dev_allocators_[device_id]->PreAllocateBlocks();
    }
  }

 private:
  BlockAllocatorGroupConfig block_allocator_group_config_;
  std::shared_ptr<BlockAllocatorInterface> host_allocator_ = nullptr;
  std::unordered_map<int, std::shared_ptr<BlockAllocatorInterface>> dev_allocators_;
  std::shared_ptr<MemoryAllocatorInterface> memory_allocator_ = nullptr;
  std::shared_ptr<Context> context_ = nullptr;
  BlockAllocatorCreationFunc block_allocator_creation_fn_ = nullptr;
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

  size_t GetVocabSize() { return vocab_size; }

 private:
  size_t vocab_size = 32000;
};

}  // namespace ksana_llm
