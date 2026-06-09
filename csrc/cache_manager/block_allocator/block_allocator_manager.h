/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "cache_manager/block_allocator/block_allocator_interface.h"
#include "configure/context.h"
#include "utils/memory_allocator_interface.h"

namespace ksana_llm {

// The config of a block allocator group.
struct BlockAllocatorGroupConfig {
  // The device ids of this group.
  std::vector<int> devices;

  // The device block num.
  size_t device_block_num;

  // The host block num.
  size_t host_block_num;

  // The block size, in bytes
  size_t block_size;
};

// The config of all block allocator groups.
using BlockAllocatorManagerConfig = std::unordered_map<int, BlockAllocatorGroupConfig>;

// Function used to create block allocator, for testing.
using BlockAllocatorCreationFunc = std::function<std::shared_ptr<BlockAllocatorInterface>(
    MemoryLocation location, size_t block_num, size_t block_size, int rank,
    std::shared_ptr<MemoryAllocatorInterface> memory_allocator, std::shared_ptr<Context> context)>;

//
class BlockAllocatorGroupInterface {
 public:
  virtual ~BlockAllocatorGroupInterface() {}

  virtual std::vector<int> GetBlockAllocatorDevices() const = 0;

  virtual std::shared_ptr<BlockAllocatorInterface> GetHostBlockAllocator() const = 0;
  virtual std::shared_ptr<BlockAllocatorInterface> GetDeviceBlockAllocator(int index = 0) const = 0;

  virtual const BlockAllocatorGroupConfig& GetBlockAllocatorGroupConfig() const = 0;
};

// Function used to create block allocator group, for testing.
using BlockAllocatorGroupCreationFunc = std::function<std::shared_ptr<BlockAllocatorGroupInterface>(
    const BlockAllocatorGroupConfig& block_allocator_group_config,
    std::shared_ptr<MemoryAllocatorInterface> memory_allocator, std::shared_ptr<Context> context,
    BlockAllocatorCreationFunc block_allocator_creation_fn)>;

// A block allocator group.
class BlockAllocatorGroup : public BlockAllocatorGroupInterface {
 public:
  virtual ~BlockAllocatorGroup() {}

  BlockAllocatorGroup(const BlockAllocatorGroupConfig& block_allocator_group_config,
                      std::shared_ptr<MemoryAllocatorInterface> memory_allocator, std::shared_ptr<Context> context,
                      BlockAllocatorCreationFunc block_allocator_creation_fn);

  virtual std::vector<int> GetBlockAllocatorDevices() const override;

  virtual std::shared_ptr<BlockAllocatorInterface> GetHostBlockAllocator() const override;
  virtual std::shared_ptr<BlockAllocatorInterface> GetDeviceBlockAllocator(int index = 0) const override;

  virtual const BlockAllocatorGroupConfig& GetBlockAllocatorGroupConfig() const override;

 private:
  void Initialize();

 private:
  BlockAllocatorGroupConfig block_allocator_group_config_;

  // The host allocator
  std::shared_ptr<BlockAllocatorInterface> host_allocator = nullptr;

  // All the device block allocators, the key is device_id.
  std::unordered_map<int, std::shared_ptr<BlockAllocatorInterface>> dev_allocators;

  std::shared_ptr<MemoryAllocatorInterface> memory_allocator_ = nullptr;

  std::shared_ptr<Context> context_ = nullptr;

  BlockAllocatorCreationFunc block_allocator_creation_fn_ = nullptr;
};

// Interface for BlockAllocatorManager to support dependency injection and testing.
class BlockAllocatorManagerInterface {
 public:
  virtual ~BlockAllocatorManagerInterface() {}

  // Initialize the block allocator manager with the given configuration.
  virtual void Initialize(const BlockAllocatorManagerConfig& block_allocator_manager_config) = 0;

  // Get the block allocator group by group_id.
  virtual std::shared_ptr<BlockAllocatorGroupInterface> GetBlockAllocatorGroup(int group_id) = 0;
};

// Used to manage all the block allocator.
class BlockAllocatorManager : public BlockAllocatorManagerInterface {
 public:
  BlockAllocatorManager(bool is_engine, std::shared_ptr<MemoryAllocatorInterface> memory_allocator,
                        std::shared_ptr<Context> context,
                        BlockAllocatorCreationFunc block_allocator_creation_fn = nullptr,
                        BlockAllocatorGroupCreationFunc block_allocator_group_creation_fn = nullptr);
  virtual void Initialize(const BlockAllocatorManagerConfig& block_allocator_manager_config) override;

  BlockAllocatorManager(bool is_engine, const BlockAllocatorManagerConfig& block_allocator_manager_config,
                        std::shared_ptr<MemoryAllocatorInterface> memory_allocator, std::shared_ptr<Context> context,
                        BlockAllocatorCreationFunc block_allocator_creation_fn = nullptr,
                        BlockAllocatorGroupCreationFunc block_allocator_group_creation_fn = nullptr);

  virtual std::shared_ptr<BlockAllocatorGroupInterface> GetBlockAllocatorGroup(int group_id) override;

 private:
  void Initialize();

 private:
  BlockAllocatorManagerConfig block_allocator_manager_config_;

  // The key is group_id, for example, every dp is a group.
  std::unordered_map<int, std::shared_ptr<BlockAllocatorGroupInterface>> allocator_groups_;

  std::shared_ptr<MemoryAllocatorInterface> memory_allocator_ = nullptr;

  std::shared_ptr<Context> context_ = nullptr;

  BlockAllocatorCreationFunc block_allocator_creation_fn_ = nullptr;
  BlockAllocatorGroupCreationFunc block_allocator_group_creation_fn_ = nullptr;
};

}  // namespace ksana_llm
