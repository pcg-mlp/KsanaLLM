/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "cache_manager/block_allocator/block_allocator_manager.h"
#include <thread>

#include "cache_manager/block_allocator/engine_block_allocator.h"
#include "cache_manager/block_allocator/executor_block_allocator.h"
#include "device/device_types.h"
#include "fmt/format.h"
#include "utils/status.h"

namespace ksana_llm {

BlockAllocatorGroup::BlockAllocatorGroup(const BlockAllocatorGroupConfig& block_allocator_group_config,
                                         std::shared_ptr<MemoryAllocatorInterface> memory_allocator,
                                         std::shared_ptr<Context> context,
                                         BlockAllocatorCreationFunc block_allocator_creation_fn) {
  block_allocator_group_config_ = block_allocator_group_config;
  memory_allocator_ = memory_allocator;
  context_ = context;
  block_allocator_creation_fn_ = block_allocator_creation_fn;

  Initialize();
}

BlockAllocatorGroup::BlockAllocatorGroup(const BlockAllocatorGroupConfig& block_allocator_group_config,
                                         std::shared_ptr<MemoryAllocatorInterface> memory_allocator,
                                         std::shared_ptr<Context> context,
                                         BlockAllocatorCreationFunc block_allocator_creation_fn,
                                         std::shared_ptr<BlockAllocatorInterface> shared_host_allocator) {
  block_allocator_group_config_ = block_allocator_group_config;
  memory_allocator_ = memory_allocator;
  context_ = context;
  block_allocator_creation_fn_ = block_allocator_creation_fn;
  host_allocator = shared_host_allocator;
  host_allocator_is_shared_ = (shared_host_allocator != nullptr);

  Initialize();
}

void BlockAllocatorGroup::Initialize() {
  if (host_allocator == nullptr) {
    host_allocator = block_allocator_creation_fn_(
        MemoryLocation::LOCATION_HOST, block_allocator_group_config_.host_block_num,
        block_allocator_group_config_.block_size, /*device_id*/ 0, memory_allocator_, context_);
  }
  for (int device_id : block_allocator_group_config_.devices) {
    dev_allocators[device_id] =
        block_allocator_creation_fn_(MemoryLocation::LOCATION_DEVICE, block_allocator_group_config_.device_block_num,
                                     block_allocator_group_config_.block_size, device_id, memory_allocator_, context_);
  }

  std::vector<std::thread> threads;
  threads.reserve(block_allocator_group_config_.devices.size() + 1);

  if (!host_allocator_is_shared_) {
    threads.emplace_back([this]() {
      KLLM_LOG_INFO << "Start to preallocate host blocks, block_num:" << block_allocator_group_config_.host_block_num
                    << ", block_size:" << block_allocator_group_config_.block_size;
#if defined(ENABLE_ACL)
      // For ascend device, clrtMallocHost will be failed if aclrtSetDevice() is not invoked.
      SetDevice(block_allocator_group_config_.devices.front());
#endif
      host_allocator->PreAllocateBlocks();
      KLLM_LOG_INFO << "Finish to preallocate host blocks.";
    });
  }
  for (int device_id : block_allocator_group_config_.devices) {
    threads.emplace_back([this, device_id]() {
      if (context_->device) {
        if (context_->device->GetDeviceId() == device_id) {
          KLLM_LOG_INFO << "Start to preallocate executor device blocks on " << device_id
                        << ", block_num:" << block_allocator_group_config_.device_block_num
                        << ", block_size:" << block_allocator_group_config_.block_size;
          dev_allocators[device_id]->PreAllocateBlocks();
          KLLM_LOG_INFO << "Finish to preallocate executor device blocks on " << device_id;
        }
      } else {
        KLLM_LOG_INFO << "Start to preallocate engine device blocks on " << device_id
                      << ", block_num:" << block_allocator_group_config_.device_block_num
                      << ", block_size:" << block_allocator_group_config_.block_size;
        dev_allocators[device_id]->PreAllocateBlocks();
        KLLM_LOG_INFO << "Finish to preallocate engine device blocks on " << device_id;
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }
}

std::vector<int> BlockAllocatorGroup::GetBlockAllocatorDevices() const { return block_allocator_group_config_.devices; }

std::shared_ptr<BlockAllocatorInterface> BlockAllocatorGroup::GetHostBlockAllocator() const { return host_allocator; }

std::shared_ptr<BlockAllocatorInterface> BlockAllocatorGroup::GetDeviceBlockAllocator(int index) const {
  size_t ull_idx = static_cast<size_t>(index);
  if (ull_idx >= block_allocator_group_config_.devices.size()) {
    return nullptr;
  }

  int device_id = block_allocator_group_config_.devices[ull_idx];
  auto device_it = dev_allocators.find(device_id);
  if (device_it == dev_allocators.end()) {
    return nullptr;
  }
  return device_it->second;
}

const BlockAllocatorGroupConfig& BlockAllocatorGroup::GetBlockAllocatorGroupConfig() const {
  return block_allocator_group_config_;
}

BlockAllocatorManager::BlockAllocatorManager(bool is_engine, std::shared_ptr<MemoryAllocatorInterface> memory_allocator,
                                             std::shared_ptr<Context> context,
                                             BlockAllocatorCreationFunc block_allocator_creation_fn,
                                             BlockAllocatorGroupCreationFunc block_allocator_group_creation_fn) {
  memory_allocator_ = memory_allocator;
  context_ = context;

  block_allocator_creation_fn_ = block_allocator_creation_fn;
  if (block_allocator_creation_fn_ == nullptr) {
    block_allocator_creation_fn_ = [is_engine](
                                       MemoryLocation location, size_t block_num, size_t block_size, int rank,
                                       std::shared_ptr<MemoryAllocatorInterface> memory_allocator,
                                       std::shared_ptr<Context> context) -> std::shared_ptr<BlockAllocatorInterface> {
      if (is_engine) {
        return std::make_shared<EngineBlockAllocator>(block_num);
      } else {
        // return nullptr if not current device.
        if (location == MemoryLocation::LOCATION_DEVICE && context->device->GetDeviceId() != rank) {
          return nullptr;
        }
        return std::make_shared<ExecutorBlockAllocator>(location, block_num, block_size, rank, memory_allocator,
                                                        context);
      }
    };
  }

  is_default_group_factory_ = (block_allocator_group_creation_fn == nullptr);
  block_allocator_group_creation_fn_ = block_allocator_group_creation_fn;
  if (block_allocator_group_creation_fn_ == nullptr) {
    block_allocator_group_creation_fn_ = [](const BlockAllocatorGroupConfig& block_allocator_group_config,
                                            std::shared_ptr<MemoryAllocatorInterface> memory_allocator,
                                            std::shared_ptr<Context> context,
                                            BlockAllocatorCreationFunc block_allocator_creation_fn) {
      return std::make_shared<BlockAllocatorGroup>(block_allocator_group_config, memory_allocator, context,
                                                   block_allocator_creation_fn);
    };
  }
}

BlockAllocatorManager::BlockAllocatorManager(bool is_engine,
                                             const BlockAllocatorManagerConfig& block_allocator_manager_config,
                                             std::shared_ptr<MemoryAllocatorInterface> memory_allocator,
                                             std::shared_ptr<Context> context,
                                             BlockAllocatorCreationFunc block_allocator_creation_fn,
                                             BlockAllocatorGroupCreationFunc block_allocator_group_creation_fn)
    : BlockAllocatorManager(is_engine, memory_allocator, context, block_allocator_creation_fn,
                            block_allocator_group_creation_fn) {
  block_allocator_manager_config_ = block_allocator_manager_config;
  Initialize();
}

void BlockAllocatorManager::Initialize(const BlockAllocatorManagerConfig& block_allocator_manager_config) {
  block_allocator_manager_config_ = block_allocator_manager_config;
  Initialize();
}

void BlockAllocatorManager::Initialize() {
  // Detect whether all group configs agree on (host_block_num, block_size).
  // Every executor process serves a single device yet gets the full topology
  // config (one entry per attn-DP group) from inference_engine.cpp, all carrying
  // the same total host_block_num. Without sharing, each group's
  // PreAllocateBlocks duplicates the entire host pool inside this one process
  // (8× on attn_dp=8). When configs match, collapse to one shared host allocator
  // + one PreAllocateBlocks call. Sharing is intra-process only; the separate
  // card processes still each hold their own host pool.
  bool host_uniform = false;
  size_t shared_host_block_num = 0;
  size_t shared_block_size = 0;
  if (is_default_group_factory_ && !block_allocator_manager_config_.empty()) {
    host_uniform = true;
    auto it = block_allocator_manager_config_.begin();
    shared_host_block_num = it->second.host_block_num;
    shared_block_size = it->second.block_size;
    for (++it; it != block_allocator_manager_config_.end(); ++it) {
      if (it->second.host_block_num != shared_host_block_num || it->second.block_size != shared_block_size) {
        host_uniform = false;
        break;
      }
    }
  }

  if (host_uniform) {
    shared_host_allocator_ =
        block_allocator_creation_fn_(MemoryLocation::LOCATION_HOST, shared_host_block_num, shared_block_size,
                                     /*rank*/ 0, memory_allocator_, context_);
    KLLM_LOG_INFO << "Start to preallocate shared host blocks, block_num:" << shared_host_block_num
                  << ", block_size:" << shared_block_size << ", groups:" << block_allocator_manager_config_.size();
    shared_host_allocator_->PreAllocateBlocks();
    KLLM_LOG_INFO << "Finish to preallocate shared host blocks.";
  }

  for (const auto& [group_id, group_config] : block_allocator_manager_config_) {
    std::shared_ptr<BlockAllocatorGroupInterface> block_allocator_group;
    if (shared_host_allocator_ != nullptr) {
      // Default-factory uniform path: bypass the typedef and use the shared
      // host directly. Test paths with custom factory still go through the
      // legacy lambda below.
      block_allocator_group = std::make_shared<BlockAllocatorGroup>(
          group_config, memory_allocator_, context_, block_allocator_creation_fn_, shared_host_allocator_);
    } else {
      block_allocator_group =
          block_allocator_group_creation_fn_(group_config, memory_allocator_, context_, block_allocator_creation_fn_);
    }

    allocator_groups_[group_id] = block_allocator_group;
  }
}

std::shared_ptr<BlockAllocatorGroupInterface> BlockAllocatorManager::GetBlockAllocatorGroup(int group_id) {
  auto group_it = allocator_groups_.find(group_id);
  if (group_it == allocator_groups_.end()) {
    return nullptr;
  }

  return group_it->second;
}

}  // namespace ksana_llm
