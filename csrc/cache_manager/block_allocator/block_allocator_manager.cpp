/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "cache_manager/block_allocator/block_allocator_manager.h"
#include <thread>

#include "cache_manager/block_allocator/engine_block_allocator.h"
#include "cache_manager/block_allocator/executor_block_allocator.h"
#include "device/device_types.h"
#include "fmt/format.h"
#include "ipc/ipc_func_wrapper.h"
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

void BlockAllocatorGroup::Initialize() {
  host_allocator = block_allocator_creation_fn_(
      MemoryLocation::LOCATION_HOST, block_allocator_group_config_.host_block_num,
      block_allocator_group_config_.block_size, /*device_id*/ 0, memory_allocator_, context_);
  for (int device_id : block_allocator_group_config_.devices) {
    dev_allocators[device_id] =
        block_allocator_creation_fn_(MemoryLocation::LOCATION_DEVICE, block_allocator_group_config_.device_block_num,
                                     block_allocator_group_config_.block_size, device_id, memory_allocator_, context_);
  }

  std::vector<std::thread> threads;
  threads.reserve(block_allocator_group_config_.devices.size() + 1);

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

Status BlockAllocatorGroup::SwapIn(int index, int device_block_id, int host_block_id) {
  std::shared_ptr<BlockAllocatorInterface> host_allocator = GetHostBlockAllocator();
  std::shared_ptr<BlockAllocatorInterface> device_allocator = GetDeviceBlockAllocator(index);

  if (device_allocator == nullptr) {
    return Status(RET_RUNTIME_FAILED, fmt::format("Swapin error, device index {} not found.", index));
  }

  int device_id = block_allocator_group_config_.devices[index];
  SetDevice(device_id);

  // Get host and device address.
  std::vector<void*> host_addrs(1, nullptr);
  STATUS_CHECK_FAILURE(host_allocator->GetBlockPtrs({host_block_id}, host_addrs));

  std::vector<void*> device_addrs(1, nullptr);
  STATUS_CHECK_FAILURE(device_allocator->GetBlockPtrs({device_block_id}, device_addrs));

  // Copy from host to device.
  Stream& stream = context_->device->GetH2DStream();
  int block_size = block_allocator_group_config_.block_size;
  memory_allocator_->MemcpyAsync(device_addrs[0], host_addrs[0], block_size, MEMCPY_HOST_TO_DEVICE, stream);
  StreamSynchronize(stream);

  return Status();
}

Status BlockAllocatorGroup::SwapOut(int index, int host_block_id, int device_block_id) {
  std::shared_ptr<BlockAllocatorInterface> host_allocator = GetHostBlockAllocator();
  std::shared_ptr<BlockAllocatorInterface> device_allocator = GetDeviceBlockAllocator(index);

  if (device_allocator == nullptr) {
    return Status(RET_RUNTIME_FAILED, fmt::format("Swapout error, device index {} not found.", index));
  }

  int device_id = block_allocator_group_config_.devices[index];
  SetDevice(device_id);

  // Get host and device address.
  std::vector<void*> host_addrs(1, nullptr);
  STATUS_CHECK_FAILURE(host_allocator->GetBlockPtrs({host_block_id}, host_addrs));

  std::vector<void*> device_addrs(1, nullptr);
  STATUS_CHECK_FAILURE(device_allocator->GetBlockPtrs({device_block_id}, device_addrs));

  // Copy from device to host.
  Stream& stream = context_->device->GetD2HStream();
  int block_size = block_allocator_group_config_.block_size;
  memory_allocator_->MemcpyAsync(host_addrs[0], device_addrs[0], block_size, MEMCPY_DEVICE_TO_HOST, stream);
  StreamSynchronize(stream);

  return Status();
}

Status BlockAllocatorGroup::BatchSwapOut(int index, const std::vector<int>& host_block_ids,
                                         const std::vector<int>& device_block_ids) {
  if (device_block_ids.size() != host_block_ids.size()) {
    return Status(RET_RUNTIME_FAILED, "BatchSwapOut error, device and host block count mismatch.");
  }

  // If IPC is enabled, use IPC to send request to executor
  if (ipc_func_wrapper_) {
    return BatchSwapViaIpc(index, /*is_swapin=*/false, device_block_ids, host_block_ids);
  }

  // Local execution
  for (size_t i = 0; i < device_block_ids.size(); ++i) {
    Status status = SwapOut(index, host_block_ids[i], device_block_ids[i]);
    if (!status.OK()) {
      return status;
    }
  }

  return Status();
}

Status BlockAllocatorGroup::BatchSwapIn(int index, const std::vector<int>& device_block_ids,
                                        const std::vector<int>& host_block_ids) {
  if (device_block_ids.size() != host_block_ids.size()) {
    return Status(RET_RUNTIME_FAILED, "BatchSwapIn error, device and host block count mismatch.");
  }

  // If IPC is enabled, use IPC to send request to executor
  if (ipc_func_wrapper_) {
    return BatchSwapViaIpc(index, /*is_swapin=*/true, device_block_ids, host_block_ids);
  }

  // Local execution
  for (size_t i = 0; i < device_block_ids.size(); ++i) {
    Status status = SwapIn(index, device_block_ids[i], host_block_ids[i]);
    if (!status.OK()) {
      return status;
    }
  }

  return Status();
}

void BlockAllocatorGroup::SetIpcFuncWrapper(std::shared_ptr<IpcFuncWrapper> ipc_func_wrapper) {
  ipc_func_wrapper_ = ipc_func_wrapper;
}

Status BlockAllocatorGroup::BatchSwapViaIpc(int index, bool is_swapin, const std::vector<int>& device_block_ids,
                                            const std::vector<int>& host_block_ids) {
  int executor_id = block_allocator_group_config_.devices[index];
  KLLM_LOG_SCHEDULER << "BatchSwapViaIpc, index:" << index << ", is_swapin:" << is_swapin
                << ", device_block_ids:" << device_block_ids << ", host_block_ids:" << host_block_ids;
  if (!ipc_func_wrapper_) {
    return Status(RET_RUNTIME_FAILED, "BatchSwapViaIpc error, IPC not configured.");
  }

  // IpcFuncWrapper::BatchSwap blocks until the swap completes on executor side.
  return ipc_func_wrapper_->BatchSwap(executor_id, is_swapin, device_block_ids, host_block_ids);
}

const BlockAllocatorGroupConfig& BlockAllocatorGroup::GetBlockAllocatorGroupConfig() const {
  return block_allocator_group_config_;
}

BlockAllocatorManager::BlockAllocatorManager(bool is_engine, std::shared_ptr<MemoryAllocatorInterface> memory_allocator,
                                             std::shared_ptr<Context> context,
                                             BlockAllocatorCreationFunc block_allocator_creation_fn,
                                             BlockAllocatorGroupCreationFunc block_allocator_group_creation_fn,
                                             std::shared_ptr<IpcFuncWrapper> ipc_func_wrapper) {
  memory_allocator_ = memory_allocator;
  context_ = context;
  ipc_func_wrapper_ = ipc_func_wrapper;

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
                                             BlockAllocatorGroupCreationFunc block_allocator_group_creation_fn,
                                             std::shared_ptr<IpcFuncWrapper> ipc_func_wrapper)
    : BlockAllocatorManager(is_engine, memory_allocator, context, block_allocator_creation_fn,
                            block_allocator_group_creation_fn, ipc_func_wrapper) {
  block_allocator_manager_config_ = block_allocator_manager_config;
  Initialize();
}

void BlockAllocatorManager::Initialize(const BlockAllocatorManagerConfig& block_allocator_manager_config) {
  block_allocator_manager_config_ = block_allocator_manager_config;
  Initialize();
}

void BlockAllocatorManager::Initialize() {
  for (const auto& [group_id, group_config] : block_allocator_manager_config_) {
    std::shared_ptr<BlockAllocatorGroupInterface> block_allocator_group =
        block_allocator_group_creation_fn_(group_config, memory_allocator_, context_, block_allocator_creation_fn_);

    // Set IPC wrapper on engine-side BlockAllocatorGroup for BatchSwap operations
    if (ipc_func_wrapper_) {
      auto concrete_group = std::dynamic_pointer_cast<BlockAllocatorGroup>(block_allocator_group);
      if (concrete_group) {
        concrete_group->SetIpcFuncWrapper(ipc_func_wrapper_);
      }
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
