/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/block_manager/block_manager.h"

#include <memory>
#include <string>

#include "ATen/core/interned_strings.h"
#include "ksana_llm/block_manager/device_allocator.h"
#include "ksana_llm/block_manager/host_allocator.h"
#include "ksana_llm/utils/common_device.h"
#include "ksana_llm/utils/device_utils.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/memory_utils.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/string_utils.h"

namespace ksana_llm {

BlockManager::BlockManager(const BlockManagerConfig& block_manager_config, std::shared_ptr<Context> context)
    : block_manager_config_(block_manager_config), context_(context) {
  KLLM_CHECK_WITH_INFO(
      block_manager_config.device_allocator_config.block_size == block_manager_config.host_allocator_config.block_size,
      FormatStr("The block size of host and device must be equal, %d vs %d",
                block_manager_config.device_allocator_config.block_size,
                block_manager_config.host_allocator_config.block_size));
  // Create host allocator
  host_allocator_ = std::make_shared<HostAllocator>(block_manager_config.host_allocator_config, context);

  // Create device allocator for every device.
  for (int worker_id = 0; worker_id < context_->GetTensorParallelSize(); ++worker_id) {
    std::shared_ptr<DeviceAllocator> device_allocator =
        std::make_shared<DeviceAllocator>(block_manager_config.device_allocator_config, context, worker_id);
    device_allocators_.push_back(device_allocator);
  }

  // Create device workspace buffer
  workspace_metas_.resize(context_->GetTensorParallelSize());
}

BlockManager::~BlockManager() { KLLM_LOG_DEBUG << "BlockManager destroyed."; }

Status BlockManager::PreAllocateBlocks() {
  host_allocator_->ResetPreAllocatedBlocks(block_manager_config_.host_allocator_config.blocks_num);
  for (auto& allocator : device_allocators_) {
    allocator->ResetPreAllocatedBlocks(block_manager_config_.device_allocator_config.blocks_num);
  }

  return Status();
}

Status BlockManager::ResetPreAllocatedBlocks() {
  size_t device_blocks_num;
  size_t host_block_num;

  if (context_->IsStandalone()) {
    Status status = GetBlockNumber(device_blocks_num, host_block_num);
    if (!status.OK()) {
      KLLM_LOG_ERROR << "Calculate block num error.";
      return status;
    }
  } else {
    // Get block number from pipeline config if in distributed mode.
    PipelineConfig pipeline_config;
    Singleton<Environment>::GetInstance()->GetPipelineConfig(pipeline_config);

    device_blocks_num = pipeline_config.device_block_num;
    host_block_num = pipeline_config.host_block_num;
  }

  KLLM_LOG_INFO << "Reset device_blocks_num:" << device_blocks_num << ", host_block_num:" << host_block_num;

  KLLM_LOG_INFO << "Start to preallocate host blocks.";
  host_allocator_->ResetPreAllocatedBlocks(host_block_num);
  KLLM_LOG_INFO << "Finish to preallocate host blocks.";

  for (auto& allocator : device_allocators_) {
    KLLM_LOG_INFO << "Start to preallocate device blocks on " << allocator->GetDeviceId();
#ifdef ENABLE_ACL
    int default_device_id = GetDeviceId();

    if (device_allocators_.size() > 1) {
      SetDeviceId(allocator->GetDeviceId());
    }
#endif
    allocator->ResetPreAllocatedBlocks(device_blocks_num);
#ifdef ENABLE_ACL
    if (device_allocators_.size() > 1) {
      SetDeviceId(default_device_id);
    }
#endif
    KLLM_LOG_INFO << "Finish to preallocate device blocks on " << allocator->GetDeviceId();
  }

  KLLM_LOG_INFO << "Reset block num finish.";
  return Status();
}

Status BlockManager::GetBlockNumber(size_t& device_blocks_num, size_t& host_block_num) {
  size_t host_total, host_free;
  size_t device_total, device_free;

  Status status =
      GetDeviceMemoryInfo(block_manager_config_.device_allocator_config.device, &device_free, &device_total);
  if (!status.OK()) {
    return status;
  }

  status = GetHostMemoryInfo(&host_free, &host_total);
  if (!status.OK()) {
    return status;
  }

  KLLM_LOG_INFO << "Get memory info, host_total:" << host_total << ", host_free:" << host_free
                << ", device_total:" << device_total << ", device_free:" << device_free
                << ", block_device_memory_ratio:" << block_manager_config_.block_device_memory_ratio
                << ", reserved_device_memory_ratio:" << block_manager_config_.reserved_device_memory_ratio
                << ", block_host_memory_factor:" << block_manager_config_.block_host_memory_factor;

  KLLM_CHECK_WITH_INFO(block_manager_config_.reserved_device_memory_ratio > 0.0,
                       "reserved_device_memory_ratio must be large than 0.0");
  KLLM_CHECK_WITH_INFO(block_manager_config_.lora_host_memory_factor >= 1.0, "lora_host_memory_factor should >= 1.0");
  KLLM_CHECK_WITH_INFO(block_manager_config_.block_host_memory_factor >= 0.0, "block_host_memory_factor should >= 0.0");

  const size_t alignment_bytes = 8;
  size_t device_block_memory_size = 0;
  if (block_manager_config_.block_device_memory_ratio >= 0.0) {
    device_block_memory_size =
        DivRoundDown(std::min((static_cast<size_t>(device_total * block_manager_config_.block_device_memory_ratio)),
                              device_free),
                     alignment_bytes) *
        alignment_bytes;
  } else {
    size_t reserved_memory_size =
        DivRoundUp((device_total * block_manager_config_.reserved_device_memory_ratio), alignment_bytes) *
        alignment_bytes;
    device_block_memory_size =
        DivRoundDown((reserved_memory_size < device_free ? device_free - reserved_memory_size : 0ul), alignment_bytes) *
        alignment_bytes;
  }

  const float block_host_memory_ratio = 0.8;
  size_t host_block_memory_size =
      DivRoundDown(
          static_cast<size_t>(std::min(device_block_memory_size * block_manager_config_.block_host_memory_factor,
                                       host_free * block_host_memory_ratio)),
          alignment_bytes) *
      alignment_bytes;

  KLLM_LOG_INFO << "Get block memory info, host_free:" << host_block_memory_size
                << ", device_free:" << device_block_memory_size
                << ", block_size:" << block_manager_config_.host_allocator_config.block_size;

  device_blocks_num = device_block_memory_size / block_manager_config_.device_allocator_config.block_size;
  host_block_num = host_block_memory_size / block_manager_config_.host_allocator_config.block_size;

  return Status();
}

void BlockManager::SetDeviceId(int device_id) { SetDevice(device_id); }

DataType BlockManager::GetDtype() { return block_manager_config_.device_allocator_config.kv_cache_dtype; }

int BlockManager::GetDeviceId() {
  int device_id = -1;
  GetDevice(&device_id);
  return device_id;
}

std::shared_ptr<DeviceAllocator>& BlockManager::GetDeviceAllocator() {
  size_t device_id = static_cast<size_t>(GetDeviceId());
  KLLM_CHECK_WITH_INFO(device_id < device_allocators_.size(), fmt::format("Invalid device id {}", device_id));
  return device_allocators_[device_id];
}

std::shared_ptr<HostAllocator>& BlockManager::GetHostAllocator() { return host_allocator_; }

Status BlockManager::AllocateBlocks(int64_t block_num, std::vector<int>& blocks) {
  return GetDeviceAllocator()->AllocateBlocks(block_num, blocks);
}

Status BlockManager::AllocateContiguous(int64_t size, int& block_id) {
  return GetDeviceAllocator()->AllocateContiguous(size, block_id);
}

Status BlockManager::FreeBlocks(const std::vector<int>& blocks) { return GetDeviceAllocator()->FreeBlocks(blocks); }

Status BlockManager::FreeContiguous(int block_id) { return GetDeviceAllocator()->FreeContiguous(block_id); }

bool BlockManager::IsContiguousUsed(const int block_id) { return GetDeviceAllocator()->IsContiguousUsed(block_id); }

Status BlockManager::GetBlockPtrs(const std::vector<int>& blocks, std::vector<void*>& addrs) {
  return GetDeviceAllocator()->GetBlockPtrs(blocks, addrs);
}

Status BlockManager::GetContiguousPtr(int block_id, void*& addr) {
  return GetDeviceAllocator()->GetContiguousPtr(block_id, addr);
}

size_t BlockManager::GetDeviceFreeBlockNumber() { return GetDeviceAllocator()->GetFreeBlockNumber(); }

size_t BlockManager::GetDeviceUsedBlockNumber() { return GetDeviceAllocator()->GetUsedBlockNumber(); }

Status BlockManager::AllocateHostBlocks(int64_t block_num, std::vector<int>& blocks) {
  return GetHostAllocator()->AllocateBlocks(block_num, blocks);
}

Status BlockManager::AllocateHostContiguous(int64_t size, int& block_id) {
  return GetHostAllocator()->AllocateContiguous(size, block_id);
}

Status BlockManager::FreeHostBlocks(const std::vector<int>& blocks) { return GetHostAllocator()->FreeBlocks(blocks); }

Status BlockManager::FreeHostContiguous(int block_id) { return GetHostAllocator()->FreeContiguous(block_id); }

Status BlockManager::GetHostBlockPtrs(const std::vector<int>& blocks, std::vector<void*>& addrs) {
  return GetHostAllocator()->GetBlockPtrs(blocks, addrs);
}

Status BlockManager::GetHostContiguousPtr(int block_id, void*& addr) {
  return GetHostAllocator()->GetContiguousPtr(block_id, addr);
}

size_t BlockManager::GetHostFreeBlockNumber() {
  return GetHostAllocator()->GetFreeBlockNumber() / context_->GetTensorParallelSize();
}

size_t BlockManager::GetHostUsedBlockNumber() {
  return GetHostAllocator()->GetUsedBlockNumber() / context_->GetTensorParallelSize();
}

Status BlockManager::SwapOut(int host_block_id, int device_block_id) {
  // Get host and device address.
  std::vector<void*> host_addrs(1, nullptr);
  STATUS_CHECK_FAILURE(host_allocator_->GetBlockPtrs({host_block_id}, host_addrs));

  int device_id = GetDeviceId();
  std::vector<void*> device_addrs(1, nullptr);
  STATUS_CHECK_FAILURE(device_allocators_[device_id]->GetBlockPtrs({device_block_id}, device_addrs));

  // Copy from device to host.
  Stream& stream = context_->GetD2HStreams()[device_id];
  int block_size = block_manager_config_.device_allocator_config.block_size;
  MemcpyAsync(host_addrs[0], device_addrs[0], block_size, MEMCPY_DEVICE_TO_HOST, stream);
  StreamSynchronize(stream);

  return Status();
}

Status BlockManager::SwapIn(int device_block_id, int host_block_id) {
  // Get host and device address.
  std::vector<void*> host_addrs(1, nullptr);
  STATUS_CHECK_FAILURE(host_allocator_->GetBlockPtrs({host_block_id}, host_addrs));

  int device_id = GetDeviceId();
  std::vector<void*> device_addrs(1, nullptr);
  STATUS_CHECK_FAILURE(device_allocators_[device_id]->GetBlockPtrs({device_block_id}, device_addrs));

  // Copy from host to device.
  Stream& stream = context_->GetH2DStreams()[device_id];
  int block_size = block_manager_config_.device_allocator_config.block_size;
  MemcpyAsync(device_addrs[0], host_addrs[0], block_size, MEMCPY_HOST_TO_DEVICE, stream);
  StreamSynchronize(stream);

  return Status();
}

Status BlockManager::SwapDrop(const std::vector<int>& host_blocks) { return host_allocator_->FreeBlocks(host_blocks); }

size_t BlockManager::GetBlockSize() const { return block_manager_config_.device_allocator_config.block_size; }

size_t BlockManager::GetBlockTokenNum() const { return block_manager_config_.device_allocator_config.block_token_num; }

const BlockManagerConfig& BlockManager::GetBlockManagerConfig() const { return block_manager_config_; }

void* BlockManager::GetBlockBasePtr() {
  int device_id = GetDeviceId();
  return device_allocators_[device_id]->GetBlocksBasePtr();
}

const AllocatorConfig& BlockManager::GetAllocatorConfig() {
  int device_id = GetDeviceId();
  return device_allocators_[device_id]->GetAllocatorConfig();
}

int BlockManager::GetBlocksBaseId() {
  int device_id = GetDeviceId();
  return device_allocators_[device_id]->GetBlocksBaseId();
}

WorkspaceMeta& BlockManager::GetWorkspaceMeta() {
  int device_id = GetDeviceId();
  return workspace_metas_[device_id];
}

Status BlockManager::UpdateConfig(const BlockManagerConfig& update_block_manager_config) {
  block_manager_config_ = update_block_manager_config;
  return Status();
}

}  // namespace ksana_llm
