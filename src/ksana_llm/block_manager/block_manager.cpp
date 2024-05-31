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
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/string_utils.h"

namespace ksana_llm {

BlockManager::BlockManager(const BlockManagerConfig& block_manager_config, std::shared_ptr<Context> context)
    : block_manager_config_(block_manager_config), context_(context) {
  NLLM_CHECK_WITH_INFO(
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
}

Status BlockManager::PreAllocateBlocks() {
  host_allocator_->ResetPreAllocatedBlocks(block_manager_config_.host_allocator_config.blocks_num);
  for (auto& allocator : device_allocators_) {
    allocator->ResetPreAllocatedBlocks(block_manager_config_.device_allocator_config.blocks_num);
  }

  return Status();
}

Status BlockManager::ResetPreAllocatedBlocks() {
  size_t host_block_num;
  size_t device_blocks_num;

  Status status = CalculateBlockNumber(device_blocks_num, host_block_num);
  if (!status.OK()) {
    NLLM_LOG_ERROR << "Calculate block num error.";
    return status;
  }

  NLLM_LOG_INFO << "Reset device_blocks_num:" << device_blocks_num << ", host_block_num:" << host_block_num;

  NLLM_LOG_INFO << "Start to preallocate host blocks.";
  host_allocator_->ResetPreAllocatedBlocks(host_block_num);
  NLLM_LOG_INFO << "Finish to preallocate host blocks.";

  for (auto& allocator : device_allocators_) {
    NLLM_LOG_INFO << "Start to preallocate device blocks on " << allocator->GetDeviceId();
    allocator->ResetPreAllocatedBlocks(device_blocks_num);
    NLLM_LOG_INFO << "Finish to preallocate device blocks on " << allocator->GetDeviceId();
  }

  NLLM_LOG_INFO << "Reset block num finish.";
  return Status();
}

Status BlockManager::CalculateBlockNumber(size_t& device_blocks_num, size_t& host_block_num) {
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

  NLLM_LOG_INFO << "Get memory info, host_total:" << host_total << ", host_free:" << host_free
                << ", device_total:" << device_total << ", device_free:" << device_free;

  NLLM_CHECK_WITH_INFO(block_manager_config_.reserved_device_memory_ratio > 0.0,
                       "reserved_device_memory_ratio must be large than 0.0");
  NLLM_CHECK_WITH_INFO(block_manager_config_.lora_host_memory_factor >= 1.0, "lora_host_memory_factor should >= 1.0");
  NLLM_CHECK_WITH_INFO(block_manager_config_.block_host_memory_factor >= 0.0, "block_host_memory_factor should >= 0.0");

  size_t alignment_bytes = 8;
  size_t device_block_memory_size = 0;
  if (block_manager_config_.block_device_memory_ratio >= 0.0) {
    device_block_memory_size = (device_total * block_manager_config_.block_device_memory_ratio) / alignment_bytes;
  } else {
    size_t reserved_memory_size =
        ((device_total * block_manager_config_.reserved_device_memory_ratio) / alignment_bytes + 1) * alignment_bytes;
    device_block_memory_size = ((device_free - reserved_memory_size) / alignment_bytes + 1) * alignment_bytes;
  }

  device_blocks_num = device_block_memory_size / block_manager_config_.device_allocator_config.block_size;
  host_block_num = device_blocks_num * block_manager_config_.block_host_memory_factor;

  size_t host_allocate_bytes = host_block_num * block_manager_config_.host_allocator_config.block_size;
  NLLM_CHECK_WITH_INFO(host_allocate_bytes < host_free,
                       FormatStr("Not enough host free memory, expect %d, free %d", host_allocate_bytes, host_free));
#ifdef ENABLE_ACL
  device_blocks_num = 4;
  host_block_num = 4;
#endif

  return Status();
}

void BlockManager::SetDeviceId(int device_id) { SetDevice(device_id); }

int BlockManager::GetDeviceId() {
  int device_id = -1;
  GetDevice(&device_id);
  return device_id;
}

std::shared_ptr<DeviceAllocator>& BlockManager::GetDeviceAllocator() {
  size_t device_id = static_cast<size_t>(GetDeviceId());
  NLLM_CHECK_WITH_INFO(device_id < device_allocators_.size(), fmt::format("Invalid device id {}", device_id));
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

Status BlockManager::SwapOut(const std::vector<int>& device_blocks, std::vector<int>& host_blocks,
                             const int host_block_num_to_add) {
  // Allocate memory on host.
  STATUS_CHECK_FAILURE(host_allocator_->AllocateBlocks(device_blocks.size() + host_block_num_to_add, host_blocks));

  // Get host and device address.
  std::vector<void*> host_addrs;
  STATUS_CHECK_FAILURE(host_allocator_->GetBlockPtrs(host_blocks, host_addrs));

  int device_id = GetDeviceId();
  int block_size = block_manager_config_.device_allocator_config.block_size;

  std::vector<void*> device_addrs;
  STATUS_CHECK_FAILURE(device_allocators_[device_id]->GetBlockPtrs(device_blocks, device_addrs));

  // Copy from device to host.
  Stream& stream = context_->GetD2HStreams()[device_id];
  for (size_t i = 0; i < device_blocks.size(); i++) {
    MemcpyAsync(host_addrs[i], device_addrs[i], block_size, MEMCPY_DEVICE_TO_HOST, stream);
  }
  StreamSynchronize(stream);

  // Free device blocks.
  device_allocators_[device_id]->FreeBlocks(device_blocks);
  return Status();
}

Status BlockManager::SwapIn(const std::vector<int>& host_blocks, std::vector<int>& device_blocks) {
  int device_id = GetDeviceId();
  int block_size = block_manager_config_.device_allocator_config.block_size;

  // Allocate memory on device.
  STATUS_CHECK_FAILURE(device_allocators_[device_id]->AllocateBlocks(host_blocks.size(), device_blocks));

  std::vector<void*> device_addrs;
  STATUS_CHECK_FAILURE(GetBlockPtrs(device_blocks, device_addrs));

  std::vector<void*> host_addrs;
  STATUS_CHECK_FAILURE(host_allocator_->GetBlockPtrs(host_blocks, host_addrs));

  // Copy from host to device.
  Stream& stream = context_->GetH2DStreams()[device_id];
  for (size_t i = 0; i < host_blocks.size(); i++) {
    MemcpyAsync(device_addrs[i], host_addrs[i], block_size, MEMCPY_HOST_TO_DEVICE, stream);
  }
  StreamSynchronize(stream);

  // Free host blocks.
  host_allocator_->FreeBlocks(host_blocks);
  return Status();
}

Status BlockManager::SwapDrop(const std::vector<int>& host_blocks) { return host_allocator_->FreeBlocks(host_blocks); }

size_t BlockManager::GetBlockSize() const { return block_manager_config_.device_allocator_config.block_size; }

size_t BlockManager::GetBlockTokenNum() const { return block_manager_config_.device_allocator_config.block_token_num; }

Status BlockManager::PreparePrefixCacheBlocks() {
  if (block_manager_config_.prefix_cache_len == 0) {
    NLLM_LOG_DEBUG << "Disalbe prefix cache";
    return Status();
  } else if (block_manager_config_.prefix_cache_len > 0) {
    NLLM_LOG_DEBUG << "Prefix cache token number " << block_manager_config_.prefix_cache_len;
  } else if (block_manager_config_.prefix_cache_len == -1) {
    throw std::invalid_argument("Not support prefix_cache_len == -1, autofix mode");
  } else {
    throw std::invalid_argument(
        fmt::format("Not support prefix_cache_len setting {}", block_manager_config_.prefix_cache_len));
  }
  int block_num =
      block_manager_config_.prefix_cache_len / block_manager_config_.device_allocator_config.block_token_num;

  // TODO(karlluo): support pipeline parallel
  // prepare prefixed cache blocks
  for (size_t device_id = 0; device_id < context_->GetTensorParallelSize(); ++device_id) {
    SetDeviceId(device_id);
    std::vector<int> prefix_cache_block_tmp;
    AllocateBlocks(block_num, prefix_cache_block_tmp);
    prefix_cache_blocks_.emplace_back(std::move(prefix_cache_block_tmp));
  }
  prefix_cache_block_num_ = block_num;
  return Status();
}

int BlockManager::GetPrefixCacheTokensNumber() const { return block_manager_config_.prefix_cache_len; }

size_t BlockManager::GetPrefixCacheBlocksNumber() const { return prefix_cache_block_num_; }

bool BlockManager::CheckReqIsValidForPrefixCache(const std::vector<int>& input_tokens) {
  if (block_manager_config_.prefix_cache_len <= 0 ||
      input_tokens.size() < static_cast<size_t>(block_manager_config_.prefix_cache_len)) {
    return false;
  }

  if (prefix_cache_tokens_.empty()) {
    // init with the first request
    prefix_cache_tokens_.resize(block_manager_config_.prefix_cache_len, 0);
    std::copy(input_tokens.begin(), input_tokens.begin() + block_manager_config_.prefix_cache_len,
              prefix_cache_tokens_.begin());
  } else {
    for (int token_idx = 0; token_idx < block_manager_config_.prefix_cache_len; ++token_idx) {
      if (prefix_cache_tokens_[token_idx] != input_tokens[token_idx]) {
        return false;
      }
    }
  }
  return true;
}

Status BlockManager::FillPrefixCacheBlocks(std::vector<std::vector<int>>& kv_cache_blocks) {
  // TODO(karlluo): support pipeline parallel
  // prepare prefixed cache blocks
  for (int device_id = 0; device_id < context_->GetTensorParallelSize(); ++device_id) {
    if (kv_cache_blocks[device_id].size() == 0) {
      std::copy(prefix_cache_blocks_[device_id].begin(), prefix_cache_blocks_[device_id].end(),
                std::back_inserter(kv_cache_blocks[device_id]));
    } else {
      std::copy(prefix_cache_blocks_[device_id].begin(), prefix_cache_blocks_[device_id].end(),
                kv_cache_blocks[device_id].begin());
    }
  }
  return Status();
}

const BlockManagerConfig& BlockManager::GetBlockManagerConfig() const { return block_manager_config_; }

}  // namespace ksana_llm
