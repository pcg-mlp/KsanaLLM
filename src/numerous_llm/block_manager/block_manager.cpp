/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/block_manager/block_manager.h"

#include <memory>
#include <string>

#include "ATen/core/interned_strings.h"
#include "numerous_llm/block_manager/host_allocator.h"
#include "numerous_llm/block_manager/nvidia_allocator.h"
#include "numerous_llm/utils/logger.h"
#include "numerous_llm/utils/status.h"

namespace numerous_llm {

BlockManager::BlockManager(const BlockManagerConfig& block_manager_config, std::shared_ptr<Context> context)
    : block_manager_config_(block_manager_config), context_(context) {
  NLLM_CHECK_WITH_INFO(
      block_manager_config.device_allocator_config.block_size == block_manager_config.cpu_allocator_config.block_size,
      "The block size of host and device must be equal.");
  // Create host allocator
  host_allocator_ = std::make_shared<HostAllocator>(block_manager_config.cpu_allocator_config, context);

  // Create device allocator for every device.
  for (int worker_id = 0; worker_id < context_->GetTensorParallelSize(); ++worker_id) {
    std::shared_ptr<NvidiaDeviceAllocator> device_allocator =
        std::make_shared<NvidiaDeviceAllocator>(block_manager_config.device_allocator_config, context, worker_id);
    device_allocators_.push_back(device_allocator);
  }
}

void BlockManager::SetDeviceId(int device_id) { CUDA_CHECK(cudaSetDevice(device_id)); }

int BlockManager::GetDeviceId() {
  int device_id;
  CUDA_CHECK(cudaGetDevice(&device_id));
  return device_id;
}

std::shared_ptr<DeviceAllocator>& BlockManager::GetDeviceAllocator() {
  int device_id = GetDeviceId();
  NLLM_CHECK_WITH_INFO(device_id < device_allocators_.size(), "Invalid device id " + std::to_string(device_id));
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

Status BlockManager::GetBlockPtrs(const std::vector<int>& blocks, std::vector<void*>& addrs) {
  return GetDeviceAllocator()->GetBlockPtrs(blocks, addrs);
}

Status BlockManager::GetContiguousPtr(int block_id, void*& addr) {
  return GetDeviceAllocator()->GetContiguousPtr(block_id, addr);
}

int BlockManager::GetFreeBlockNumber() { return GetDeviceAllocator()->GetFreeBlockNumber(); }

int BlockManager::GetUsedBlockNumber() { return GetDeviceAllocator()->GetUsedBlockNumber(); }

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

int BlockManager::GetHostFreeBlockNumber() { return GetHostAllocator()->GetFreeBlockNumber(); }

int BlockManager::GetHostUsedBlockNumber() { return GetHostAllocator()->GetUsedBlockNumber(); }

Status BlockManager::SwapOut(const std::vector<int>& device_blocks, std::vector<int>& host_blocks) {
  // Allocate memory on host.
  STATUS_CHECK_RETURN(host_allocator_->AllocateBlocks(device_blocks.size(), host_blocks));

  // Get host and device address.
  std::vector<void*> host_addrs;
  STATUS_CHECK_RETURN(host_allocator_->GetBlockPtrs(host_blocks, host_addrs));

  int device_id = GetDeviceId();
  int block_size = block_manager_config_.device_allocator_config.block_size;

  std::vector<void*> device_addrs;
  STATUS_CHECK_RETURN(device_allocators_[device_id]->GetBlockPtrs(device_blocks, device_addrs));

  cudaStream_t* stream;
  if (context_->IsRunContextDecodeAndDecodeSerially()) {
    stream = &(context_->GetComputeStreams()[device_id]);
  } else {
    // TODO(karlluo): implement multiple thread stream event concurrent.
    throw std::runtime_error("Context decode and decode run in concurrently is unimplemented.");
  }

  // Copy from device to host.
  for (size_t i = 0; i < device_blocks.size(); i++) {
    CUDA_CHECK(cudaMemcpyAsync(host_addrs[i], device_addrs[i], block_size, cudaMemcpyDeviceToHost, (*stream)));
  }

  if (!context_->IsRunContextDecodeAndDecodeSerially()) {
    // TODO(karlluo): implement multiple thread stream event concurrent.
    throw std::runtime_error("Context decode and decode run in concurrently is unimplemented.");
  }

  // Free device blocks.
  device_allocators_[device_id]->FreeBlocks(device_blocks);
  return Status();
}

Status BlockManager::SwapIn(const std::vector<int>& host_blocks, std::vector<int>& device_blocks) {
  int device_id = GetDeviceId();
  int block_size = block_manager_config_.device_allocator_config.block_size;

  // Allocate memory on device.
  STATUS_CHECK_RETURN(device_allocators_[device_id]->AllocateBlocks(host_blocks.size(), device_blocks));

  std::vector<void*> device_addrs;
  STATUS_CHECK_RETURN(GetBlockPtrs(device_blocks, device_addrs));

  std::vector<void*> host_addrs;
  STATUS_CHECK_RETURN(host_allocator_->GetBlockPtrs(host_blocks, host_addrs));

  cudaStream_t* stream;
  if (context_->IsRunContextDecodeAndDecodeSerially()) {
    stream = &(context_->GetComputeStreams()[device_id]);
  } else {
    // TODO(karlluo): implement multiple thread stream event concurrent.
    throw std::runtime_error("Context decode and decode run in concurrently is unimplemented.");
  }

  // Copy from host to device.
  for (size_t i = 0; i < host_blocks.size(); i++) {
    CUDA_CHECK(cudaMemcpyAsync(device_addrs[i], host_addrs[i], block_size, cudaMemcpyHostToDevice, (*stream)));
  }

  if (!context_->IsRunContextDecodeAndDecodeSerially()) {
    // TODO(karlluo): implement multiple thread stream event concurrent.
    throw std::runtime_error("Context decode and decode run in concurrently is unimplemented.");
  }
  // Free host blocks.
  host_allocator_->FreeBlocks(host_blocks);
  return Status();
}

Status BlockManager::SwapDrop(const std::vector<int>& host_blocks) { return host_allocator_->FreeBlocks(host_blocks); }

size_t BlockManager::GetBlockSize() const { return block_manager_config_.device_allocator_config.block_size; }

size_t BlockManager::GetBlockTokenNum() const { return block_manager_config_.device_allocator_config.block_token_num; }

}  // namespace numerous_llm
