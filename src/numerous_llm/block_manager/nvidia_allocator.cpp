/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/block_manager/nvidia_allocator.h"
#include "numerous_llm/block_manager/device_allocator.h"
#include "numerous_llm/block_manager/memory_block.h"
#include "numerous_llm/utils/environment.h"
#include "numerous_llm/utils/status.h"

namespace numerous_llm {

NvidiaDeviceAllocator::NvidiaDeviceAllocator(const AllocatorConfig& allocator_config, std::shared_ptr<Context> context,
                                             int device_id)
    : DeviceAllocator(allocator_config, context, device_id) {
  PreAllocateBlocks();
}

void NvidiaDeviceAllocator::AllocateMemory(void** memory_ptr, size_t bytes) {
  CUDA_CHECK(cudaMallocAsync(memory_ptr, bytes, context_->memory_manage_streams_[device_id_]));
}

void NvidiaDeviceAllocator::FreeMemory(void* memory_ptr) {
  CUDA_CHECK(cudaFreeAsync(memory_ptr, context_->memory_manage_streams_[device_id_]));
}

}  // namespace numerous_llm
