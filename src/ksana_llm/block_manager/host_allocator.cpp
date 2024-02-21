/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/block_manager/host_allocator.h"
#include "ksana_llm/block_manager/base_allocator.h"

namespace ksana_llm {

HostAllocator::HostAllocator(const AllocatorConfig& allocator_config, std::shared_ptr<Context> context)
    : BaseAllocator(allocator_config, context) {}

void HostAllocator::AllocateMemory(void** memory_ptr, size_t bytes) {
  CUDA_CHECK(cudaHostAlloc(memory_ptr, bytes, cudaHostAllocDefault));
}

void HostAllocator::FreeMemory(void* memory_ptr) {
  CUDA_CHECK(cudaFreeAsync(memory_ptr, context_->GetH2DStreams()[0]));
}

}  // namespace ksana_llm
