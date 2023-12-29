/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/block_manager/host_allocator.h"
#include "numerous_llm/block_manager/base_allocator.h"

namespace numerous_llm {

HostAllocator::HostAllocator(const AllocatorConfig& allocator_config, std::shared_ptr<Context> context)
    : BaseAllocator(allocator_config, context) {
  PreAllocateBlocks();
}

void HostAllocator::AllocateMemory(void** memory_ptr, size_t bytes) {
  CUDA_CHECK(cudaHostAlloc(memory_ptr, bytes, cudaHostAllocDefault));
}

void HostAllocator::FreeMemory(void* memory_ptr) { CUDA_CHECK(cudaFreeAsync(memory_ptr, context_->h2d_streams_[0])); }

}  // namespace numerous_llm
