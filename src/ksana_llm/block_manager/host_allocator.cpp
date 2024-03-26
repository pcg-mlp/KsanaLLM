/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/block_manager/host_allocator.h"
#include "ksana_llm/block_manager/base_allocator.h"

#ifdef ENABLE_ACL
#  include "ksana_llm/utils/ascend/acl_utils.h"
#endif

namespace ksana_llm {

HostAllocator::HostAllocator(const AllocatorConfig& allocator_config, std::shared_ptr<Context> context)
    : BaseAllocator(allocator_config, context) {}

void HostAllocator::AllocateMemory(void** memory_ptr, size_t bytes) {
#ifdef ENABLE_CUDA
  CUDA_CHECK(cudaHostAlloc(memory_ptr, bytes, cudaHostAllocDefault));
#else
  // NOTE(karlluo): 由于异步内存复制时，要求首地址64字节对齐，因此申请内存时，size需加64
  ACL_CHECK(aclrtMallocHost(memory_ptr, bytes));
#endif
}

void HostAllocator::FreeMemory(void* memory_ptr) {
#ifdef ENABLE_CUDA
  CUDA_CHECK(cudaFreeAsync(memory_ptr, context_->GetH2DStreams()[0].GetStreamIns()));
#else
  ACL_CHECK(aclrtFreeHost(memory_ptr));
#endif
}

}  // namespace ksana_llm
