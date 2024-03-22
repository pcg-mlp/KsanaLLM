/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/block_manager/host_allocator.h"
#include "ksana_llm/block_manager/base_allocator.h"
#include "ksana_llm/utils/device_helper.h"

namespace ksana_llm {

HostAllocator::HostAllocator(const AllocatorConfig& allocator_config, std::shared_ptr<Context> context)
    : BaseAllocator(allocator_config, context) {}

void HostAllocator::AllocateMemory(void** memory_ptr, size_t bytes) { HostAlloc(memory_ptr, bytes); }

void HostAllocator::FreeMemory(void* memory_ptr) { FreeHost(memory_ptr); }

}  // namespace ksana_llm
