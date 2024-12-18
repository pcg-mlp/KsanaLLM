/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/block_manager/device_allocator.h"
#include "ksana_llm/block_manager/base_allocator.h"
#include "ksana_llm/utils/device_utils.h"

namespace ksana_llm {

DeviceAllocator::DeviceAllocator(const AllocatorConfig& allocator_config, std::shared_ptr<Context> context,
                                 int device_id)
    : BaseAllocator(allocator_config, context), device_id_(device_id) {
  // Set to specified device first.
  SetDevice(device_id_);
}

DeviceAllocator::~DeviceAllocator() { Clear(); }

int DeviceAllocator::GetDeviceId() { return device_id_; }

void DeviceAllocator::AllocateMemory(void** memory_ptr, size_t bytes) {
  MallocAsync(memory_ptr, bytes, context_->GetMemoryManageStreams()[device_id_]);
}

void DeviceAllocator::FreeMemory(void* memory_ptr) {
  FreeAsync(memory_ptr, context_->GetMemoryManageStreams()[device_id_]);
}

}  // namespace ksana_llm
