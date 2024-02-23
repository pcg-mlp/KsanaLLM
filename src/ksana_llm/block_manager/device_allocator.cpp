/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/block_manager/device_allocator.h"
#include "ksana_llm/block_manager/base_allocator.h"

namespace ksana_llm {

DeviceAllocator::DeviceAllocator(const AllocatorConfig& allocator_config, std::shared_ptr<Context> context,
                                 int device_id)
    : BaseAllocator(allocator_config, context), device_id_(device_id) {
  // Set to specified device first.
  CUDA_CHECK(cudaSetDevice(device_id_));
}

DeviceAllocator::~DeviceAllocator() {}

int DeviceAllocator::GetDeviceId() { return device_id_; }

}  // namespace ksana_llm
