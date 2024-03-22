/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/block_manager/device_allocator.h"
#include "ksana_llm/block_manager/base_allocator.h"

#ifdef ENABLE_ACL
#  include "ksana_llm/utils/ascend/acl_utils.h"
#endif

namespace ksana_llm {

DeviceAllocator::DeviceAllocator(const AllocatorConfig& allocator_config, std::shared_ptr<Context> context,
                                 int device_id)
    : BaseAllocator(allocator_config, context), device_id_(device_id) {
  // Set to specified device first.
  if (allocator_config.device == MemoryDevice::MEMORY_GPU) {
#ifdef ENABLE_CUDA
    // Set to specified device first.
    CUDA_CHECK(cudaSetDevice(device_id_));
#else
    throw std::invalid_argument("Using NVIDIA GPU but not compile WITH_CUDA=ON");
#endif
  } else if (allocator_config.device == MemoryDevice::MEMORY_ASCEND) {
#ifdef ENABLE_ACL
    ACL_CHECK(aclrtSetDevice(device_id));
#else
    throw std::invalid_argument("Using Huawei Ascend but not compile WITH_ACL=ON");
#endif
  } else {
    throw std::invalid_argument("Unknown device type during DeviceAllocator construction");
  }
}

DeviceAllocator::~DeviceAllocator() {}

int DeviceAllocator::GetDeviceId() { return device_id_; }

}  // namespace ksana_llm
