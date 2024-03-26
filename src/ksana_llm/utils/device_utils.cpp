/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/utils/device_utils.h"

#include "ksana_llm/utils/ret_code.h"

#ifdef ENABLE_ACL
#  include "ksana_llm/utils/ascend/acl_utils.h"
#endif

#ifdef ENABLE_CUDA
#  include "ksana_llm/utils/nvidia/cuda_utils.h"
#endif
namespace ksana_llm {

uint32_t GetDeviceNumber(MemoryDevice device_type) {
  uint32_t device_num = 0;

  if (device_type == MemoryDevice::MEMORY_GPU) {
#ifdef ENABLE_CUDA
    int nvidia_device_num = 0;
    CUDA_CHECK(cudaGetDeviceCount(&nvidia_device_num));
    device_num = static_cast<uint32_t>(nvidia_device_num);
#endif
  } else if (device_type == MemoryDevice::MEMORY_ASCEND) {
#ifdef ENABLE_ACL
    ACL_CHECK(aclrtGetDeviceCount(&device_num));
#endif
  }

  return device_num;
}

}  // namespace ksana_llm