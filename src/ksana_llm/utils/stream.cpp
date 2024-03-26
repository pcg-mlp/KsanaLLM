/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/utils/stream.h"

#include <stdexcept>

#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/status.h"

#ifdef ENABLE_ACL
#  include "ksana_llm/utils/ascend/acl_utils.h"
#endif

namespace ksana_llm {

Stream::Stream(const int32_t rank, const MemoryDevice device_type) : device_type_(device_type), rank_(rank) {
  if (is_init_) {
    return;
  }
  if (device_type_ == MemoryDevice::MEMORY_GPU) {
#ifdef ENABLE_CUDA
    CUDA_CHECK(cudaSetDevice(rank_));
    CUDA_CHECK(cudaStreamCreate(&cuda_stream_));
#else
    throw std::invalid_argument("Using NVIDIA GPU but not compile WITH_CUDA=ON");
#endif
  } else if (device_type_ == MemoryDevice::MEMORY_ASCEND) {
#ifdef ENABLE_ACL
    ACL_CHECK(aclrtSetDevice(rank_));
    ACL_CHECK(aclrtCreateStream(&acl_stream_));
#else
    throw std::invalid_argument("Using Huawei Ascend NPU but not compile WITH_ACL=ON");
#endif
  } else {
    throw std::invalid_argument("Unknown device type during Stream construction");
  }

  is_init_ = true;
}

void Stream::Destroy() {
  if (!is_init_) {
    return;
  }

  if (device_type_ == MemoryDevice::MEMORY_GPU) {
#ifdef ENABLE_CUDA
    CUDA_CHECK(cudaSetDevice(rank_));
    CUDA_CHECK(cudaStreamDestroy(cuda_stream_));
#else
    throw std::invalid_argument("Using NVIDIA GPU but not compile WITH_CUDA=ON");
#endif
  } else if (device_type_ == MemoryDevice::MEMORY_ASCEND) {
#ifdef ENABLE_ACL
    ACL_CHECK(aclrtSetDevice(rank_));
    ACL_CHECK(aclrtDestroyStream(acl_stream_));
#else
    throw std::invalid_argument("Using Huawei Ascend NPU but not compile WITH_ACL=ON");
#endif
  } else {
    throw std::invalid_argument("Unknown device type during Stream construction");
  }

  is_init_ = false;
}

}  // namespace ksana_llm