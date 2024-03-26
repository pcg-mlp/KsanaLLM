/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/utils/memory_device.h"

#ifdef ENABLE_CUDA
#  include <cuda_runtime.h>
#  include "ksana_llm/utils/nvidia/cuda_utils.h"
#endif

#ifdef ENABLE_ACL
#  include <acl/acl.h>
#  include <acl/acl_base.h>
#  include <acl/acl_rt.h>
#endif

namespace ksana_llm {

class Stream {
 public:
  Stream(const int32_t rank, const MemoryDevice device_type);
  ~Stream() {};

  void Destroy();

#ifdef ENABLE_CUDA
  cudaStream_t& GetStreamIns() { return cuda_stream_; }
#endif

#ifdef ENABLE_ACL
  aclrtStream& GetStreamIns() { return acl_stream_; }
#endif

 private:
  int32_t rank_{0};
  MemoryDevice device_type_{MemoryDevice::MEMORY_GPU};
  bool is_init_{false};

#ifdef ENABLE_CUDA
  cudaStream_t cuda_stream_;
#endif

#ifdef ENABLE_ACL
  aclrtStream acl_stream_;
#endif
};

}  // namespace ksana_llm