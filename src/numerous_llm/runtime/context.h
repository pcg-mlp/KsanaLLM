/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <vector>

#include "numerous_llm/utils/nvidia/cuda_utils.h"
#include "numerous_llm/utils/nvidia/nccl_utils.h"
#include "numerous_llm/utils/ret_code.h"

namespace numerous_llm {

// The global context, like cuda stream, nccl handler.
class Context {
 public:
  Context(const int tensor_parallel_size, const int pipeline_parallel_size);
  ~Context();

 private:
  int device_num_{0};
  int tensor_parallel_size_{0};
  int pipeline_parallel_size_{0};

  ncclUniqueId nccl_uid_;
  std::vector<cudaStream_t> compute_streams_;
  std::vector<cudaStream_t> h2d_streams_;
  std::vector<cudaStream_t> d2h_streams_;
  std::vector<cudaStream_t> nccl_streams_;
  std::vector<NCCLParam> nccl_params_;
};

}  // namespace numerous_llm
