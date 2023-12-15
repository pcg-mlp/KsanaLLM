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

  int GetTensorParallelSize() { return tensor_parallel_size_; }

  int GetPipeLineParallelSize() { return pipeline_parallel_size_; }

 private:
  int device_num_{0};
  int tensor_parallel_size_{0};
  int pipeline_parallel_size_{0};
  const int defalt_device_num_{0};

  // cuda streams
  std::vector<cudaStream_t> compute_streams_;
  std::vector<cudaStream_t> h2d_streams_;
  std::vector<cudaStream_t> d2h_streams_;
  std::vector<cudaStream_t> nccl_streams_;
  // nccl comms
  ncclUniqueId nccl_uid_;
  std::vector<NCCLParam> nccl_params_;
  // cublas handles
  std::vector<cublasHandle_t> cublas_handles_;
  std::vector<cublasLtHandle_t> cublaslt_handles_;
};

}  // namespace numerous_llm
