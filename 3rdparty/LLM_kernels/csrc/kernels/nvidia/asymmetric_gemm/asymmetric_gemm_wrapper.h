/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "csrc/utils/nvidia/cuda_bf16_wrapper.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace llm_kernels {
namespace nvidia {

enum WeightType { INT4, INT8 };

template <typename T, WeightType WT>
class FpAIntBGPTQGemmWrapper {
 public:
  void GetWorkspaceSize(size_t m, size_t n, size_t k, size_t& ws_bytes);

  void Gemm(void* output, const void* input, const void* weight, const void* scales, void* ws, size_t m, size_t n,
            size_t k, size_t groupsize, cudaStream_t stream);
};

}  // namespace nvidia
}  // namespace llm_kernels