/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "csrc/utils/nvidia/cuda_bf16_wrapper.h"
#include "csrc/kernels/nvidia/asymmetric_gemm/asymmetric_gemm_wrapper.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace llm_kernels {
namespace nvidia {

template <typename T, WeightType WT>
class FpAIntBGroupCudaGemmWrapper {
 public:
  FpAIntBGroupCudaGemmWrapper();

  bool IsSupport();

  void Gemm(void* output, const void* input, const void* weight, const void* scales, const void* zeros, size_t m,
            size_t n, size_t k, size_t groupsize, cudaStream_t stream);

 private:
  int arch;
};

}  // namespace nvidia
}  // namespace llm_kernels