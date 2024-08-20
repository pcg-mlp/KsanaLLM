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
class FpAIntBGroupCutlassGemmWrapper {
 public:
  void GetWorkspaceSize(size_t m, size_t n, size_t k, size_t& ws_bytes);

  void Gemm(void* output, const void* input, const void* weight, const void* scales, void* ws, size_t m, size_t n,
            size_t k, size_t groupsize, size_t config_index, cudaStream_t stream);

  size_t GetBestConfigIndex(size_t warmup, size_t iter, void* output, const void* input, const void* weight,
                            const void* scales, void* ws, size_t m, size_t n, size_t k, size_t groupsize,
                            cudaStream_t stream);
};

template <typename T, WeightType WT>
class FpAIntBGroupCudaGemmWrapper {
 public:
  FpAIntBGroupCudaGemmWrapper();

  bool IsSupport();

  void Gemm(void* output, const void* input, const void* weight, const void* scales, size_t m, size_t n, size_t k,
            size_t groupsize, cudaStream_t stream);
 private:
  int arch;
};

}  // namespace nvidia
}  // namespace llm_kernels