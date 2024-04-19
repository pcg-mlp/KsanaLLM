/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuda_runtime_api.h>
#include "csrc/kernels/nvidia/cutlass_extensions/gemm_configs.h"
#include "csrc/kernels/nvidia/cutlass_extensions/weight_only_quant_op.h"

namespace tkc = llm_kernels::nvidia::cutlass_extensions;

namespace llm_kernels {
namespace nvidia {

enum class ActivationType { Gelu, Relu, Silu, Identity, InvalidType };

// NOTE(karlluo): This runner only supports:
// T in {half, __nv_bfloat} WeightType in {int8_t, cutlass::uint4b_t}
// Activations, biases, scales and outputs are all assumed to be row-major.
// However, it is assumed that B is in a special format governed by cutlass_extensions/gemm/kernel/mixed_gemm_B_layout.
// In this case, B must be preprocessed using the cutlass weight only quant preprocessors. The weight preprocessor
// will instantiate the layout and preprocess based on the instantiation, so layout changes should only require
// modifications to mix_gemm_B_layout.h.
class CutlassFpAIntBGemmRunnerInterface {
 public:
  CutlassFpAIntBGemmRunnerInterface() {}

  virtual ~CutlassFpAIntBGemmRunnerInterface() {}

  virtual void Gemm(const void* A, const void* B, const void* weight_scales, void* C, int32_t m, int32_t n, int32_t k,
                    tkc::CutlassGemmConfig gemm_config, char* workspace_ptr, const size_t workspace_bytes,
                    cudaStream_t stream) = 0;

  virtual void Gemm(const void* A, const void* B, const void* weight_scales, const void* weight_zero_points,
                    const void* biases, void* C, int32_t m, int32_t n, int32_t k, const int32_t group_size,
                    tkc::CutlassGemmConfig gemm_config, char* workspace_ptr, const size_t workspace_bytes,
                    cudaStream_t stream) = 0;

  // Returns desired workspace size in bytes.
  virtual size_t GetWorkspaceSize(const int32_t m, const int32_t n, const int32_t k) = 0;

  virtual std::vector<tkc::CutlassGemmConfig> GetConfigs() const = 0;

 protected:
  static constexpr int32_t SPLIT_K_LIMIT = 7;
  static constexpr int32_t MIN_M_TILE = 32;
  static constexpr int32_t MIN_N_TILE = 64;
};

template <typename T, typename WeightType, cutlass::WeightOnlyQuantOp QuantOp>
class CutlassFpAIntBGemmRunner : public virtual CutlassFpAIntBGemmRunnerInterface {
 public:
  CutlassFpAIntBGemmRunner();
  ~CutlassFpAIntBGemmRunner();

  void Gemm(const void* A, const void* B, const void* weight_scales, void* C, int32_t m, int32_t n, int32_t k,
            tkc::CutlassGemmConfig gemm_config, char* workspace_ptr, const size_t workspace_bytes,
            cudaStream_t stream) override;

  void Gemm(const void* A, const void* B, const void* weight_scales, const void* weight_zero_points, const void* biases,
            void* C, int32_t m, int32_t n, int32_t k, const int32_t group_size, tkc::CutlassGemmConfig gemm_config,
            char* workspace_ptr, const size_t workspace_bytes, cudaStream_t stream) override;

  // Returns desired workspace size in bytes.
  size_t GetWorkspaceSize(const int32_t m, const int32_t n, const int32_t k) override;

  std::vector<tkc::CutlassGemmConfig> GetConfigs() const override;

 private:
  template <typename EpilogueTag>
  void dispatch_to_arch(const T* A, const WeightType* B, const T* weight_scales, const T* weight_zero_points,
                        const T* biases, T* C, int32_t m, int32_t n, int32_t k, const int32_t group_size,
                        tkc::CutlassGemmConfig gemm_config, char* workspace_ptr, const size_t workspace_bytes,
                        cudaStream_t stream, int32_t* occupancy = nullptr);

 private:
  int32_t sm_;
  int32_t multi_processor_count_;
};

}  // namespace nvidia
}  // namespace llm_kernels
