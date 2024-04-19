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
#include "csrc/utils/nvidia/quantization.h"

namespace llm_kernels {
namespace nvidia {

/*
  This runner supports:
  int8_t inputs (A and B)
  float alpha scalings (either per-col, or per-col x per-row)
  T output (D) where T = {float, half, __nv_bfloat16} // TODO

  Activations, biases, scales and outputs are all assumed to be row-major.
  Weights are assumed to be column-major.
*/

class CutlassInt8GemmRunnerInterface {
 public:
  CutlassInt8GemmRunnerInterface() {}

  virtual ~CutlassInt8GemmRunnerInterface() {}

  virtual void Gemm(const int8_t* A, const int8_t* B, llm_kernels::utils::QuantMode quant_option,
                    const float* alpha_col, const float* alpha_row, void* C, int32_t m, int32_t n, int32_t k,
                    llm_kernels::nvidia::cutlass_extensions::CutlassGemmConfig gemm_config, char* workspace_ptr,
                    const size_t workspace_bytes, cudaStream_t stream) = 0;

  // Returns desired workspace size in bytes.
  virtual size_t GetWorkspaceSize(const int32_t m, const int32_t n, const int32_t k) = 0;

  virtual std::vector<llm_kernels::nvidia::cutlass_extensions::CutlassGemmConfig> GetConfigs() const = 0;

 protected:
  static constexpr int32_t SPLIT_K_LIMIT = 7;
  static constexpr int32_t MIN_M_TILE = 32;
  static constexpr int32_t MIN_N_TILE = 64;
};

template <typename T>
class CutlassInt8GemmRunner : public virtual CutlassInt8GemmRunnerInterface {
 public:
  CutlassInt8GemmRunner();
  ~CutlassInt8GemmRunner();

  void Gemm(const int8_t* A, const int8_t* B, llm_kernels::utils::QuantMode quant_option, const float* alpha_col,
            const float* alpha_row, void* C, int32_t m, int32_t n, int32_t k,
            llm_kernels::nvidia::cutlass_extensions::CutlassGemmConfig gemm_config, char* workspace_ptr,
            const size_t workspace_bytes, cudaStream_t stream) override;

  // Returns desired workspace size in bytes.
  size_t GetWorkspaceSize(const int32_t m, const int32_t n, const int32_t k) override;

  std::vector<llm_kernels::nvidia::cutlass_extensions::CutlassGemmConfig> GetConfigs() const override;

 private:
  void DispatchToArch(const int8_t* A, const int8_t* B, llm_kernels::utils::QuantMode quant_option,
                      const float* alpha_col, const float* alpha_row, T* C, int32_t m, int32_t n, int32_t k,
                      llm_kernels::nvidia::cutlass_extensions::CutlassGemmConfig gemm_config, char* workspace_ptr,
                      const size_t workspace_bytes, cudaStream_t stream, int32_t* occupancy = nullptr);

  int32_t sm_;
  int32_t multiprocessor_count_;
};

}  // namespace nvidia
}  // namespace llm_kernels
