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
#include <vector>
#include "csrc/kernels/nvidia/cutlass_extensions/gemm_configs.h"
#include "csrc/utils/nvidia/quantization.h"

using namespace llm_kernels::utils;

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

  virtual void gemm(int8_t const* A, int8_t const* B, llm_kernels::utils::QuantMode quantOption, float const* alphaCol,
                    float const* alphaRow, void* C, int m, int n, int k,
                    llm_kernels::nvidia::cutlass_extensions::CutlassGemmConfig gemmConfig, char* workspacePtr,
                    const size_t workspaceBytes, cudaStream_t stream) = 0;

  // Returns desired workspace size in bytes.
  virtual size_t getWorkspaceSize(int const m, int const n, int const k) = 0;

  virtual std::vector<llm_kernels::nvidia::cutlass_extensions::CutlassGemmConfig> getConfigs() const = 0;

 protected:
  static constexpr int SPLIT_K_LIMIT = 7;
  static constexpr int MIN_M_TILE = 32;
  static constexpr int MIN_N_TILE = 64;
};

template <typename T>
class CutlassInt8GemmRunner : public virtual CutlassInt8GemmRunnerInterface {
 public:
  CutlassInt8GemmRunner();
  ~CutlassInt8GemmRunner();

  void gemm(int8_t const* A, int8_t const* B, llm_kernels::utils::QuantMode quantOption, float const* alphaCol,
            float const* alphaRow, void* C, int m, int n, int k,
            llm_kernels::nvidia::cutlass_extensions::CutlassGemmConfig gemmConfig, char* workspacePtr,
            const size_t workspaceBytes, cudaStream_t stream) override;

  // Returns desired workspace size in bytes.
  size_t getWorkspaceSize(int const m, int const n, int const k) override;

  std::vector<llm_kernels::nvidia::cutlass_extensions::CutlassGemmConfig> getConfigs() const override;

 private:
  void dispatchToArch(int8_t const* A, int8_t const* B, llm_kernels::utils::QuantMode quantOption,
                      float const* alphaCol, float const* alphaRow, T* C, int m, int n, int k,
                      llm_kernels::nvidia::cutlass_extensions::CutlassGemmConfig gemmConfig, char* workspacePtr,
                      const size_t workspaceBytes, cudaStream_t stream, int* occupancy = nullptr);

  int mSm;
  int mMultiProcessorCount;
};

}  // namespace nvidia
}  // namespace llm_kernels
