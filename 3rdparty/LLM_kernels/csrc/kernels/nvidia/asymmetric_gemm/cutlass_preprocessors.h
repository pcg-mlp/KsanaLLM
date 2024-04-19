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

#include <stdint.h>
#include <cstddef>
#include <vector>

#include "csrc/utils/nvidia/cuda_utils.h"

namespace llm_kernels {
namespace nvidia {

enum class QuantType { INT8_WEIGHT_ONLY, PACKED_INT4_WEIGHT_ONLY };
int32_t GetBitsInQuantType(QuantType quant_type);

// Shapes here can be 2 or 3D. 2-D shapes are [num_rows, num_cols]
// 3-D shapes are [num_experts, num_rows, num_cols]
void PermuteBRowsForMixedGemm(int8_t* permuted_quantized_tensor, const int8_t* quantized_tensor,
                              const std::vector<size_t>& shape, QuantType quant_type, const int64_t arch_version);

void SubbyteTranspose(int8_t* transposed_quantized_tensor, const int8_t* quantized_tensor,
                      const std::vector<size_t>& shape, QuantType quant_type);

void AddBiasAndInterleaveQuantizedTensorInplace(int8_t* tensor, const size_t num_elts, QuantType quant_type);

void PreprocessWeightsForMixedGemm(int8_t* preprocessed_quantized_weight, const int8_t* row_major_quantized_weight,
                                   const std::vector<size_t>& shape, QuantType quant_type);

template <typename ComputeType, typename WeightType>
void SymmetricQuantize(int8_t* processed_quantized_weight, ComputeType* scale_ptr, const WeightType* input_weight_ptr,
                       const std::vector<size_t>& shape, QuantType quant_type);

// This is exposed so that we can write tests that use the processed weights for CUTLASS but the unprocessed weight
// to implement a simple reference implementation.
template <typename ComputeType, typename WeightType>
void SymmetricQuantize(int8_t* processed_quantized_weight, int8_t* unprocessed_quantized_weight, ComputeType* scale_ptr,
                       const WeightType* input_weight_ptr, const std::vector<size_t>& shape, QuantType quant_type);

}  // namespace nvidia
}  // namespace llm_kernels
