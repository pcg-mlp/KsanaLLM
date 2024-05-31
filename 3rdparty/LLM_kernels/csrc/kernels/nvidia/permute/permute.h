/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <cuda_runtime.h>
#include "csrc/kernels/nvidia/permute/nd_index_offset_helper.h"

namespace llm_kernels {
namespace nvidia {

template <size_t num_dims, typename IndexType>
struct PermuteKernelParams {
  NdIndexOffsetHelper<IndexType, num_dims> src_index_helper;
  NdIndexOffsetHelper<IndexType, num_dims> dst_index_helper;
  int permutation[num_dims]{};
  IndexType count{};
  const void* src{};
  void* dst{};
};

/// @brief implement of permute
/// @tparam num_dims dimention number of tensor
/// @tparam movement_size copy trunk size of data in GPU global memory, movement_size is the size of trunk
/// @tparam IndexType memory access pointer's type
/// @param params PermuteKernelParams for permute
/// @param stream cuda stream for asynchronize inference
template <size_t num_dims, size_t movement_size>
void InvokePermute(void* input, void* output, std::vector<size_t> input_shape, std::vector<size_t> permutation,
                   cudaStream_t& stream);

}  // namespace nvidia
}  // namespace llm_kernels
