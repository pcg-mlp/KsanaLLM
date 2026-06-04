/* Copyright 2026 Tencent Inc.  All rights reserved.
 * Adapted from https://github.com/sgl-project/sglang/blob/f63def851/sgl-kernel/csrc/moe/moe_topk_softmax_kernels.cu
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
 ==============================================================================*/

#pragma once

#include <cuda_runtime.h>

namespace llm_kernels {
namespace nvidia {

/**
 * @brief Warp-parallel softmax + topk kernel (sglang-derived)
 *
 * Fuses softmax computation and top-k selection into a single kernel using
 * butterfly shuffle reductions. ~5x faster than the serial BasicSoftmaxTopk.
 * Requires num_experts to be a power of 2 (1..256).
 *
 * @tparam T Input data type (float, half, __nv_bfloat16)
 * @param gating_output Input logits [num_rows, num_experts]
 * @param topk_weights_ptr Output top-k weights [num_rows, topk], float32
 * @param topk_ids_ptr Output top-k indices [num_rows, topk], int32
 * @param num_rows Number of tokens
 * @param num_experts Number of experts (must be power of 2, <= 256)
 * @param topk Number of experts to select
 * @param renormalize Whether to renormalize topk weights to sum=1
 * @param routed_scaling_factor Scaling factor applied to output weights
 * @param stream CUDA stream
 */
template <typename T>
void InvokeTopkGatingSoftmax(const void* gating_output, void* topk_weights_ptr, void* topk_ids_ptr, int num_rows,
                             int num_experts, int topk, bool renormalize, float routed_scaling_factor,
                             cudaStream_t stream);

}  // namespace nvidia
}  // namespace llm_kernels
