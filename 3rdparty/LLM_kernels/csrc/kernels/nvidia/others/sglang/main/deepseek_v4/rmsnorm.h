/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

/*
 * Adapted from
 * https://github.com/sgl-project/sglang/blob/main/python/sglang/jit_kernel/csrc/deepseek_v4/rmsnorm.cuh
 *
 * Copyright 2023-2024 SGLang Team
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
#include <cstdint>

namespace llm_kernels {
namespace nvidia {


// -----------------------------------------------------------------------------
// rmsnorm_self
// -----------------------------------------------------------------------------
// Per-head RMSNorm operating on a strided (non-contiguous) input tensor.
//
// Input  layout: [batch_size, num_head, head_dim] with arbitrary strides
//                (stride_batch_bytes and stride_head_bytes in bytes).
// Output layout: contiguous [batch_size * num_head, head_dim].
//
// No learnable weight is applied (identity weight), matching SGLang's
// DeepSeek-V4 rmsnorm.cuh semantics.
//
// Template parameters:
//   DType    - element type: __nv_bfloat16 or float
//   kHeadDim - compile-time head dimension (must satisfy kHeadDim % 64 == 0)
template <typename DType, int64_t kHeadDim>
void rmsnorm_self(const void* input, void* output, int64_t stride_batch_bytes, int64_t stride_head_bytes,
                  uint32_t batch_size, uint32_t num_head, float eps, cudaStream_t stream);

}  // namespace nvidia
}  // namespace llm_kernels
