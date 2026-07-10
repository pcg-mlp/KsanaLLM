/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

/*
 * Adapted from
 * https://github.com/sgl-project/sglang/blob/main/python/sglang/jit_kernel/include/sgl_kernel/deepseek_v4/compress.cuh
 * and
 * https://github.com/sgl-project/sglang/blob/main/python/sglang/jit_kernel/csrc/deepseek_v4/common.cuh
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

#include "csrc/kernels/nvidia/others/sglang/main/deepseek_v4/plan.h"

namespace llm_kernels {
namespace nvidia {


// -----------------------------------------------------------------------------
// plan_prefill_compress
// -----------------------------------------------------------------------------
// Build compress_plan / write_plan for prefill-time compression.
//
// Inputs (all on device):
//   seq_lens    : [batch_size], int64_t   - full sequence length (prefix + extend)
//   extend_lens : [batch_size], int64_t   - extend (new) tokens length in this step
//
// Outputs (all on device, pre-allocated with sufficient capacity):
//   compress_plan : overwritten with PrefillPlan entries (compact, no padding).
//   write_plan    : same as above.
//
void plan_prefill_compress(PrefillPlan* compress_plan, PrefillPlan* write_plan, const int64_t* seq_lens,
                           const int64_t* extend_lens, uint32_t batch_size, uint32_t num_tokens,
                           uint32_t compress_ratio, bool is_overlap, cudaStream_t stream);

namespace {

constexpr uint32_t kPlanBlockSize = 1024;

struct CompressParams {
  PrefillPlan* __restrict__ compress_plan;
  PrefillPlan* __restrict__ write_plan;
  const IndicesT* __restrict__ seq_lens;
  const IndicesT* __restrict__ extend_lens;
  uint32_t batch_size;
  uint32_t num_tokens;
  uint32_t compress_ratio;
  bool is_overlap;
};

// Device-side single-block plan builder.
// Mirrors SGLang's `plan_prefill_cuda` with identical semantics.
__global__ __launch_bounds__(kPlanBlockSize,
                             1) void plan_prefill_compress_kernel(const __grid_constant__ CompressParams params) {
  const auto& [compress_plan, write_plan, seq_lens, extend_lens,   // pointers
               batch_size, num_tokens, compress_ratio, is_overlap  // values
  ] = params;

  __shared__ uint32_t compress_counter;
  __shared__ uint32_t write_counter;

  uint32_t batch_id = 0;
  uint32_t counter = 0;
  uint32_t extend_len = extend_lens[0];

  const auto tid = threadIdx.x;
  if (tid == 0) {
    compress_counter = 0;
    write_counter = 0;
  }
  __syncthreads();

  // Each thread sequentially scans a strided subset of ragged tokens.
  // The running `batch_id / counter / extend_len` state must be re-initialized
  // per-thread (no shared scan context).
  for (uint32_t i = tid; i < num_tokens; i += blockDim.x) {
    const uint32_t ragged_id = i;
    // Advance (batch_id, counter, extend_len) until `ragged_id - counter < extend_len`.
    uint32_t j = ragged_id - counter;
    while (j >= extend_len) {
      j -= extend_len;
      batch_id += 1;
      if (batch_id >= batch_size) [[unlikely]]
        break;
      counter += extend_len;
      extend_len = extend_lens[batch_id];
    }
    if (batch_id >= batch_size) [[unlikely]]
      break;

    const uint32_t seq_len = static_cast<uint32_t>(seq_lens[batch_id]);
    const uint32_t extend_len = static_cast<uint32_t>(extend_lens[batch_id]);
    const uint32_t prefix_len = seq_len - extend_len;
    const uint32_t ratio_eff = compress_ratio * (1u + (is_overlap ? 1u : 0u));
    const uint32_t window_len = (j + 1u < ratio_eff) ? (ratio_eff - (j + 1u)) : 0u;
    const uint32_t position = prefix_len + j;

    // start_write_pos: the earliest position that should be written out
    // in overlap mode we roll back one window; otherwise align down.
    // For MTP kv fallback (extend spans up to threshold tokens, may cross
    // page boundaries), write every extend token so prior pages get
    // correctly repopulated:
    //   -   c4: bundle exposes 2 back pages → span up to 3 pages = 1 + 4 + 4 = 9 tokens.
    //   - c128: load_indices for 1 back page → span up to 2 pages
    constexpr uint32_t kFallbackWriteAllThreshold = 9u;
    const uint32_t start_write_pos = [seq_len, extend_len, prefix_len, compress_ratio, is_overlap] {
      if (extend_len <= kFallbackWriteAllThreshold) {
        return prefix_len;
      }
      const uint32_t pos = seq_len / compress_ratio * compress_ratio;
      if (!is_overlap) return pos;
      return pos >= compress_ratio ? pos - compress_ratio : 0u;
    }();

    if ((position + 1u) % compress_ratio == 0u) {
      const auto write_pos = atomicAdd(&compress_counter, 1u);
      compress_plan[write_pos] = PrefillPlan{
          .ragged_id = ragged_id,
          .batch_id = batch_id,
          .position = position,
          .window_len = window_len,
      };
    }
    if (position >= start_write_pos) {
      // For write plans, `window_len` is repurposed to encode `page_back`:
      // number of pages between this token's page and the last page.
      const uint32_t last_page_id = (seq_len - 1u) / compress_ratio;
      const uint32_t token_page_id = position / compress_ratio;
      const uint32_t page_back = last_page_id - token_page_id;
      const auto write_pos = atomicAdd(&write_counter, 1u);
      write_plan[write_pos] = PrefillPlan{
          .ragged_id = ragged_id,
          .batch_id = batch_id,
          .position = position,
          .window_len = page_back,
      };
    }
  }
}

}  // namespace

inline void plan_prefill_compress(PrefillPlan* compress_plan, PrefillPlan* write_plan, const IndicesT* seq_lens,
                                  const IndicesT* extend_lens, uint32_t batch_size, uint32_t num_tokens,
                                  uint32_t compress_ratio, bool is_overlap, cudaStream_t stream) {
  const CompressParams params{compress_plan, write_plan, seq_lens,       extend_lens,
                              batch_size,    num_tokens, compress_ratio, is_overlap};
  plan_prefill_compress_kernel<<<1, kPlanBlockSize, 0, stream>>>(params);
}

}  // namespace nvidia
}  // namespace llm_kernels
