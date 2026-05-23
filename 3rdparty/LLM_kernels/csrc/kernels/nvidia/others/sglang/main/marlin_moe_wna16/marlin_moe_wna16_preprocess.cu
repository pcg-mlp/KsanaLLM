/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * Weight preprocessing utilities for Marlin MoE WNA16.
 * Adapted from gptq_marlin_repack.cu and gptq_marlin.cu (permute_scales).
 */

#include "marlin_moe_wna16_preprocess.h"
#include "marlin.cuh"
#include "csrc/utils/nvidia/assert.h"

using namespace ::marlin_moe_wna16;

namespace llm_kernels {
namespace nvidia {
namespace marlin_moe_wna16 {

// ============================================================================
// Repack kernel: transforms quantized weights from GPTQ layout to Marlin tiles.
// Copied from gptq_marlin_repack.cu with namespace adaptation.
// ============================================================================

template <int const num_threads, int const num_bits>
__global__ void marlin_moe_repack_kernel(uint32_t const* __restrict__ b_q_weight_ptr,
                                         uint32_t* __restrict__ out_ptr,
                                         int num_experts, int size_k, int size_n) {
  constexpr int pack_factor = 32 / num_bits;

  int k_tiles = size_k / tile_k_size;
  int n_tiles = size_n / tile_n_size;
  int block_k_tiles = div_ceil(k_tiles, gridDim.x);

  auto start_k_tile = blockIdx.x * block_k_tiles;
  if (start_k_tile >= k_tiles) {
    return;
  }

  int expert_idx = blockIdx.y;
  // NOTE: All three operands are 32-bit `int`. For large MoE shapes (e.g. up_gate_proj with
  // num_experts=96, size_k=7168, size_n=6144) the product exceeds INT_MAX and silently
  // overflows to a negative number, which then becomes a huge size_t after promotion → IMA.
  // Cast the leading operand to size_t so the entire chain is 64-bit.
  size_t offset = static_cast<size_t>(expert_idx) * static_cast<size_t>(size_k) *
                  static_cast<size_t>(size_n) / static_cast<size_t>(pack_factor);
  uint32_t const* expert_b_q_weight_ptr = b_q_weight_ptr + offset;
  uint32_t* expert_out_ptr = out_ptr + offset;

  int finish_k_tile = min(start_k_tile + block_k_tiles, k_tiles);

  // Wait until the next thread tile has been loaded to shared memory.
  auto wait_for_stage = [&]() {
    // We only have `stages - 2` active fetches since we are double buffering
    // and can only issue the next fetch when it is guaranteed that the previous
    // shared memory load is fully complete (as it may otherwise be
    // overwritten).
    cp_async_wait<repack_stages - 2>();
    __syncthreads();
  };

  extern __shared__ int4 sh[];

  int4* sh_pipe_ptr = sh;

  constexpr int tile_ints = tile_k_size / pack_factor;

  constexpr int stage_n_threads = tile_n_size / 4;
  constexpr int stage_k_threads = tile_ints;
  constexpr int stage_size = stage_k_threads * stage_n_threads;

  auto fetch_to_shared = [&](int pipe, int k_tile_id, int n_tile_id) {
    if (n_tile_id >= n_tiles) {
      cp_async_fence();
      return;
    }

    int first_n = n_tile_id * tile_n_size;

    int4* sh_ptr = sh_pipe_ptr + stage_size * pipe;

    if (threadIdx.x < stage_size) {
      auto k_id = threadIdx.x / stage_n_threads;
      auto n_id = threadIdx.x % stage_n_threads;

      int first_k = k_tile_id * tile_k_size;
      int first_k_packed = first_k / pack_factor;

      cp_async4(&sh_ptr[k_id * stage_n_threads + n_id],
                reinterpret_cast<int4 const*>(
                    &(expert_b_q_weight_ptr[(first_k_packed + k_id) * size_n + first_n + (n_id * 4)])));
    }

    cp_async_fence();
  };

  auto repack_tile = [&](int pipe, int k_tile_id, int n_tile_id) {
    if (n_tile_id >= n_tiles) {
      return;
    }

    auto warp_id = threadIdx.x / 32;
    auto th_id = threadIdx.x % 32;

    if (warp_id >= 4) {
      return;
    }

    int tc_col = th_id / 4;
    int tc_row = (th_id % 4) * 2;

    constexpr int tc_offsets[4] = {0, 1, 8, 9};

    int cur_n = warp_id * 16 + tc_col;

    constexpr int sh_stride = 64;
    constexpr uint32_t mask = (1 << num_bits) - 1;

    int4* sh_stage_ptr = sh_pipe_ptr + stage_size * pipe;
    uint32_t* sh_stage_int_ptr = reinterpret_cast<uint32_t*>(sh_stage_ptr);

    uint32_t vals[8];

    uint32_t b1_vals[tile_ints];
    uint32_t b2_vals[tile_ints];

#pragma unroll
    for (int i = 0; i < tile_ints; i++) {
      b1_vals[i] = sh_stage_int_ptr[cur_n + sh_stride * i];
      b2_vals[i] = sh_stage_int_ptr[cur_n + 8 + sh_stride * i];
    }

#pragma unroll
    for (int i = 0; i < 4; i++) {
      int cur_elem = tc_row + tc_offsets[i];
      int cur_int = cur_elem / pack_factor;
      int cur_pos = cur_elem % pack_factor;

      vals[i] = (b1_vals[cur_int] >> (cur_pos * num_bits)) & mask;
      vals[4 + i] = (b2_vals[cur_int] >> (cur_pos * num_bits)) & mask;
    }

    constexpr int tile_sz = tile_k_size * tile_n_size / pack_factor;
    int out_offset = (k_tile_id * n_tiles + n_tile_id) * tile_sz;

    // Result of:
    // https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h
    if constexpr (num_bits == 4) {
      constexpr int pack_idx[8] = {0, 2, 4, 6, 1, 3, 5, 7};

      uint32_t res = 0;
#pragma unroll
      for (int i = 0; i < 8; i++) {
        res |= vals[pack_idx[i]] << (i * 4);
      }

      expert_out_ptr[out_offset + th_id * 4 + warp_id] = res;

    } else {
      constexpr int pack_idx[4] = {0, 2, 1, 3};

      uint32_t res1 = 0;
      uint32_t res2 = 0;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        res1 |= vals[pack_idx[i]] << (i * 8);
        res2 |= vals[4 + pack_idx[i]] << (i * 8);
      }

      expert_out_ptr[out_offset + th_id * 8 + (warp_id * 2) + 0] = res1;
      expert_out_ptr[out_offset + th_id * 8 + (warp_id * 2) + 1] = res2;
    }
  };

  auto start_pipes = [&](int k_tile_id, int n_tile_id) {
#pragma unroll
    for (int pipe = 0; pipe < repack_stages - 1; pipe++) {
      fetch_to_shared(pipe, k_tile_id, n_tile_id + pipe);
    }

    wait_for_stage();
  };
#pragma unroll
  for (int k_tile_id = start_k_tile; k_tile_id < finish_k_tile; k_tile_id++) {
    int n_tile_id = 0;

    start_pipes(k_tile_id, n_tile_id);

    while (n_tile_id < n_tiles) {
#pragma unroll
      for (int pipe = 0; pipe < repack_stages; pipe++) {
        fetch_to_shared((pipe + repack_stages - 1) % repack_stages, k_tile_id, n_tile_id + pipe + repack_stages - 1);
        repack_tile(pipe, k_tile_id, n_tile_id + pipe);
        wait_for_stage();
      }
      n_tile_id += repack_stages;
    }
  }
}

#define CALL_IF_MOE_REPACK(NUM_BITS)                                                                              \
  else if (num_bits == NUM_BITS) {                                                                                \
    cudaFuncSetAttribute(marlin_moe_repack_kernel<repack_threads, NUM_BITS>,                                      \
                         cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem);                             \
    marlin_moe_repack_kernel<repack_threads, NUM_BITS>                                                            \
        <<<blocks, repack_threads, max_shared_mem, stream>>>(b_q_weight_ptr, out_ptr, num_experts, size_k, size_n); \
  }

void marlin_moe_repack(const uint32_t* b_q_weight_ptr, uint32_t* out_ptr,
                       int64_t num_experts, int64_t size_k, int64_t size_n,
                       int64_t num_bits, int rank, cudaStream_t stream) {
  int processors = 0, max_shared_mem = 0;
  cudaDeviceGetAttribute(&processors, cudaDevAttrMultiProcessorCount, rank);
  cudaDeviceGetAttribute(&max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlockOptin, rank);
  KLLM_KERNEL_CHECK(max_shared_mem > 0);
  dim3 blocks(processors, num_experts, 1);

  if (false) {
  }
  CALL_IF_MOE_REPACK(4)
  CALL_IF_MOE_REPACK(8)
  else {
    KLLM_KERNEL_CHECK_WITH_INFO(false, "Unsupported repack config: num_bits = %d", static_cast<int>(num_bits));
  }
}

#undef CALL_IF_MOE_REPACK

std::vector<int64_t> marlin_moe_repack_meta(int64_t size_k, int64_t size_n, int64_t num_bits) {
  int const pack_factor = 32 / num_bits;
  int const tile_sz = 16;
  return {size_k / tile_sz, size_n * tile_sz / pack_factor};
}

// ============================================================================
// Permute scales kernel: reorders quantization scales for Marlin layout.
// Adapted from gptq_marlin.cu permute_scales, extended with num_experts.
// ============================================================================

template <typename T>
__global__ void marlin_moe_permute_scales_kernel(const T* input, T* output,
                                                 const size_t scale_perm_size, const size_t len) {
  size_t base = blockIdx.x * blockDim.x;
  size_t offset = threadIdx.x;

  int64_t scale_perm[64];
  int idx = 0;
  if (scale_perm_size == 64) {
    for (int i = 0; i < 8; i++) {
      for (int j = 0; j < 8; j++) {
        scale_perm[idx++] = i + 8 * j;
      }
    }
  } else {
    for (int i = 0; i < 4; i++) {
      for (int j : {0, 1, 8, 9, 16, 17, 24, 25}) {
        scale_perm[idx++] = 2 * i + j;
      }
    }
  }

  if (base + offset < len) {
    output[base + offset] = input[base + scale_perm[offset]];
  }
}

template <typename T>
void marlin_moe_permute_scales(cudaStream_t stream, const T* input, T* output,
                               int64_t num_experts, size_t k, size_t n, int64_t groupsize) {
  size_t scale_perm_size;
  if (groupsize < static_cast<int64_t>(k) && groupsize != -1) {
    scale_perm_size = 64;
  } else {
    scale_perm_size = 32;
  }
  KLLM_KERNEL_CHECK_WITH_INFO(scale_perm_size != 0, "scale_perm_size must not be 0");
  KLLM_KERNEL_CHECK_WITH_INFO(k % groupsize == 0, "k must be divisible by groupsize");
  size_t len_per_expert = k / groupsize * n;
  size_t total_len = len_per_expert * num_experts;
  KLLM_KERNEL_CHECK_WITH_INFO(len_per_expert % scale_perm_size == 0,
                              "k / groupsize * n must be divisible by scale_perm_size");
  dim3 blockSize(scale_perm_size);
  dim3 gridSize(total_len / scale_perm_size);
  marlin_moe_permute_scales_kernel<T><<<gridSize, blockSize, 0, stream>>>(input, output, scale_perm_size, total_len);
}

// Template instantiations
template void marlin_moe_permute_scales<half>(cudaStream_t, const half*, half*, int64_t, size_t, size_t, int64_t);
template void marlin_moe_permute_scales<__nv_bfloat16>(cudaStream_t, const __nv_bfloat16*, __nv_bfloat16*, int64_t, size_t, size_t, int64_t);
template void marlin_moe_permute_scales<float>(cudaStream_t, const float*, float*, int64_t, size_t, size_t, int64_t);
template void marlin_moe_permute_scales<uint8_t>(cudaStream_t, const uint8_t*, uint8_t*, int64_t, size_t, size_t, int64_t);

// ============================================================================
// MXFP4 scale processing: [0,2,1,3] swizzle + exponent bias for E8M0 scales.
// ============================================================================

__global__ void mxfp4_process_scales_kernel(uint8_t* data, int64_t num_groups) {
  int64_t group_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (group_idx >= num_groups) return;

  int64_t base = group_idx * 4;
  uint8_t v0 = data[base + 0];
  uint8_t v1 = data[base + 1];
  uint8_t v2 = data[base + 2];
  uint8_t v3 = data[base + 3];

  // Swizzle [0,2,1,3] only — no exponent bias for bf16 compute path.
  // The +6 bias is only needed for fp4→fp8 activation scaling (not our case).
  data[base + 0] = v0;
  data[base + 1] = v2;
  data[base + 2] = v1;
  data[base + 3] = v3;
}

void mxfp4_process_scales(cudaStream_t stream, uint8_t* data, int64_t num_elements) {
  int64_t num_groups = num_elements / 4;
  int block = 256;
  int grid = (num_groups + block - 1) / block;
  mxfp4_process_scales_kernel<<<grid, block, 0, stream>>>(data, num_groups);
}

// ============================================================================
// Batched 2D transpose: [num_batches, rows, cols] → [num_batches, cols, rows]
// ============================================================================

template <typename T>
__global__ void transpose_2d_batched_kernel(const T* __restrict__ input, T* __restrict__ output,
                                            int64_t num_batches, int64_t rows, int64_t cols) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = num_batches * rows * cols;
  if (idx >= total) return;

  int64_t batch = idx / (rows * cols);
  int64_t rem = idx % (rows * cols);
  int64_t r = rem / cols;
  int64_t c = rem % cols;

  // input[batch][r][c] → output[batch][c][r]
  int64_t out_idx = batch * (cols * rows) + c * rows + r;
  output[out_idx] = input[idx];
}

void transpose_2d_batched(cudaStream_t stream, const void* input, void* output,
                          int64_t num_batches, int64_t rows, int64_t cols,
                          int element_size) {
  int64_t total = num_batches * rows * cols;
  int block = 256;
  int grid = static_cast<int>((total + block - 1) / block);

  switch (element_size) {
    case 1:
      transpose_2d_batched_kernel<uint8_t><<<grid, block, 0, stream>>>(
          reinterpret_cast<const uint8_t*>(input), reinterpret_cast<uint8_t*>(output),
          num_batches, rows, cols);
      break;
    case 2:
      transpose_2d_batched_kernel<uint16_t><<<grid, block, 0, stream>>>(
          reinterpret_cast<const uint16_t*>(input), reinterpret_cast<uint16_t*>(output),
          num_batches, rows, cols);
      break;
    case 4:
      transpose_2d_batched_kernel<uint32_t><<<grid, block, 0, stream>>>(
          reinterpret_cast<const uint32_t*>(input), reinterpret_cast<uint32_t*>(output),
          num_batches, rows, cols);
      break;
    default:
      KLLM_KERNEL_CHECK_WITH_INFO(false, "Unsupported element_size for transpose: %d", element_size);
  }
}

}  // namespace marlin_moe_wna16
}  // namespace nvidia
}  // namespace llm_kernels
