/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */
#include "csrc/kernels/nvidia/dspark/dspark.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "csrc/kernels/nvidia/others/vllm/main/deepseek_v4/fused_deepseek_v4_knorm_rope_kv_insert_kernel.h"
#include "csrc/utils/nvidia/assert.h"
#include "csrc/utils/nvidia/cuda_utils.h"

namespace llm_kernels {
namespace nvidia {

namespace {

// One thread block per token row; each thread strides over hidden_dim, summing
// the hc_mult middle-axis values and writing the mean into the destination slot.
template <typename T>
__global__ void DSparkCaptureMeanReduceKernel(uint8_t* __restrict__ dst, const uint8_t* __restrict__ src, int hc_mult,
                                              int hidden_dim, int64_t dst_pitch_bytes) {
  const int token_idx = blockIdx.x;
  const T* src_row = reinterpret_cast<const T*>(src) + static_cast<int64_t>(token_idx) * hc_mult * hidden_dim;
  T* dst_row = reinterpret_cast<T*>(dst + token_idx * dst_pitch_bytes);

  const float inv_hc = 1.0f / static_cast<float>(hc_mult);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

  for (int h = threadIdx.x; h < hidden_dim; h += blockDim.x) {
    float acc = 0.0f;
#pragma unroll 4
    for (int m = 0; m < hc_mult; ++m) {
      acc += static_cast<float>(src_row[m * hidden_dim + h]);
    }
    dst_row[h] = static_cast<T>(acc * inv_hc);
  }
}

}  // namespace

template <typename T>
void InvokeDSparkCaptureMeanReduce(void* dst, const void* src, int num_tokens, int hc_mult, int hidden_dim,
                                   int64_t dst_pitch_bytes, cudaStream_t stream) {
  if (num_tokens <= 0) {
    return;
  }
  constexpr int kThreadsPerBlock = 256;
  constexpr auto kernel = DSparkCaptureMeanReduceKernel<T>;
  LAUNCH_KERNEL_WITH_PDL(kernel, num_tokens, kThreadsPerBlock, 0, stream, reinterpret_cast<uint8_t*>(dst),
                         reinterpret_cast<const uint8_t*>(src), hc_mult, hidden_dim, dst_pitch_bytes);
}

template void InvokeDSparkCaptureMeanReduce<__half>(void*, const void*, int, int, int, int64_t, cudaStream_t);
template void InvokeDSparkCaptureMeanReduce<__nv_bfloat16>(void*, const void*, int, int, int, int64_t, cudaStream_t);

template <llm_kernels::utils::KVCacheType KV_DTYPE>
void InvokeDSparkInjectKV(const __nv_bfloat16* kv_in, const __nv_bfloat16* kv_norm_weight, void* swa_kv_cache,
                          const int64_t* positions, const __nv_bfloat16* cos_sin_cache, const int32_t* raw_out_loc,
                          float norm_eps, int num_tokens, int64_t page_stride, int page_bits, cudaStream_t stream) {
  if (num_tokens <= 0) {
    return;
  }
  InvokeFusedKNormRopeKVInsert<KV_DTYPE>(kv_in, kv_norm_weight, swa_kv_cache, positions, cos_sin_cache, raw_out_loc,
                                         norm_eps, num_tokens, page_stride, page_bits, stream);
}

template void InvokeDSparkInjectKV<llm_kernels::utils::KVCacheType::kFp8DsMla>(const __nv_bfloat16*,
                                                                               const __nv_bfloat16*, void*,
                                                                               const int64_t*, const __nv_bfloat16*,
                                                                               const int32_t*, float, int, int64_t, int,
                                                                               cudaStream_t);
template void InvokeDSparkInjectKV<llm_kernels::utils::KVCacheType::kFp8E4M3>(const __nv_bfloat16*,
                                                                              const __nv_bfloat16*, void*,
                                                                              const int64_t*, const __nv_bfloat16*,
                                                                              const int32_t*, float, int, int64_t, int,
                                                                              cudaStream_t);

namespace {

template <int BYTES>
struct ByteVec;
template <>
struct ByteVec<16> {
  using type = int4;
};
template <>
struct ByteVec<8> {
  using type = int2;
};
template <>
struct ByteVec<4> {
  using type = int;
};
template <>
struct ByteVec<2> {
  using type = short;
};

// One block per request, threads strided over hidden_dim. VEC_SIZE = elements per vector load
// (max int4 = 16B = 8 fp16/bf16 elements).
template <typename T, int VEC_SIZE>
__global__ void DSparkMarkovEmbedLookupKernel(T* __restrict__ output, const uint32_t* __restrict__ token_ids,
                                              const T* __restrict__ emb_table, int hidden_dim) {
  using VecT = typename ByteVec<VEC_SIZE * sizeof(T)>::type;

  const int bid = blockIdx.x;
  const uint32_t token = token_ids[bid];
  const int vec_count = hidden_dim / VEC_SIZE;
  const VecT* src_vec = reinterpret_cast<const VecT*>(emb_table + static_cast<int64_t>(token) * hidden_dim);
  VecT* out_vec = reinterpret_cast<VecT*>(output + static_cast<int64_t>(bid) * hidden_dim);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

  for (int v = threadIdx.x; v < vec_count; v += blockDim.x) {
    out_vec[v] = src_vec[v];
  }
}

template <typename T>
void LaunchDSparkMarkovEmbedLookup(T* output, const uint32_t* token_ids, const T* emb_table, int batch_size,
                                   int hidden_dim, cudaStream_t stream) {
  constexpr int kThreadsPerBlock = 256;
  auto aligned = [](const void* p, size_t bytes) { return reinterpret_cast<uintptr_t>(p) % bytes == 0; };

  if (constexpr int kVec = 16 / sizeof(T); hidden_dim % kVec == 0 && aligned(output, 16) && aligned(emb_table, 16)) {
    constexpr auto kernel = DSparkMarkovEmbedLookupKernel<T, kVec>;
    LAUNCH_KERNEL_WITH_PDL(kernel, batch_size, kThreadsPerBlock, 0, stream, output, token_ids, emb_table, hidden_dim);
  } else if (constexpr int kVec = 8 / sizeof(T);
             hidden_dim % kVec == 0 && aligned(output, 8) && aligned(emb_table, 8)) {
    constexpr auto kernel = DSparkMarkovEmbedLookupKernel<T, kVec>;
    LAUNCH_KERNEL_WITH_PDL(kernel, batch_size, kThreadsPerBlock, 0, stream, output, token_ids, emb_table, hidden_dim);
  } else if (constexpr int kVec = 4 / sizeof(T);
             hidden_dim % kVec == 0 && aligned(output, 4) && aligned(emb_table, 4)) {
    constexpr auto kernel = DSparkMarkovEmbedLookupKernel<T, kVec>;
    LAUNCH_KERNEL_WITH_PDL(kernel, batch_size, kThreadsPerBlock, 0, stream, output, token_ids, emb_table, hidden_dim);
  } else {
    constexpr auto kernel = DSparkMarkovEmbedLookupKernel<T, 1>;
    LAUNCH_KERNEL_WITH_PDL(kernel, batch_size, kThreadsPerBlock, 0, stream, output, token_ids, emb_table, hidden_dim);
  }
}

}  // namespace

template <typename T>
void InvokeDSparkMarkovEmbedLookup(void* output, const uint32_t* token_ids, const void* emb_table, int batch_size,
                                   int hidden_dim, cudaStream_t stream) {
  if (batch_size <= 0) {
    return;
  }
  LaunchDSparkMarkovEmbedLookup<T>(reinterpret_cast<T*>(output), token_ids, reinterpret_cast<const T*>(emb_table),
                                   batch_size, hidden_dim, stream);
}

template void InvokeDSparkMarkovEmbedLookup<__half>(void*, const uint32_t*, const void*, int, int, cudaStream_t);
template void InvokeDSparkMarkovEmbedLookup<__nv_bfloat16>(void*, const uint32_t*, const void*, int, int, cudaStream_t);

namespace {

// 2D grid: blockIdx.y = row, blockIdx.x = vocab chunk. Each block covers
// blockDim.x * VEC_SIZE elements per iteration and grid-strides the row.
template <typename BiasT, int VEC_SIZE>
__global__ void DSparkLogitsAddKernel(float* __restrict__ logits, const BiasT* __restrict__ bias, int vocab_size) {
  using LogitsVecT = typename ByteVec<VEC_SIZE * sizeof(float)>::type;
  using BiasVecT = typename ByteVec<VEC_SIZE * sizeof(BiasT)>::type;

  const int row = blockIdx.y;
  const int vec_count = vocab_size / VEC_SIZE;
  const int64_t row_offset = static_cast<int64_t>(row) * vocab_size;
  LogitsVecT* logits_vec_row = reinterpret_cast<LogitsVecT*>(logits + row_offset);
  const BiasVecT* bias_vec_row = reinterpret_cast<const BiasVecT*>(bias + row_offset);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  for (int v = tid; v < vec_count; v += stride) {
    LogitsVecT lv = logits_vec_row[v];
    BiasVecT bv = bias_vec_row[v];
    float* lf = reinterpret_cast<float*>(&lv);
    const BiasT* bf = reinterpret_cast<const BiasT*>(&bv);
#pragma unroll
    for (int e = 0; e < VEC_SIZE; ++e) {
      lf[e] += static_cast<float>(bf[e]);
    }
    logits_vec_row[v] = lv;
  }
}

template <typename BiasT>
void LaunchDSparkLogitsAdd(float* logits, const BiasT* bias, int batch_size, int vocab_size, cudaStream_t stream) {
  constexpr int kThreadsPerBlock = 256;
  auto aligned = [](const void* p, size_t bytes) { return reinterpret_cast<uintptr_t>(p) % bytes == 0; };

  // Caps the column parallelism to avoid over-scheduling.
  constexpr int kMaxColBlocks = 512;
  auto col_blocks = [&](int elems_per_block) {
    const int blocks = (vocab_size + elems_per_block - 1) / elems_per_block;
    return blocks > kMaxColBlocks ? kMaxColBlocks : blocks;
  };

  // VEC_SIZE picked so that both float and BiasT vector loads/stores remain 16B-aligned when possible.
  const int64_t logits_row_bytes = static_cast<int64_t>(vocab_size) * sizeof(float);
  const int64_t bias_row_bytes = static_cast<int64_t>(vocab_size) * sizeof(BiasT);
  if (aligned(logits, 16) && aligned(bias, 8) && logits_row_bytes % 16 == 0 && bias_row_bytes % 8 == 0) {
    constexpr int kVec = 4;  // 4 floats = 16B, 4 bf16/fp16 = 8B
    constexpr auto kernel = DSparkLogitsAddKernel<BiasT, kVec>;
    dim3 grid(col_blocks(kThreadsPerBlock * kVec), batch_size);
    LAUNCH_KERNEL_WITH_PDL(kernel, grid, kThreadsPerBlock, 0, stream, logits, bias, vocab_size);
  } else if (aligned(logits, 8) && aligned(bias, 4) && logits_row_bytes % 8 == 0 && bias_row_bytes % 4 == 0) {
    constexpr int kVec = 2;
    constexpr auto kernel = DSparkLogitsAddKernel<BiasT, kVec>;
    dim3 grid(col_blocks(kThreadsPerBlock * kVec), batch_size);
    LAUNCH_KERNEL_WITH_PDL(kernel, grid, kThreadsPerBlock, 0, stream, logits, bias, vocab_size);
  } else {
    constexpr auto kernel = DSparkLogitsAddKernel<BiasT, 1>;
    dim3 grid(col_blocks(kThreadsPerBlock), batch_size);
    LAUNCH_KERNEL_WITH_PDL(kernel, grid, kThreadsPerBlock, 0, stream, logits, bias, vocab_size);
  }
}

}  // namespace

template <typename BiasT>
void InvokeDSparkLogitsAdd(float* logits, const void* bias, int batch_size, int vocab_size, cudaStream_t stream) {
  if (batch_size <= 0) {
    return;
  }
  LaunchDSparkLogitsAdd<BiasT>(logits, reinterpret_cast<const BiasT*>(bias), batch_size, vocab_size, stream);
}

template void InvokeDSparkLogitsAdd<__half>(float*, const void*, int, int, cudaStream_t);
template void InvokeDSparkLogitsAdd<__nv_bfloat16>(float*, const void*, int, int, cudaStream_t);

}  // namespace nvidia
}  // namespace llm_kernels
