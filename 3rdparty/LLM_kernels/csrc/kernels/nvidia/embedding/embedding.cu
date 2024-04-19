/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include "embedding.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace nvidia {

template <typename T>
__global__ void LookupFusedEmbeddingKernelWithCSRKernel(T* output_hidden_units, const T* embedding_table,
                                                        const T* pos_table,
                                                        InvokeInputIdsEmbeddingLookupPosEncodingParam<T> prompt_param,
                                                        const int32_t* input_ids, const int32_t start_step,
                                                        const size_t* ids_offset, const int32_t batch_size,
                                                        const uint32_t hidden_units, const size_t vocab_size,
                                                        const size_t vocab_id) {
  // NOTE(karlluo): config of grid and block
  // grid: (min(batch_size, 65536), 32);
  // block: (min(hidden_units, 512));
  size_t input_ids_idx_offset = ids_offset[blockIdx.x];
  size_t ids_num = ids_offset[blockIdx.x + 1] - input_ids_idx_offset;
  int32_t step = start_step + blockIdx.y;

  // TODO(karlluo): optimization unroll ?
#pragma unroll
  for (uint32_t token_id = blockIdx.y; token_id < ids_num; token_id += gridDim.y) {
    int32_t real_token_id = input_ids[input_ids_idx_offset + token_id];
    // on each GPU, emb range is [vocab_id * vocab_size, (vocab_id + 1) * vocab_size)
    // read_id bigger than the vocabulary size, handle next id
    if (real_token_id >= ((vocab_id + 1) * vocab_size) || real_token_id < (vocab_id * vocab_size)) {
      continue;
    }
    step = start_step + token_id;

    // copy emb vec value
    // TODO(karlluo): optimization veterize ?
#pragma unroll
    for (uint32_t emb_id = threadIdx.x; emb_id < hidden_units; emb_id += blockDim.x) {
      T emb_vec_val = embedding_table[(real_token_id - vocab_id * vocab_size) * hidden_units + emb_id];
      T pos_emb_vec_val = pos_table == nullptr ? (T)0.f : pos_table[(step - 1) * hidden_units + emb_id];
      // NOTE(karlluo): half has not static cast in kernel
      output_hidden_units[(input_ids_idx_offset + token_id) * hidden_units + emb_id] = emb_vec_val + pos_emb_vec_val;
    }
  }
}

template <typename T>
void LookupFusedEmbeddingWithCSRInputs(T* output_hidden_units, const T* embedding_table, const T* pos_table,
                                       InvokeInputIdsEmbeddingLookupPosEncodingParam<T> prompt_param,
                                       const int32_t* input_ids, const int32_t start_step, const size_t* ids_offsets,
                                       const int32_t batch_size, const uint32_t hidden_units, const size_t vocab_size,
                                       const size_t vocab_id, cudaStream_t stream) {
  // each block handle one sample among batch's token last hidden units
  constexpr int32_t tokens_stride = 32;
  dim3 grid(min(static_cast<int32_t>(batch_size), DEFAULT_CUDA_GPU_DEVICE_MAX_BLOCKS_NUM), tokens_stride);
  dim3 block(min(hidden_units, DEFAULT_CUDA_BLOCK_THREADS_NUM));

  LookupFusedEmbeddingKernelWithCSRKernel<T>
      <<<grid, block, 0, stream>>>(output_hidden_units, embedding_table, pos_table, prompt_param, input_ids, start_step,
                                   ids_offsets, batch_size, hidden_units, vocab_size, vocab_id);
}

#define INSTANTIATE_LOOKUP_FUSED_EMBEDDING_WITH_CSR_INPUTS(T)                                                     \
  template void LookupFusedEmbeddingWithCSRInputs(                                                                \
      T* output_hidden_units, const T* embedding_table, const T* pos_table,                                       \
      InvokeInputIdsEmbeddingLookupPosEncodingParam<T> prompt_param, const int32_t* input_ids,                    \
      const int32_t start_step, const size_t* ids_offsets, const int32_t batch_size, const uint32_t hidden_units, \
      const size_t vocab_size, const size_t vocab_id, cudaStream_t stream);

INSTANTIATE_LOOKUP_FUSED_EMBEDDING_WITH_CSR_INPUTS(float);
INSTANTIATE_LOOKUP_FUSED_EMBEDDING_WITH_CSR_INPUTS(half);
#ifdef ENABLE_BF16
INSTANTIATE_LOOKUP_FUSED_EMBEDDING_WITH_CSR_INPUTS(__nv_bfloat16);
#endif

#undef INSTANTIATE_LOOKUP_FUSED_EMBEDDING_WITH_CSR_INPUTS

}  // namespace nvidia
}  // namespace llm_kernels