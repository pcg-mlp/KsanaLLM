/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include "csrc/kernels/nvidia/assemble_last_token/assemble_last_token.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace nvidia {

template <typename T>
__global__ void AssembleLastTokenKernel(const T* input, const size_t* ids_offsets, const size_t* prefix_offsets,
                                        const int32_t batch_size, const int32_t hidden_units_num, T* output) {
  // NOTE(karlluo): config of grid and block
  // grid: (min(batch_size, 65536));
  // block: (min(hidden_units, 512));
  // batch_id is blockIdx.x
  size_t ids_num = ids_offsets[blockIdx.x + 1] - prefix_offsets[blockIdx.x + 1] - 1ul;

  // get last token hidden units offset
  const T* input_token_hidden_units_offset = input + (ids_num * hidden_units_num);

  // TODO(karlluo): optimization unroll ?
  for (uint32_t emb_id = threadIdx.x; emb_id < hidden_units_num; emb_id += blockDim.x) {
    output[blockIdx.x * hidden_units_num + emb_id] = input_token_hidden_units_offset[emb_id];
  }
}

template <typename T>
void AssembleLastToken(const T* input, const size_t* ids_offsets, const size_t* prefix_offsets,
                       const int32_t batch_size, const int32_t hidden_units_num, T* output, cudaStream_t& stream) {
  // each block handle one sample among batch's token last hidden units
  dim3 grid(min(static_cast<int32_t>(batch_size), DEFAULT_CUDA_GPU_DEVICE_MAX_BLOCKS_NUM));
  dim3 block(min(hidden_units_num, DEFAULT_CUDA_BLOCK_THREADS_NUM));

  AssembleLastTokenKernel<T>
      <<<grid, block, 0, stream>>>(input, ids_offsets, prefix_offsets, batch_size, hidden_units_num, output);
}

#define INSTANTIATE_ASSEMBLE_LAST_TOKEN(T)                                                                 \
  template void AssembleLastToken(const T* input, const size_t* ids_offsets, const size_t* prefix_offsets, \
                                  const int32_t batch_size, const int32_t hidden_units_num, T* output,     \
                                  cudaStream_t& stream);

INSTANTIATE_ASSEMBLE_LAST_TOKEN(float);
INSTANTIATE_ASSEMBLE_LAST_TOKEN(half);
#ifdef ENABLE_BF16
INSTANTIATE_ASSEMBLE_LAST_TOKEN(__nv_bfloat16);
#endif

#undef INSTANTIATE_ASSEMBLE_LAST_TOKEN

}  // namespace nvidia
}  // namespace llm_kernels
