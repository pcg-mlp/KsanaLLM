/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include "greedy.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <cub/cub.cuh>

#include "csrc/utils/nvidia/cuda_utils.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace nvidia {

template <typename T>
using ArgMaxPair = cub::KeyValuePair<int, T>;  // (idx, val)

template <typename T>
__global__ void InvokeArgMaxReduceKernel(const T* input, const int32_t batch_size, const int32_t vocab_size,
                                         uint32_t* result) {
  using BlockReduce = cub::BlockReduce<ArgMaxPair<T>, DEFAULT_CUDA_BLOCK_THREADS_NUM>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  // First reduce in each thread.
  const int offset = blockIdx.x * vocab_size;
  int idx = 0;
  T val = input[offset];
  for (int compare_idx = threadIdx.x; compare_idx < vocab_size; compare_idx += blockDim.x) {
    T compare_val = input[offset + compare_idx];
    if (val < compare_val) {
      idx = compare_idx;
      val = compare_val;
    }
  }

  // Then reduce in the block.
  cub::ArgMax argmax_op;
  idx = BlockReduce(temp_storage).Reduce(ArgMaxPair<T>{idx, val}, argmax_op).key;

  // Write result to global memory.
  if (threadIdx.x == 0) {
    result[blockIdx.x] = idx;
  }
}

template <typename T>
void InvokeArgMaxReduce(const T* input, const int32_t batch_size, const int32_t vocab_size, uint32_t* result,
                        cudaStream_t& stream) {
  dim3 grid(batch_size);
  dim3 block(DEFAULT_CUDA_BLOCK_THREADS_NUM);

  InvokeArgMaxReduceKernel<<<grid, block, 0, stream>>>(input, batch_size, vocab_size, result);
}

#define INSTANTIATE_INVOKE_ARG_MAX_REDUCE(T)                                                           \
  template void InvokeArgMaxReduce(const T* input, const int32_t batch_size, const int32_t vocab_size, \
                                   uint32_t* result, cudaStream_t& stream);

INSTANTIATE_INVOKE_ARG_MAX_REDUCE(float);
INSTANTIATE_INVOKE_ARG_MAX_REDUCE(half);
#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_ARG_MAX_REDUCE(__nv_bfloat16);
#endif

#undef INSTANTIATE_INVOKE_ARG_MAX_REDUCE

}  // namespace nvidia
}  // namespace llm_kernels
