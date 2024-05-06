/*
 * Modify from
 * https://github.com/vllm-project/vllm/blob/v0.2.3/csrc/pos_encoding_kernels.cu
 * Copyright (c) 2024, Tencent Inc.
 * Copyright (c) 2023, The vLLM team.
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

#include "rotary_embedding.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <cmath>

#include "csrc/utils/nvidia/cuda_utils.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace nvidia {

template <typename T, bool IS_NEOX>
inline __device__ void ApplyRotaryEmbedding(T* __restrict__ arr, const T* __restrict__ cos_ptr,
                                            const T* __restrict__ sin_ptr, int rot_offset, int embed_dim) {
  int x_index, y_index;
  T cos, sin;
  if (IS_NEOX) {
    // GPT-NeoX style rotary embedding.
    x_index = rot_offset;
    y_index = embed_dim + rot_offset;
    cos = __ldg(cos_ptr + x_index);
    sin = __ldg(sin_ptr + x_index);
  } else {
    // GPT-J style rotary embedding.
    x_index = 2 * rot_offset;
    y_index = 2 * rot_offset + 1;
    cos = __ldg(cos_ptr + x_index / 2);
    sin = __ldg(sin_ptr + x_index / 2);
  }

  const T x = arr[x_index];
  const T y = arr[y_index];
  arr[x_index] = x * cos - y * sin;
  arr[y_index] = y * cos + x * sin;
}

template <typename T, bool IS_NEOX>
__global__ void InvokeRotaryEmbeddingKernel(
    const int64_t* __restrict__ positions,  // [batch_size, seq_len] or [num_tokens]
    T* __restrict__ query,  // [batch_size, seq_len, num_heads, head_size] or [num_tokens, num_heads, head_size]
    T* __restrict__ key,    // [batch_size, seq_len, num_kv_heads, head_size] or [num_tokens, num_kv_heads,
                            // head_size]
    const T* __restrict__ cos_sin_cache,  // [max_position_embeddings, 2, rotary_dim // 2]
    const int rotary_dim, const int64_t query_stride, const int64_t key_stride, const int num_heads,
    const int num_kv_heads, const int head_size) {
  // Each thread block is responsible for one token.
  const int token_idx = blockIdx.x;
  int64_t pos = positions[token_idx];
  const T* cache_ptr = cos_sin_cache + pos * rotary_dim;

  const int embed_dim = rotary_dim / 2;
  const T* cos_ptr = cache_ptr;
  const T* sin_ptr = cache_ptr + embed_dim;

  const int nq = num_heads * embed_dim;
  for (int i = threadIdx.x; i < nq; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int64_t token_head = token_idx * query_stride + head_idx * head_size;
    const int rot_offset = i % embed_dim;
    ApplyRotaryEmbedding<T, IS_NEOX>(query + token_head, cos_ptr, sin_ptr, rot_offset, embed_dim);
  }

  const int nk = num_kv_heads * embed_dim;
  for (int i = threadIdx.x; i < nk; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int64_t token_head = token_idx * key_stride + head_idx * head_size;
    const int rot_offset = i % embed_dim;
    ApplyRotaryEmbedding<T, IS_NEOX>(key + token_head, cos_ptr, sin_ptr, rot_offset, embed_dim);
  }
}

template <typename T>
void LaunchRotaryEmbedding(const RotaryEmbeddingParam<T>& params) {
  dim3 grid(params.num_tokens_);
  dim3 block(std::min(params.num_heads * params.rotary_dim / 2, 512));
  if (params.is_neox) {
    InvokeRotaryEmbeddingKernel<T, true><<<grid, block, 0, params.stream>>>(
        params.positions, params.query_, params.key_, params.cos_sin_cache, params.rotary_dim, params.query_stride,
        params.key_stride, params.num_heads, params.num_kv_heads, params.head_size);
  } else {
    InvokeRotaryEmbeddingKernel<T, false><<<grid, block, 0, params.stream>>>(
        params.positions, params.query_, params.key_, params.cos_sin_cache, params.rotary_dim, params.query_stride,
        params.key_stride, params.num_heads, params.num_kv_heads, params.head_size);
  }
}

template void LaunchRotaryEmbedding<float>(const RotaryEmbeddingParam<float>& params);
template void LaunchRotaryEmbedding<half>(const RotaryEmbeddingParam<half>& params);
template void LaunchRotaryEmbedding<__nv_bfloat16>(const RotaryEmbeddingParam<__nv_bfloat16>& params);

template <typename T>
__global__ void InvokeComputeCosSinWithCacheKernel(T* __restrict__ cos_sin_cache, const int rotary_dim,
                                                   const int max_position_embeddings, const float base) {
  int pos = blockIdx.x;
  for (int rid = threadIdx.x; rid < rotary_dim / 2; rid += blockDim.x) {
    float inv_freq = 1.0 / pow(base, rid * 2 / (float)rotary_dim);
    float freq = pos * inv_freq;
    cos_sin_cache[pos * rotary_dim + rid] = (T)cos(freq);
    cos_sin_cache[pos * rotary_dim + rotary_dim / 2 + rid] = (T)sin(freq);
  }
}

template <typename T>
void ComputeCosSinWithCache(const RotaryEmbeddingParam<T>& params) {
  size_t extend_max_len = params.max_position_embeddings;
  dim3 block(std::min(params.rotary_dim / 2, DEFAULT_CUDA_BLOCK_THREADS_NUM));

  float base = params.base;
  // https://github.com/vllm-project/vllm/blob/523e30ea0c5abcb447763dcd9a77b54d5c5f3239/vllm/model_executor/layers/rotary_embedding.py#L219
  if (params.rotary_embedding_type == RotaryEmbeddingType::DYNAMIC_NTK_SCALING) {
    extend_max_len = params.max_position_embeddings * params.scaling_factor;
    base = std::pow(params.base * ((params.scaling_factor * extend_max_len / params.max_position_embeddings) -
                                   (params.scaling_factor - 1)),
                    (params.rotary_dim / (params.rotary_dim - 2)));
  }
  dim3 grid(extend_max_len);
  InvokeComputeCosSinWithCacheKernel<T>
      <<<grid, block, 0, params.stream>>>(params.cos_sin_cache, params.rotary_dim, extend_max_len, base);
}

template void ComputeCosSinWithCache<float>(const RotaryEmbeddingParam<float>& params);
template void ComputeCosSinWithCache<half>(const RotaryEmbeddingParam<half>& params);
template void ComputeCosSinWithCache<__nv_bfloat16>(const RotaryEmbeddingParam<__nv_bfloat16>& params);

template <typename T>
void RotaryEmbeddingCuda<T>::SetInput(
    const int64_t* positions,  // [batch_size, seq_len] or [num_tokens]
    T* query,                  // [batch_size, seq_len, num_heads * head_size] or [num_tokens, num_heads * head_size]
    T* key,  // [batch_size, seq_len, num_kv_heads * head_size] or [num_tokens, num_kv_heads * head_size]
    int num_tokens, cudaStream_t& stream) {
  params_.positions = positions;
  params_.query_ = query;
  params_.key_ = key;
  params_.num_tokens_ = num_tokens;
  params_.stream = stream;
}

template void RotaryEmbeddingCuda<float>::SetInput(const int64_t* positions, float* query, float* key, int num_tokens,
                                                   cudaStream_t& stream);
template void RotaryEmbeddingCuda<half>::SetInput(const int64_t* positions, half* query, half* key, int num_tokens,
                                                  cudaStream_t& stream);
template void RotaryEmbeddingCuda<__nv_bfloat16>::SetInput(const int64_t* positions, __nv_bfloat16* query,
                                                           __nv_bfloat16* key, int num_tokens, cudaStream_t& stream);

template <typename T>
void RotaryEmbeddingCuda<T>::Forward() {
  LaunchRotaryEmbedding(params_);
}

template void RotaryEmbeddingCuda<float>::Forward();
template void RotaryEmbeddingCuda<half>::Forward();
template void RotaryEmbeddingCuda<__nv_bfloat16>::Forward();

template <typename T>
void RotaryEmbeddingCuda<T>::SetConfig(T* cos_sin_cache, const int rotary_dim, const int max_position_embeddings,
                                       const float base, const int head_size, const int num_heads,
                                       const int num_kv_heads, const int stride_size, const bool is_neox,
                                       cudaStream_t& stream, const RotaryEmbeddingType rotary_embedding_type,
                                       const float scaling_factor) {
  params_.cos_sin_cache = cos_sin_cache;
  params_.rotary_dim = rotary_dim;
  params_.max_position_embeddings = max_position_embeddings;
  params_.base = base;
  params_.head_size = head_size;
  params_.num_heads = num_heads;
  params_.num_kv_heads = num_kv_heads;
  params_.is_neox = is_neox;
  params_.stream = stream;
  params_.query_stride = stride_size;
  params_.key_stride = stride_size;
  params_.rotary_embedding_type = rotary_embedding_type;
  params_.scaling_factor = scaling_factor;
  if (params_.rotary_embedding_type == RotaryEmbeddingType::LINEAR_SCALING) {
    throw std::invalid_argument("Linear scaling rotary embedding is not support.");
  }
  ComputeCosSinWithCache(params_);
}

template void RotaryEmbeddingCuda<float>::SetConfig(float* cos_sin_cache, const int rotary_dim,
                                                    const int max_position_embeddings, const float base,
                                                    const int head_size, const int num_heads, const int num_kv_heads,
                                                    const int stride_size, const bool is_neox, cudaStream_t& stream,
                                                    const RotaryEmbeddingType rotary_embedding_type,
                                                    const float scaling_factor);
template void RotaryEmbeddingCuda<half>::SetConfig(half* cos_sin_cache, const int rotary_dim,
                                                   const int max_position_embeddings, const float base,
                                                   const int head_size, const int num_heads, const int num_kv_heads,
                                                   const int stride_size, const bool is_neox, cudaStream_t& stream,
                                                   const RotaryEmbeddingType rotary_embedding_type,
                                                   const float scaling_factor);
template void RotaryEmbeddingCuda<__nv_bfloat16>::SetConfig(
    __nv_bfloat16* cos_sin_cache, const int rotary_dim, const int max_position_embeddings, const float base,
    const int head_size, const int num_heads, const int num_kv_heads, const int stride_size, const bool is_neox,
    cudaStream_t& stream, const RotaryEmbeddingType rotary_embedding_type, const float scaling_factor);

}  // namespace nvidia
}  // namespace llm_kernels
