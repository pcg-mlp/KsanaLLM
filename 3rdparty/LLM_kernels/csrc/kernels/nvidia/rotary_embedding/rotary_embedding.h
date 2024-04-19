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

#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace llm_kernels {
namespace nvidia {

enum RotaryEmbeddingType { DEFAULT, LINEAR_SCALING, DYNAMIC_NTK_SCALING };

template <typename T>
struct RotaryEmbeddingParam {
  T* cos_sin_cache;  // [max_position_embeddings, rotary_dim]
  int rotary_dim;
  int max_position_embeddings;
  int head_size;
  int num_heads;
  int num_kv_heads;
  int64_t query_stride;
  int64_t key_stride;
  float base;
  bool is_neox;
  cudaStream_t stream;

  const int64_t* positions;
  T* query_;
  T* key_;
  int num_tokens_;

  RotaryEmbeddingType rotary_embedding_type = RotaryEmbeddingType::DEFAULT;
  float scaling_factor = 1.0f;
};

template <typename T>
class RotaryEmbeddingCuda {
 public:
  void SetConfig(T* cos_sin_cache,  // temp buffer, [max_position_embeddings, rotary_dim]
                 const int rotary_dim, const int max_position_embeddings, const float base, const int head_size,
                 const int num_heads, const int num_kv_heads, const int stride_size, const bool is_neox,
                 cudaStream_t& stream, const RotaryEmbeddingType rotary_embedding_type = RotaryEmbeddingType::DEFAULT,
                 const float scaling_factor = 1.0f);

  void SetInput(const int64_t* positions,  // [batch_size, seq_len] or [num_tokens]
                T* query,  // [batch_size, seq_len, num_heads * head_size] or [num_tokens, num_heads * head_size]
                T* key,    // [batch_size, seq_len, num_kv_heads * head_size] or [num_tokens, num_kv_heads * head_size]
                int num_tokens, cudaStream_t& stream);

  void Forward();

 private:
  RotaryEmbeddingParam<T> params_;
};

}  // namespace nvidia
}  // namespace llm_kernels
