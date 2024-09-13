/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <unordered_map>

#include "acl/acl.h"

#include "csrc/kernels/ascend/rotary_embedding/rotary_embedding_kernel.h"
#include "csrc/utils/ascend/common.h"

namespace llm_kernels {
namespace ascend {

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
  aclrtStream stream;

  int64_t* positions;
  T* query_;
  T* key_;
  T* workspace;
  int num_tokens_;

  RotaryEmbeddingType rotary_embedding_type = RotaryEmbeddingType::DEFAULT;
  float scaling_factor = 1.0f;

  RotaryEmbeddingTilingConfig tiling_config;
};

template <typename T>
class AscendCRotaryEmbedding {
 public:
  void SetConfig(T* cos_sin_cache,  // temp buffer, [max_position_embeddings, rotary_dim]
                 const int rotary_dim, const int max_position_embeddings, const float base, const int head_size,
                 const int num_heads, const int num_kv_heads, const int stride_size, const bool is_neox,
                 aclrtStream& stream, const RotaryEmbeddingType rotary_embedding_type = RotaryEmbeddingType::DEFAULT,
                 const float scaling_factor = 1.0f);

  void SetInput(int64_t* positions,  // [num_tokens]
                T* query,            // [num_tokens, num_heads * head_size]
                T* key,              // [num_tokens, num_kv_heads * head_size]
                int num_tokens, aclrtStream& stream);

  void Forward();

  ~AscendCRotaryEmbedding() {
    if (tiling_config_device_ptr != nullptr) {
      ACL_CHECK_RET(aclrtFree(tiling_config_device_ptr));
      tiling_config_device_ptr = nullptr;
    }
  }

 private:
  RotaryEmbeddingParam<T> params_;
  uint8_t* tiling_config_device_ptr{nullptr};
  uint8_t* workspace_device_ptr{nullptr};
};

}  // namespace ascend
}  // namespace llm_kernels
