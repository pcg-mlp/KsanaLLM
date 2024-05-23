/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */
#include <math.h>

#include "fused_rotary_embedding.h"

#include "aclrtlaunch_InvokeRotaryEmbeddingHalfKernel.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace ascend {

template <typename T>
void RotaryEmbeddingAscendC<T>::SetConfig(T* cos_sin_cache, const int rotary_dim, const int max_position_embeddings,
                                          const float base, const int head_size, const int num_heads,
                                          const int num_kv_heads, const int stride_size, const bool is_neox,
                                          aclrtStream& stream, const RotaryEmbeddingType rotary_embedding_type,
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

  size_t extend_max_len = params_.max_position_embeddings;
  size_t inner_iter_num = std::min(params_.rotary_dim / 2, 512);
  float new_base = params_.base;
  float scaling = 1.0f;
  // https://github.com/vllm-project/vllm/blob/523e30ea0c5abcb447763dcd9a77b54d5c5f3239/vllm/model_executor/layers/rotary_embedding.py#L219
  if (params_.rotary_embedding_type == RotaryEmbeddingType::DYNAMIC_NTK_SCALING) {
    extend_max_len = params_.max_position_embeddings * params_.scaling_factor;
    new_base = std::pow(params_.base * ((params_.scaling_factor * extend_max_len / params_.max_position_embeddings) -
                                        (params_.scaling_factor - 1)),
                        (params_.rotary_dim / (params_.rotary_dim - 2)));
  }
  if (params_.rotary_embedding_type == RotaryEmbeddingType::LINEAR_SCALING) {
    extend_max_len = params_.max_position_embeddings * params_.scaling_factor;
    scaling = params_.scaling_factor;
  }

  std::vector<T> cos_sin_cache_host(max_position_embeddings * rotary_dim, 0u);
  for (size_t token_idx = 0; token_idx < extend_max_len; ++token_idx) {
    int pos = token_idx;
    for (size_t rid = 0; rid < rotary_dim / 2; ++rid) {
      float inv_freq = 1.0 / std::pow(new_base, rid * 2 / (float)rotary_dim);
      float freq = pos * inv_freq / scaling;
      if (std::is_same<T, aclFloat16>::value) {
        cos_sin_cache_host[pos * rotary_dim + rid] = aclFloatToFloat16(std::cos(freq));
        cos_sin_cache_host[pos * rotary_dim + rotary_dim / 2 + rid] = aclFloatToFloat16(std::sin(freq));

        // cos_sin_cache_host[pos * rotary_dim + rid] = aclFloatToFloat16(1.0f);
        // cos_sin_cache_host[pos * rotary_dim + rotary_dim / 2 + rid] = aclFloatToFloat16(1.0f);
      } else if (std::is_same<T, float>::value) {
        cos_sin_cache_host[pos * rotary_dim + rid] = std::cos(freq);
        cos_sin_cache_host[pos * rotary_dim + rotary_dim / 2 + rid] = std::sin(freq);
      } else {
        throw std::invalid_argument("Invalid rope compute type, only support float16 or float32.");
      }
    }
  }

  ACL_CHECK_RET(aclrtMemcpyAsync(params_.cos_sin_cache, cos_sin_cache_host.size() * sizeof(T),
                                 cos_sin_cache_host.data(), cos_sin_cache_host.size() * sizeof(T),
                                 ACL_MEMCPY_HOST_TO_DEVICE, stream));
  ACL_CHECK_RET(aclrtSynchronizeStream(stream));

  ACL_CHECK_RET(
      aclrtMalloc((void**)&tiling_config_device_ptr, sizeof(RotaryEmbeddingTilingConfig), ACL_MEM_MALLOC_HUGE_FIRST));

  params_.tiling_config.hidden_units_num = params_.rotary_dim / 2;
  params_.tiling_config.rotary_dim = params_.rotary_dim;
  params_.tiling_config.num_heads = params_.num_heads;
  params_.tiling_config.num_kv_heads = params_.num_kv_heads;
  params_.tiling_config.head_size = params_.head_size;
}

template void RotaryEmbeddingAscendC<float>::SetConfig(float* cos_sin_cache, const int rotary_dim,
                                                       const int max_position_embeddings, const float base,
                                                       const int head_size, const int num_heads, const int num_kv_heads,
                                                       const int stride_size, const bool is_neox, aclrtStream& stream,
                                                       const RotaryEmbeddingType rotary_embedding_type,
                                                       const float scaling_factor);
template void RotaryEmbeddingAscendC<aclFloat16>::SetConfig(
    aclFloat16* cos_sin_cache, const int rotary_dim, const int max_position_embeddings, const float base,
    const int head_size, const int num_heads, const int num_kv_heads, const int stride_size, const bool is_neox,
    aclrtStream& stream, const RotaryEmbeddingType rotary_embedding_type, const float scaling_factor);

template <typename T>
void RotaryEmbeddingAscendC<T>::SetInput(int64_t* positions,  // [num_tokens]
                                         T* query,            // [num_tokens, num_heads * head_size]
                                         T* key,              // [num_tokens, num_kv_heads * head_size]
                                         int num_tokens, aclrtStream& stream) {
  params_.positions = positions;
  params_.query_ = query;
  params_.key_ = key;
  params_.num_tokens_ = num_tokens;
  params_.stream = stream;

  params_.tiling_config.seq_len = num_tokens;
}

template void RotaryEmbeddingAscendC<float>::SetInput(int64_t* positions, float* query, float* key, int num_tokens,
                                                      aclrtStream& stream);
template void RotaryEmbeddingAscendC<aclFloat16>::SetInput(int64_t* positions, aclFloat16* query, aclFloat16* key,
                                                           int num_tokens, aclrtStream& stream);

template <typename T>
void RotaryEmbeddingAscendC<T>::Forward() {
  RotaryEmbeddingTilingConfig* buf = &(params_.tiling_config);
  ACL_CHECK_RET(aclrtMemcpyAsync(tiling_config_device_ptr, sizeof(RotaryEmbeddingTilingConfig), buf,
                                 sizeof(RotaryEmbeddingTilingConfig), ACL_MEMCPY_HOST_TO_DEVICE, params_.stream));
  if (std::is_same<T, aclFloat16>::value) {
    // NOTE(karlluo): each token handle by one ai core
    ACL_CHECK_RET(ACLRT_LAUNCH_KERNEL(InvokeRotaryEmbeddingHalfKernel)(
        params_.num_tokens_, params_.stream, params_.positions, params_.query_, params_.key_, params_.cos_sin_cache,
        tiling_config_device_ptr, params_.query_, params_.key_));
  } else if (std::is_same<T, float>::value) {
  } else {
    throw std::invalid_argument("Invalid rope compute type, only support float16 or float32.");
  }
}

template void RotaryEmbeddingAscendC<float>::Forward();
template void RotaryEmbeddingAscendC<aclFloat16>::Forward();

}  // namespace ascend
}  // namespace llm_kernels
