/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#ifdef ENABLE_CUDA
#  include <optional>

#  include "csrc/kernels/nvidia/alibi/alibi.h"
#  include "csrc/kernels/nvidia/rotary_embedding/rotary_embedding.h"
#elif ENABLE_ACL
#  include "csrc/kernels/ascend/attention/attention.h"
#endif

#include "csrc/utils/quant_type.h"
#include "layers/base_layer.h"

namespace ksana_llm {

// The positional encoding.
enum PositionEncoding { LEARNED_ABSOLUTE = 0, ROPE = 1, ALIBI = 2, NO_ROPE = 3 };

#ifdef ENABLE_KUNLUN
// Frees the process-wide neox-format cos/sin caches shared by all attention layers.
// Not needed for normal shutdown (the driver reclaims device memory on process exit);
// provided for a future dynamic model unload path.
void ReleaseNeoxCosSinCache();
#endif

template <typename T, template <typename, typename, llm_kernels::utils::KVCacheType> class ATTENTION_LAYER>
std::shared_ptr<BaseLayer> CreateAttentionLayer(DataType kv_cache_dtype) {
  switch (kv_cache_dtype) {
#ifdef ENABLE_CUDA
    case TYPE_FP8_E5M2:
      return std::make_shared<ATTENTION_LAYER<T, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2>>();
    case TYPE_FP8_E4M3:
      return std::make_shared<ATTENTION_LAYER<T, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3>>();
#endif
    default:
      return std::make_shared<ATTENTION_LAYER<T, T, llm_kernels::utils::KVCacheType::kAuto>>();
  }
}

class AttentionLayer : public BaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;

#ifdef ENABLE_CUDA
  // for deepseek yarn - static helper functions
  static float deepseek_yarn_get_mscale(const float scale, const float mscale);

  static float common_yarn_get_mscale(const float scale);
#endif

 private:
  template <typename T>
  Status InitT(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
               std::shared_ptr<Context> context, int rank);

 protected:
  int layer_num_;
  int layer_index_;
  int block_size_;
  int block_token_num_;
  int num_heads_;
  int num_kv_heads_;
  int head_size_;
  int stride_size_;
  size_t tensor_para_size_;
  int max_position_embeddings_;
  float base_;
  bool use_qk_norm_;                           // Check if normlize the attention out q and k.
  float layernorm_eps_;                        // The epsilon value used in layer normalization layers.
  bool enable_qk_pre_norm_before_rotary_pos_;  // Whether to normalize q and k before rotary position
  // kv_cache storage type and kv scale
  DataType kv_cache_dtype_;
  float k_scale_;
  float v_scale_;

  bool is_causal_{true};
  // mla config params
  uint32_t q_lora_rank_;
  uint32_t kv_lora_rank_;
  uint32_t qk_nope_head_dim_;
  uint32_t qk_rope_head_dim_;
  uint32_t v_head_dim_;
  float attn_scale_;
  bool attn_scale_initialized_ = false;  // Tracks whether attn_scale_ has been explicitly set
  // TODO(winminkong): the matmul op will be removed from mla attn in the subsequent steps.
  QuantMode mm_quant_mode_;

#ifdef ENABLE_CUDA
  std::optional<llm_kernels::nvidia::RotaryEmbeddingCuda> rotary_embedding_cuda_;
  std::optional<void*> alibi_slopes_;
#endif

#ifdef ENABLE_ACL
  size_t max_batch_size_;
  bool is_multi_token_forward_;
#endif

#ifdef ENABLE_KUNLUN
  // Two layouts of the SAME RoPE table (not two different RoPE variants) — the
  // standalone and the fused kernels require different memory layouts:
  //   cos_sin_cache_ptr_      [max_pos, rotary_dim]     — half cos | half sin, one
  //                                                        value per frequency. Consumed by
  //                                                        infer_ops::rotary_embedding_neox.
  //   cos_sin_cache_neox_ptr_ [2, 1, max_pos, rotary_dim] — separate cos/sin planes, each
  //                                                        frequency duplicated across both
  //                                                        halves (full-width). Required by
  //                                                        infer_ops::split_norm_rope_neox.
  // Both are derived from the same base_/rotary_dim, so they stay numerically consistent.
  // The neox one is only built when a fused QK-norm path is active; it is shared across
  // all layers via g_neox_cache_map (see attention_layer.cpp).
  void* cos_sin_cache_ptr_ = nullptr;
  void* cos_sin_cache_neox_ptr_ = nullptr;

  // Fused split(QKV)+RoPE for models WITHOUT QK-Norm (infer_ops::split_rotary_embedding).
  // Reads KLLM_KUNLUN_FUSE_SPLIT_ROPE once; enabled unless explicitly set to "0".
  static bool IsSplitRopeFusionEnabled();
  // Latched off for this layer once the kernel reports failure, so we do not
  // pay the failed-launch cost (and log spam) on every subsequent token.
  bool split_rope_fusion_failed_ = false;
#endif

  bool no_rope_;
  size_t attn_temperature_tuning_;
  size_t floor_scale_;
};

// Dispatch function by kv cache type
#ifdef ENABLE_CUDA
#  define DISPATCH_BY_KVTYPE(dtype, kv_cache_dtype, func, ...)                               \
    switch (kv_cache_dtype) {                                                                \
      case TYPE_FP8_E5M2:                                                                    \
        return func<dtype, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2>(__VA_ARGS__); \
      case TYPE_FP8_E4M3:                                                                    \
        return func<dtype, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3>(__VA_ARGS__); \
      default:                                                                               \
        return func<dtype, dtype, llm_kernels::utils::KVCacheType::kAuto>(__VA_ARGS__);      \
    }
#else
#  define DISPATCH_BY_KVTYPE(dtype, kv_cache_dtype, func, ...) \
    return func<dtype, dtype, llm_kernels::utils::KVCacheType::kAuto>(__VA_ARGS__);
#endif

// Dispatch function by dtype(float16/bfloat16/float) and kv cache dtype(dtype/fp8)
#define DISPATCH_BY_DTYPE_AND_KVTYPE(dtype, kv_cache_dtype, func, ...)                        \
  switch (dtype) {                                                                            \
    case DataType::TYPE_FP16:                                                                 \
      DISPATCH_BY_KVTYPE(float16, kv_cache_dtype, func, __VA_ARGS__);                         \
    case DataType::TYPE_BF16:                                                                 \
      DISPATCH_BY_KVTYPE(bfloat16, kv_cache_dtype, func, __VA_ARGS__);                        \
    case DataType::TYPE_FP32:                                                                 \
      DISPATCH_BY_KVTYPE(float, kv_cache_dtype, func, __VA_ARGS__);                           \
    default:                                                                                  \
      KLLM_THROW(fmt::format("{}: Unsupported Dtype type: {}.", __PRETTY_FUNCTION__, dtype)); \
  }

#ifdef ENABLE_CUDA
// Utility function for initializing YARN rotary embedding (used by sparse MLA indexer layers)
template <typename T>
Status InitYarnRotaryEmbedding(std::optional<llm_kernels::nvidia::RotaryEmbeddingCuda>& rotary_embedding_cuda,
                               const RoPEScalingFactor& rope_scaling_factor_config, void* cos_sin_cache_ptr,
                               float rope_theta, int rope_head_dim, int max_seq_len, int head_dim, int n_heads,
                               bool is_neox, cudaStream_t stream);
#endif
}  // namespace ksana_llm
