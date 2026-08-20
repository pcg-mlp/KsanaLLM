/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/attention_layer.h"

#include <xpu/runtime.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "device/device_utils.h"
#include "layers/kunlun/host_fp_helpers.h"
#include "utils/logger.h"

namespace {
// Shared neox-format cache: all layers share the same cos_sin_cache, so we allocate once.
//
// Lifetime: entries are keyed by the caller-owned cos_sin_cache_ptr and are intentionally
// held until process exit. The map is bounded by the number of distinct cos_sin_cache
// buffers (one per rank, allocated once at model load), NOT by request or step count, so
// this is a fixed-size cache rather than an unbounded leak. XPU device memory is reclaimed
// by the driver on process teardown. If dynamic model load/unload is ever supported, call
// ReleaseNeoxCosSinCache() from the unload path.
std::mutex g_neox_cache_mu;
std::unordered_map<void*, void*> g_neox_cache_map;  // original_ptr → neox_ptr
}  // namespace

namespace ksana_llm {

bool AttentionLayer::IsSplitRopeFusionEnabled() {
  // Evaluated once per process. Set KLLM_KUNLUN_FUSE_SPLIT_ROPE=0 to fall back
  // to the legacy 3x copy2d + rotary_embedding_neox sequence.
  static const bool enabled = []() {
    const char* env = std::getenv("KLLM_KUNLUN_FUSE_SPLIT_ROPE");
    bool on = !(env != nullptr && std::string(env) == "0");
    KLLM_LOG_INFO << "AttentionLayer: fused split+rope (no QK-Norm) " << (on ? "enabled" : "disabled");
    return on;
  }();
  return enabled;
}

Status AttentionLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                            std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);
  int parameter_index = 0;
  mm_quant_mode_ = std::any_cast<const QuantMode>(parameters[parameter_index++]);
  layernorm_eps_ = std::any_cast<const float>(parameters[parameter_index++]);
  use_qk_norm_ = std::any_cast<const bool>(parameters[parameter_index++]);
  layer_index_ = std::any_cast<const int>(parameters[parameter_index++]);
  layer_num_ = std::any_cast<const int>(parameters[parameter_index++]);
  max_position_embeddings_ = std::any_cast<const int>(parameters[parameter_index++]);
  num_heads_ = std::any_cast<const int>(parameters[parameter_index++]);
  num_kv_heads_ = std::any_cast<const int>(parameters[parameter_index++]);
  head_size_ = std::any_cast<const int>(parameters[parameter_index++]);
  stride_size_ = std::any_cast<const int>(parameters[parameter_index++]);
  tensor_para_size_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  kv_cache_dtype_ = std::any_cast<DataType>(parameters[parameter_index++]);
  k_scale_ = std::any_cast<const float>(parameters[parameter_index++]);
  v_scale_ = std::any_cast<const float>(parameters[parameter_index++]);
  uint32_t rotary_dim = std::any_cast<const int>(parameters[parameter_index++]);
  base_ = std::any_cast<const float>(parameters[parameter_index++]);
  qk_rope_head_dim_ = std::any_cast<const uint32_t>(parameters[parameter_index++]);
  qk_nope_head_dim_ = std::any_cast<const uint32_t>(parameters[parameter_index++]);
  q_lora_rank_ = std::any_cast<const uint32_t>(parameters[parameter_index++]);
  kv_lora_rank_ = std::any_cast<const uint32_t>(parameters[parameter_index++]);
  v_head_dim_ = std::any_cast<const uint32_t>(parameters[parameter_index++]);

  bool is_neox = std::any_cast<const bool>(parameters[parameter_index++]);
  PositionEncoding position_encoding = std::any_cast<const PositionEncoding>(parameters[parameter_index++]);
  void* cos_sin_cache_ptr = std::any_cast<void*>(parameters[parameter_index++]);
  RoPEScalingFactor rope_scaling_factor_config = std::any_cast<const RoPEScalingFactor>(parameters[parameter_index++]);
  // max_batch_size (Ascend ATB 用，Kunlun 不需要但必须消费参数)
  (void)std::any_cast<const size_t>(parameters[parameter_index++]);

  attn_temperature_tuning_ = std::any_cast<const size_t>(parameters[parameter_index++]) > 0;
  attn_scale_ = std::any_cast<const float>(parameters[parameter_index++]);
  floor_scale_ = std::any_cast<const size_t>(parameters[parameter_index++]);

  // is_multi_token_forward (Ascend ATB flag, consume but unused on Kunlun)
  if (parameter_index < static_cast<int>(parameters.size())) {
    (void)std::any_cast<const bool>(parameters[parameter_index++]);
  }

  // mrope/xdrope section ptr (consume but unused on Kunlun)
  if (parameter_index < static_cast<int>(parameters.size())) {
    (void)parameters[parameter_index++];
  }

  // enable_qk_pre_norm_before_rotary_pos (Qwen3, GLM4-MoE: pre-RoPE QK norm)
  if (parameter_index < static_cast<int>(parameters.size())) {
    enable_qk_pre_norm_before_rotary_pos_ = std::any_cast<const bool>(parameters[parameter_index++]);
  }

  no_rope_ = (position_encoding == PositionEncoding::NO_ROPE);

  block_size_ = runtime_config.attn_backend_config.block_size;
  block_token_num_ = runtime_config.attn_backend_config.block_token_num;

  // DeepSeek yarn rope: override rotary_dim to qk_rope_head_dim (matching NVIDIA's AttentionLayer behavior).
  // The cos_sin_cache buffer was allocated with this dimension in forwarding_context.cpp.
  if (rope_scaling_factor_config.type == "yarn" && rope_scaling_factor_config.use_deepseek_rope) {
    rotary_dim = qk_rope_head_dim_;

    // Compute attn_scale_ for MLA attention kernels.
    // Formula: scale = 1/sqrt(qk_nope_head_dim + qk_rope_head_dim) * mscale * mscale
    // This matches NVIDIA's AttentionLayer and xsglang's layer.scaling.
    if (!attn_scale_initialized_) {
      float scaling_factor = rope_scaling_factor_config.factor;
      float mscale_all_dim = rope_scaling_factor_config.mscale_all_dim;
      // deepseek_yarn_get_mscale: if scale <= 1 return 1, else 0.1*mscale*log(scale)+1
      float mscale = (scaling_factor <= 1.0f) ? 1.0f : 0.1f * mscale_all_dim * std::log(scaling_factor) + 1.0f;
      float qk_head_dim_scale = 1.0f / std::sqrt(static_cast<float>(qk_nope_head_dim_ + qk_rope_head_dim_));
      attn_scale_ = qk_head_dim_scale * mscale * mscale;
      attn_scale_initialized_ = true;
    }
  }

  // Determine cos_sin_cache dtype: DeepSeek YARN requires FP32; others use inter_data_type.
  const bool is_yarn = (rope_scaling_factor_config.type == "yarn");
  const bool is_deepseek_yarn = is_yarn && rope_scaling_factor_config.use_deepseek_rope;
  DataType cos_sin_cache_dtype = is_deepseek_yarn ? DataType::TYPE_FP32 : inter_data_type_;

  // QK-norm models (Qwen3, GLM4-MoE) always take the fused split_norm_rope_neox
  // path and never read the standalone cos_sin_cache. DeepSeek YARN is excluded:
  // it skips the neox cache (FP32 vs fp16/bf16) and still needs the standalone
  // FP32 cache for its MLA attention layers.
  const bool use_fused_qk_norm_rope = (use_qk_norm_ || enable_qk_pre_norm_before_rotary_pos_) && !is_deepseek_yarn;

  // Compute and fill cos_sin_cache for RoPE.
  // Layout: [max_pos, rotary_dim] with [cos_half | sin_half] per position.
  //
  // freqs / cache_mscale 提升到块外作用域：下面构建 neox-format cache 时必须复用同一组
  // 频率与 mscale。若在那里另算一遍朴素频率，YARN 等缩放方案就只会作用于 standalone
  // cache，导致走 fused split_norm_rope_neox 的模型静默拿到错误的 RoPE。
  std::vector<float> freqs;
  float cache_mscale = 1.0f;
  if (cos_sin_cache_ptr != nullptr && !no_rope_) {
    int max_pos = max_position_embeddings_;
    int half_dim = rotary_dim / 2;
    float theta = base_;

    freqs.resize(half_dim);

    if (is_yarn) {
      // YARN frequency interpolation (shared by DeepSeek YARN and common YARN).
      // Matches NVIDIA's InvokeYarnComputeCosSinWithCacheKernel.
      float scaling_factor = rope_scaling_factor_config.factor;
      float beta_fast_val = rope_scaling_factor_config.beta_fast;
      float beta_slow_val = rope_scaling_factor_config.beta_slow;
      int orig_max_pos = rope_scaling_factor_config.original_max_position_embeddings;

      float low_dim = (rotary_dim * std::log(static_cast<float>(orig_max_pos) / (beta_fast_val * 2.0f * M_PI))) /
                      (2.0f * std::log(theta));
      float low = std::max(std::floor(low_dim), 0.0f);
      float high_dim = (rotary_dim * std::log(static_cast<float>(orig_max_pos) / (beta_slow_val * 2.0f * M_PI))) /
                       (2.0f * std::log(theta));
      float high = std::min(std::ceil(high_dim), static_cast<float>(rotary_dim) - 1.0f);

      for (int i = 0; i < half_dim; ++i) {
        float pos_freqs = std::pow(theta, static_cast<float>(2 * i) / rotary_dim);
        float inv_freq_extrapolation = 1.0f / pos_freqs;
        float inv_freq_interpolation = 1.0f / (scaling_factor * pos_freqs);
        float ramp_mask = (low == high) ? (static_cast<float>(i) - low) / (high + 0.001f - low)
                                        : (static_cast<float>(i) - low) / (high - low);
        ramp_mask = std::max(0.0f, std::min(1.0f, ramp_mask));
        float inv_freq_mask = (1.0f - ramp_mask) * 1.0f;
        freqs[i] = inv_freq_interpolation * (1.0f - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask;
      }
    } else {
      // Default: simple frequency without scaling.
      for (int i = 0; i < half_dim; ++i) {
        freqs[i] = 1.0f / std::pow(theta, static_cast<float>(2 * i) / rotary_dim);
      }
    }

    // Compute mscale applied to cos/sin values in the cache.
    if (is_yarn) {
      auto yarn_get_mscale = [](float scale, float m) -> float {
        return (scale <= 1.0f) ? 1.0f : 0.1f * m * std::log(scale) + 1.0f;
      };
      float scaling_factor = rope_scaling_factor_config.factor;
      if (is_deepseek_yarn) {
        cache_mscale = yarn_get_mscale(scaling_factor, rope_scaling_factor_config.mscale) /
                       yarn_get_mscale(scaling_factor, rope_scaling_factor_config.mscale_all_dim);
      } else {
        cache_mscale = (scaling_factor <= 1.0f) ? 1.0f : 0.1f * std::log(scaling_factor) + 1.0f;
      }
    }

    // Fill standalone cache only when the fused QK-norm path is NOT active.
    if (!use_fused_qk_norm_rope) {
      // Write cos_sin_cache to host buffer and upload to device.
      if (cos_sin_cache_dtype == DataType::TYPE_FP32) {
        std::vector<float> cache_host(static_cast<size_t>(max_pos) * rotary_dim);
        for (int pos = 0; pos < max_pos; ++pos) {
          for (int i = 0; i < half_dim; ++i) {
            float angle = pos * freqs[i];
            cache_host[pos * rotary_dim + i] = std::cos(angle) * cache_mscale;
            cache_host[pos * rotary_dim + half_dim + i] = std::sin(angle) * cache_mscale;
          }
        }
        size_t cache_bytes = cache_host.size() * sizeof(float);
        MemcpyAsync(cos_sin_cache_ptr, cache_host.data(), cache_bytes, MEMCPY_HOST_TO_DEVICE,
                    context_->device->GetMemoryManageStream());
      } else {
        using kunlun_host::Fp32ToHalf;
        const bool cache_is_bf16 = (inter_data_type_ == DataType::TYPE_BF16);
        std::vector<uint16_t> cache_host(static_cast<size_t>(max_pos) * rotary_dim);
        for (int pos = 0; pos < max_pos; ++pos) {
          for (int i = 0; i < half_dim; ++i) {
            float angle = pos * freqs[i];
            float cos_val = std::cos(angle) * cache_mscale;
            float sin_val = std::sin(angle) * cache_mscale;
            cache_host[pos * rotary_dim + i] =
                cache_is_bf16 ? Fp32ToHalf<bfloat16>(cos_val) : Fp32ToHalf<float16>(cos_val);
            cache_host[pos * rotary_dim + half_dim + i] =
                cache_is_bf16 ? Fp32ToHalf<bfloat16>(sin_val) : Fp32ToHalf<float16>(sin_val);
          }
        }
        size_t cache_bytes = cache_host.size() * sizeof(uint16_t);
        MemcpyAsync(cos_sin_cache_ptr, cache_host.data(), cache_bytes, MEMCPY_HOST_TO_DEVICE,
                    context_->device->GetMemoryManageStream());
      }
      StreamSynchronize(context_->device->GetMemoryManageStream());
      KLLM_LOG_DEBUG << "AttentionLayer: filled cos_sin_cache, rotary_dim=" << rotary_dim << " max_pos=" << max_pos
                     << " theta=" << theta << " dtype="
                     << (cos_sin_cache_dtype == DataType::TYPE_FP32   ? "fp32"
                         : cos_sin_cache_dtype == DataType::TYPE_BF16 ? "bf16"
                                                                      : "fp16")
                     << " mscale=" << cache_mscale << " is_yarn=" << is_yarn;
    }  // if (!use_fused_qk_norm_rope)
  }

  cos_sin_cache_ptr_ = use_fused_qk_norm_rope ? nullptr : cos_sin_cache_ptr;

  // Build neox-format cache for split_norm_rope_neox fused kernel.
  //
  // 数值必须与上面的 standalone cos_sin_cache 完全一致：直接复用同一组 freqs / cache_mscale
  // （已含 YARN 插值与 mscale），不要在这里重算频率，否则两条路径会静默产生不同的 RoPE。
  //
  // DeepSeek YARN 例外：它的 standalone cache 是 FP32，而 split_norm_rope_neox 只实例化了
  // fp16/bf16；且 DeepSeek 走 MLA 路径本就不经过该 fused kernel。这里显式跳过并告警，
  // 而不是构造一个 dtype 不匹配的 cache 让下游 reinterpret_cast 成错误类型。
  // 末项把 "freqs 已按 half_dim 填好" 这个不变量折进条件：正常路径下它与前三项同真，
  // 万一未来重构让两处条件走散，这里会安全退化成不建 neox cache（fused 路径自动关闭），
  // 而不是拿一个空 freqs 越界读出垃圾频率。
  const bool want_neox_cache = cos_sin_cache_ptr != nullptr && !no_rope_ &&
                               (use_qk_norm_ || enable_qk_pre_norm_before_rotary_pos_) &&
                               freqs.size() == static_cast<size_t>(rotary_dim / 2);
  if (want_neox_cache && is_deepseek_yarn) {
    KLLM_LOG_WARNING << "AttentionLayer: skip neox cos_sin cache for DeepSeek YARN (fp32 cache is incompatible with "
                        "the fp16/bf16 split_norm_rope_neox kernel); fused QK-Norm+RoPE path stays disabled";
  }
  if (want_neox_cache && !is_deepseek_yarn) {
    std::lock_guard<std::mutex> lock(g_neox_cache_mu);
    auto it = g_neox_cache_map.find(cos_sin_cache_ptr);
    if (it != g_neox_cache_map.end()) {
      cos_sin_cache_neox_ptr_ = it->second;
    } else {
      int max_pos = max_position_embeddings_;
      int half_dim = rotary_dim / 2;
      size_t neox_elements = static_cast<size_t>(2) * max_pos * rotary_dim;
      size_t neox_bytes = neox_elements * sizeof(uint16_t);
      std::vector<uint16_t> neox_host(neox_elements);

      using kunlun_host::Fp32ToHalf;
      // dtype 必须跟 activation (inter_data_type_) 走，而不是 kv_cache_dtype_：
      // 下游 PagedAttention/FlashAttention 里是 reinterpret_cast<const T*> 且 T = SCALAR_T
      // = inter_data_type_。当前昆仑上两者恒相等，但用错变量在 kv cache 量化后会立刻炸。
      const bool is_bf16 = (inter_data_type_ == DataType::TYPE_BF16);

      uint16_t* cos_plane = neox_host.data();
      uint16_t* sin_plane = neox_host.data() + max_pos * rotary_dim;
      for (int pos = 0; pos < max_pos; ++pos) {
        for (int i = 0; i < half_dim; ++i) {
          float angle = pos * freqs[i];
          float cos_val = std::cos(angle) * cache_mscale;
          float sin_val = std::sin(angle) * cache_mscale;
          uint16_t cos_bits = is_bf16 ? Fp32ToHalf<bfloat16>(cos_val) : Fp32ToHalf<float16>(cos_val);
          uint16_t sin_bits = is_bf16 ? Fp32ToHalf<bfloat16>(sin_val) : Fp32ToHalf<float16>(sin_val);
          cos_plane[pos * rotary_dim + i] = cos_bits;
          cos_plane[pos * rotary_dim + half_dim + i] = cos_bits;
          sin_plane[pos * rotary_dim + i] = sin_bits;
          sin_plane[pos * rotary_dim + half_dim + i] = sin_bits;
        }
      }

      void* neox_dev = nullptr;
      if (xpu_malloc(&neox_dev, neox_bytes) != 0 || neox_dev == nullptr) {
        KLLM_LOG_ERROR << "AttentionLayer: failed to allocate neox cache (" << neox_bytes << " bytes)";
      } else {
        MemcpyAsync(neox_dev, neox_host.data(), neox_bytes, MEMCPY_HOST_TO_DEVICE,
                    context_->device->GetMemoryManageStream());
        StreamSynchronize(context_->device->GetMemoryManageStream());
        g_neox_cache_map[cos_sin_cache_ptr] = neox_dev;
        KLLM_LOG_DEBUG << "AttentionLayer: allocated neox cache, " << neox_bytes << " bytes";
      }
      cos_sin_cache_neox_ptr_ = neox_dev;
    }
  }

  (void)is_neox;

  return Status();
}

void ReleaseNeoxCosSinCache() {
  std::lock_guard<std::mutex> lock(g_neox_cache_mu);
  for (auto& kv : g_neox_cache_map) {
    if (kv.second != nullptr) {
      xpu_free(kv.second);
    }
  }
  g_neox_cache_map.clear();
  KLLM_LOG_DEBUG << "AttentionLayer: released all neox cos_sin caches";
}

}  // namespace ksana_llm
