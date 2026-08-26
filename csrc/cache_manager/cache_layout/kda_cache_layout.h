/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "cache_manager/cache_layout/base_cache_layout.h"
#include "configure/model_config.h"
#include "configure/schedule_config_parser.h"
#include "device/device_types.h"
#include "utils/logger.h"
#include "utils/memory_utils.h"

namespace ksana_llm {

// Per-layer KDA state layout within one block:
//   [conv_state | recurrent_state]
// Only the two segment offsets are recorded (K/V share each offset).
// Non-KDA layers (full-attn hybrid layers) get zero-offset placeholders.
class KDALayerCacheLayout : public BaseLayerCacheLayout {
 public:
  KDALayerCacheLayout() = default;
  virtual ~KDALayerCacheLayout() = default;

  size_t conv_offset = 0;
  size_t recurrent_offset = 0;
};

// The block-level cache layout for a KDA state.
// A block holds the states of all layers.
class KDAStateLayout : public BaseBlockCacheLayout {
 public:
  KDAStateLayout() = default;
  KDAStateLayout(const ModelConfig& model_config, const PipelineConfig& pipeline_config,
                 const RuntimeConfig& runtime_config, DataType recurrent_state_dtype = TYPE_FP32)
      : recurrent_state_dtype_(recurrent_state_dtype) {
    Initialize(model_config, pipeline_config, runtime_config, recurrent_state_dtype);
  }
  virtual ~KDAStateLayout() = default;

  virtual size_t GetBlockCacheSize() override { return block_cache_size_; }

  // Returns per-layer conv/recurrent byte offsets (K/V share each offset).
  // Non-KDA (full-attn hybrid) layers return offset 0 with cache_size 0.
  virtual KvCacheLayerOffsets GetKvCacheLayerOffsets(int32_t layer_index, const std::string& kv_cache_type) override {
    if (layer_index < 0 || layer_index >= static_cast<int>(layer_cache_layouts_.size())) {
      throw std::runtime_error(fmt::format("KDAStateLayout: The layer_index {} exceeds layerlayout_size {}.",
                                           layer_index, layer_cache_layouts_.size()));
    }
    const auto& layer = layer_cache_layouts_[layer_index];
    if (kv_cache_type == "conv") {
      return {layer->conv_offset, layer->conv_offset};
    }
    if (kv_cache_type == "recurrent") {
      return {layer->recurrent_offset, layer->recurrent_offset};
    }
    throw std::runtime_error(
        fmt::format("KDAStateLayout: kv_cache_type '{}' not supported (expected conv / recurrent).", kv_cache_type));
  }

  virtual std::vector<std::string> GetActiveKvCacheTypes() override { return {"conv", "recurrent"}; }

  void Initialize(const ModelConfig& model_config, const PipelineConfig& pipeline_config,
                  const RuntimeConfig& runtime_config, DataType recurrent_state_dtype) {
    recurrent_state_dtype_ = recurrent_state_dtype;
    // MTP part.
    const bool predict_nextn =
        static_cast<int>(pipeline_config.lower_nextn_layer_idx) >= static_cast<int>(model_config.num_layer);
    const size_t node_nextn_layer_num =
        predict_nextn ? pipeline_config.upper_nextn_layer_idx - pipeline_config.lower_nextn_layer_idx + 1 : 0;
    // All layers including MTP.
    const size_t node_layer_num =
        pipeline_config.upper_layer_idx - pipeline_config.lower_layer_idx + 1 + node_nextn_layer_num;

    const size_t attn_tp = runtime_config.parallel_basic_config.attn_tensor_parallel_size;
    const uint32_t head_dim = model_config.kimi_linear_config.head_dim;
    const size_t num_heads_local = DivRoundUp(model_config.head_num, attn_tp);
    const size_t conv_dim_local = 3 * num_heads_local * head_dim;  // (q|k|v packed) per TP rank
    // SD layout: (state_len, conv_dim) — dim contiguous.
    const uint32_t conv_kernel = model_config.kimi_linear_config.short_conv_kernel_size;
    const size_t conv_state_len = (conv_kernel > 0) ? static_cast<size_t>(conv_kernel) - 1 : 0;

    // conv_state is always bfloat16; recurrent_state defaults to fp32 (overridable).
    const size_t conv_size = conv_state_len * conv_dim_local * GetTypeSize(TYPE_BF16);
    const size_t recurrent_size = num_heads_local * head_dim * head_dim * GetTypeSize(recurrent_state_dtype_);

    size_t layer_base_offset = 0;
    for (size_t layer_index = pipeline_config.lower_layer_idx;
         layer_index < (pipeline_config.lower_layer_idx + node_layer_num); ++layer_index) {
      auto layer_cache_layout = std::make_shared<KDALayerCacheLayout>();
      layer_cache_layout->base_offset = layer_base_offset;
      layer_cache_layout->layer_index = static_cast<int32_t>(layer_index);

      // Only KDA layers own conv+recurrent state; full-attn hybrid layers are
      // inactive in this pool (guarded by IsKdaLayer at access sites).
      if (model_config.kimi_linear_config.IsKdaLayer(static_cast<int>(layer_index))) {
        layer_cache_layout->conv_offset = layer_base_offset;
        layer_cache_layout->recurrent_offset = layer_base_offset + conv_size;
        layer_cache_layout->cache_size = conv_size + recurrent_size;
        layer_base_offset += layer_cache_layout->cache_size;
      } else {
        // Inactive layer: zero-offset placeholder; never read (IsKdaLayer guard).
        layer_cache_layout->conv_offset = 0;
        layer_cache_layout->recurrent_offset = 0;
        layer_cache_layout->cache_size = 0;
      }

      KLLM_LOG_DEBUG << fmt::format("KDA layer {} state layout: conv_offset={} recurrent_offset={} cache_size={}",
                                    layer_index, layer_cache_layout->conv_offset, layer_cache_layout->recurrent_offset,
                                    layer_cache_layout->cache_size);
      layer_cache_layouts_.push_back(layer_cache_layout);
    }
    block_cache_size_ = layer_base_offset;
  }

  // Overload required by BaseBlockCacheLayout; delegates with the recurrent
  // dtype stored on this instance (defaults to TYPE_FP32)
  void Initialize(const ModelConfig& model_config, const PipelineConfig& pipeline_config,
                  const RuntimeConfig& runtime_config) override {
    Initialize(model_config, pipeline_config, runtime_config, recurrent_state_dtype_);
  }

 private:
  size_t block_cache_size_ = 0;
  DataType recurrent_state_dtype_ = TYPE_FP32;
  std::vector<std::shared_ptr<KDALayerCacheLayout>> layer_cache_layouts_;
};

}  // namespace ksana_llm
