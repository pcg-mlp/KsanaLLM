/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include "configure/model_config.h"
#include "device/device_types.h"
#include "models/base/base_model_config.h"

namespace ksana_llm {

// Model config for GLM4-MoE architecture.
// This config inherits directly from BaseModelConfig (like Qwen2Moe)
// instead of DeepSeekV3Config, as GLM4-MoE uses standard GQA attention
// rather than MLA (Multi-head Latent Attention).
struct Glm4MoeConfig : public BaseModelConfig {
  virtual ~Glm4MoeConfig() {}

  // Check if a QuantMode is W4A8/GPTQ int4 quantization
  static bool IsW4A8Method(QuantMode method) {
    return method == QUANT_W4A8_AWQ || method == QUANT_GPTQ;
  }

  // Check if a weight matches W4A8/GPTQ int4 quantization pattern (for W4A8 mixed quantization)
  bool IsWeightMatchW4A8(const std::string& weight_name) const {
    if (IsW4A8Method(quant_config.method)) {
      return true;
    }
    for (const auto& sub_quant_config : sub_quant_configs) {
      if (IsW4A8Method(sub_quant_config.method)) {
        for (const auto& pattern_layer : sub_quant_config.pattern_layers) {
          if (weight_name.find(pattern_layer) != std::string::npos) {
            // Check if it's in ignored layers
            bool ignored = false;
            for (const auto& ignored_layer : sub_quant_config.ignored_layers) {
              if (weight_name.find(ignored_layer) != std::string::npos) {
                ignored = true;
                break;
              }
            }
            if (!ignored) return true;
          }
        }
      }
    }
    return false;
  }

  // Check if a weight contains W4A8-specific suffixes
  bool IsW4A8Contain(const std::string& weight_name) const {
    std::vector<std::string> w4a8_weights_suffix = {"qweight", "scales", "qzeros"};
    QuantConfig w4a8_config = GetW4A8QuantConfig();
    if (w4a8_config.input_scale) {
      w4a8_weights_suffix.push_back("input_scale");
      w4a8_weights_suffix.push_back("weight_scale");
    }
    for (const auto& suffix : w4a8_weights_suffix) {
      if (weight_name.find(suffix) != std::string::npos) {
        return true;
      }
    }
    return false;
  }

  // Get the W4A8/GPTQ quant config (from sub_quant_configs if mixed quantization)
  QuantConfig GetW4A8QuantConfig() const {
    if (IsW4A8Method(quant_config.method)) {
      return quant_config;
    }
    for (const auto& sub_quant_config : sub_quant_configs) {
      if (IsW4A8Method(sub_quant_config.method)) {
        return sub_quant_config;
      }
    }
    return QuantConfig();  // Return empty config if not found
  }

  // Type of weight
  DataType weight_data_type;

  std::string tokenizer_path;

  // The max number of (input + output tokens)
  size_t max_token_num;

  size_t max_step_token_num;

  size_t tensor_para_size;

  size_t head_num;
  uint32_t size_per_head;
  uint32_t inter_size;
  uint32_t hidden_units;
  uint32_t num_layer;
  uint32_t rotary_embedding;
  float rope_theta;
  float layernorm_eps;
  uint32_t vocab_size;
  uint32_t start_id;
  uint32_t end_id;
  uint32_t pad_id;
  size_t num_key_value_heads;
  int max_batch_size;
  int max_position_embeddings;
  size_t block_token_num;

  // Multiple end token IDs support (GLM4-MoE may have multiple eos tokens)
  std::vector<uint32_t> end_ids;

  RoPEScalingFactor rope_scaling_factor_config;

  bool tie_word_embeddings;
  bool exist_tie_embeddings_param = true;

  // The activation function used.
  std::string activation_function{"swiglu"};

  // Determines if the model is a visual llm model.
  bool is_visual = false;

  // Determines if the model is a quant model.
  bool is_quant = false;
  QuantConfig quant_config;
  std::vector<QuantConfig> sub_quant_configs;

  // GLM4-MoE is always a MoE model with shared experts, these are architectural constants.
  // Unlike generic configs, these are fixed by the model architecture definition and
  // should NOT be overridden by config.json.
  bool is_moe = true;
  bool has_shared_experts = true;
  MoeConfig moe_config;

  // Shared expert configuration
  uint32_t shared_expert_intermediate_size;

  // GLM4-MoE specific features — hardcoded per architecture spec:
  // - load_bias: GLM4-MoE always has bias on QKV projections (config.json: add_qkv_bias=true)
  // - use_qk_norm: QK Norm is always enabled (config.json: use_qk_norm=true)
  // - enable_add_qkv_bias: QKV bias is always present in this architecture
  // NOTE: use_qk_norm here is for Glm4MoeConfig's internal use.
  // model_instance.cpp sets ModelConfig.use_qk_norm=false because KsanaLLM's attention layer
  // uses enable_qk_pre_norm_before_rotary_pos instead (different code path, same effect).
  bool load_bias = true;
  bool use_qk_norm = true;
  bool enable_add_qkv_bias = true;

  // Partial rotary factor (GLM4-MoE uses partial rotary embedding)
  float partial_rotary_factor = 0.5f;

  // MTP (Multi-Token Prediction) support
  uint32_t num_nextn_predict_layers = 0;

  // others attributes
  std::unordered_map<std::string, std::string> model_attributes;

  // Helper method to check if model contains W4A8/GPTQ int4 weights
  bool ContainW4A8Weights() const {
    if (IsW4A8Method(quant_config.method)) {
      return true;
    }
    for (const auto& sub_config : sub_quant_configs) {
      if (IsW4A8Method(sub_config.method)) {
        return true;
      }
    }
    return false;
  }
};

}  // namespace ksana_llm
