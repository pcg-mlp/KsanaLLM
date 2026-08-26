/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include "configure/model_config.h"
#include "device/device_types.h"
#include "models/base/base_model_config.h"

namespace ksana_llm {

// Model config for HunYuan V3 (hy_v3 / HYV3ForCausalLM).
//
// hy_v3 is architecturally a superset of Qwen3-MoE's attention (standard GQA
// with pre-RoPE QK-Norm, no QKV bias) combined with DeepSeek-V3 style MoE gating
// (sigmoid scoring + aux-loss-free expert bias + a single shared expert + a
// first_k_dense prefix). It uses standard GQA attention, NOT MLA, so this config
// inherits directly from BaseModelConfig (like Qwen3MoeConfig / Glm4MoeConfig).
struct HyV3Config : public BaseModelConfig {
  virtual ~HyV3Config() {}

  // Weight/activation dtype (bfloat16 for hy_v3).
  DataType weight_data_type;

  // Core dimensions.
  size_t head_num;
  size_t num_key_value_heads;
  uint32_t size_per_head;
  uint32_t inter_size;  // dense MLP intermediate size (first_k_dense layers)
  uint32_t hidden_units;
  uint32_t num_layer;
  uint32_t rotary_embedding;
  float rope_theta;
  float layernorm_eps;
  uint32_t vocab_size;
  uint32_t start_id;
  uint32_t end_id;
  std::vector<uint32_t> end_ids;
  uint32_t pad_id;
  int max_position_embeddings;

  bool tie_word_embeddings;
  bool exist_tie_embeddings_param = true;

  // Quantization: unset for BF16 checkpoints, FP8-E4M3 static per-tensor otherwise.
  bool is_quant = false;
  QuantConfig quant_config;

  // hy_v3 is always a MoE model with a single shared expert.
  bool is_moe = true;
  bool has_shared_experts = true;
  MoeConfig moe_config;

  // Shared expert intermediate size = num_shared_experts * expert_hidden_dim.
  uint32_t shared_expert_intermediate_size;

  // hy_v3 applies pre-RoPE QK-Norm (q_norm/k_norm) and has NO bias on QKV.
  // NOTE: use_qk_norm here is for HyV3Config's internal use, and reflects config.json's
  // `qk_norm`. model_instance.cpp sets ModelConfig.use_qk_norm=false because KsanaLLM's
  // attention layer uses enable_qk_pre_norm_before_rotary_pos instead (different code
  // path, same effect); ModelConfig.use_qk_norm means post-RoPE QK-Norm.
  bool use_qk_norm = true;
  bool enable_add_qkv_bias = false;

  // MTP (Multi-Token Prediction) head count; 0 disables MTP.
  uint32_t num_nextn_predict_layers = 0;
};

}  // namespace ksana_llm
