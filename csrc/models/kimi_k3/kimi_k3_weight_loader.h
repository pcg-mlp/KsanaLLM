/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "models/deepseek_v3/deepseek_v3_weight_loader.h"
#include "models/kimi_k3/kimi_k3_config.h"

namespace ksana_llm {

// The Kimi-K3 weight loader.
//
// Kimi-K3's text backbone (KimiLinear) is highly homogeneous with DeepSeek-V3:
// MLA attention + MoE (with an extra Latent-MoE projection) + shared experts.
// It therefore inherits DeepSeekV3WeightLoader to reuse the shared infrastructure
// (weight_impl_, pipeline/runtime config, InitExpertMap, weights_to_permute_) but
// implements ProcessModelWeights / PostProcessModelWeights independently to handle
// Kimi-K3 specific structures:
//   - KDA (Kimi Delta Attention) linear-attention layers: q/k/v/g/b/f_a/f_b_proj,
//     q/k/v_conv1d, A_log, dt_bias, o_norm.
//   - Latent MoE projections: routed_expert_up_proj / routed_expert_down_proj /
//     routed_expert_norm (fully replicated, not quantized).
//   - MXFP4 (compressed-tensors) quantized routed experts (weight + weight_scale).
//   - Attention-residual projections (self_attention_res_proj / mlp_res_proj,
//     output dim == 1, fully replicated).
//
// NOTE: the multimodal vision tower (vision_tower.* / mm_projector.*) is filtered
// out for now and handled in a later multimodal stage.
class KimiK3WeightLoader : public DeepSeekV3WeightLoader {
 public:
  KimiK3WeightLoader(std::shared_ptr<BaseModelConfig> model_config, std::shared_ptr<Environment> env,
                     std::shared_ptr<Context> context);
  virtual ~KimiK3WeightLoader() override;

  // Do some filter on model weight names (skip vision/mm_projector/rotary + distributed layers).
  virtual Status FilterWeightNames(std::vector<std::string>& weight_names) override;

  // Process weights independently (does NOT call V3 parent's ProcessModelWeights).
  virtual Status ProcessModelWeights(const std::unordered_map<std::string, Tensor>& host_model_weights, int dev_rank,
                                     std::unordered_map<std::string, Tensor>& device_model_weights,
                                     std::unordered_map<std::string, Tensor>& left_host_weights) override;

  // Post-process weights (bind quant scale / permute / dtype cast).
  virtual Status PostProcessModelWeights(std::unordered_map<std::string, Tensor>& dev_weights_map,
                                         int dev_rank) override;

 private:
  // ---- MLA attention (full-attention layers) ----
  // q_a_proj / kv_a_proj_with_mqa are replicated and fused into fused_lora_a_proj (reuse V3 semantics).
  // q_b_proj -> q_b_nope_rope_proj (column split + permute); kv_b_proj -> kv_b_nope/v_head/w_uk_t/w_uv;
  // o_proj -> row split; g_proj (MLA output gate) -> column split.
  // Returns true if handled.
  bool ProcessMlaWeight(const std::string& file_weight_name, const Tensor& host_weight_tensor, int dev_rank,
                        std::unordered_map<std::string, Tensor>& device_model_weights);

  // ---- KDA linear-attention layers ----
  // q/k/v/g_proj fused into fused_qkvg_proj (4 segments, column split by attn_tp).
  // q/k/v_conv1d fused into qkv_conv1d (3 segments, column split, dtype forced fp32).
  // b_proj/f_b_proj: column split individually.
  // Replicated: f_a_proj, o_norm.
  // Row-split: o_proj.
  // Special: A_log (narrow to local heads), dt_bias (dim0 split), forced fp32.
  // Returns true if handled.
  bool ProcessKdaWeight(const std::string& file_weight_name, const Tensor& host_weight_tensor, int dev_rank,
                        std::unordered_map<std::string, Tensor>& device_model_weights);

  // ---- Latent MoE + router (fully replicated) ----
  // routed_expert_up_proj / routed_expert_down_proj / routed_expert_norm / gate.weight /
  // gate.e_score_correction_bias -> copied to each device (gate.weight transposed).
  // Returns true if handled.
  bool ProcessLatentMoeAndRouter(const std::string& file_weight_name, const Tensor& host_weight_tensor, int dev_rank,
                                 std::unordered_map<std::string, Tensor>& device_model_weights);

  // Copy a host tensor to device without any split (fully replicated), optionally transpose.
  void CopyReplicated(const std::string& weight_name, const Tensor& host_weight_tensor, int dev_rank,
                      std::unordered_map<std::string, Tensor>& device_model_weights, bool transpose = false);

  // Column split along dim0 following attn_tp, then optional permute (transpose). For KDA q/k/v/g/b/f_b_proj.
  void ColumnSplitAttn(const std::string& weight_name, const Tensor& host_weight_tensor, int dev_rank,
                       std::unordered_map<std::string, Tensor>& device_model_weights);

  // Cached Kimi-K3 config pointer, initialized once in constructor.
  std::shared_ptr<KimiK3Config> kimi_k3_config_;
};

}  // namespace ksana_llm
