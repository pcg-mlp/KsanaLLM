/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "distributed/expert_parallel/expert_parallel_manager.h"
#include "layers/base_layer.h"
#include "models/common_moe/moe_config.h"

namespace ksana_llm {

struct MoeLayerParameters : public BaseLayerParameters {
  // Normalization mode for expert routing weights after top-k selection:
  // NO_NORM keeps raw probabilities; RE_NORM renormalizes selected weights to sum to 1.
  MoeScaleNormMode moe_scale_norm_mode{MoeScaleNormMode::NO_NORM};
  // Max tokens handled in one forward pass; used to size workspace buffers.
  size_t max_token_num{0};
  // Index of this MoE layer within the model; used for hash-gating checks.
  int layer_idx{0};
  // Number of experts local to this node after expert-parallel sharding.
  size_t expert_num_per_node{0};
  // Hidden dim of routed experts; equals model hidden_units, or the latent dim for LatentMoE.
  size_t expert_hidden_size{0};
  // Per-rank intermediate (FFN) size of each expert (moe_inter_size / moe_tensor_para_size).
  size_t expert_inter_size{0};
  // Number of experts selected per token (a.k.a. num_experts_per_tok).
  size_t expert_topk{0};
  // Tensor-parallel size used by the MoE layer.
  int tp_size{0};
  // Whether to use the vLLM-style grouped MoE path (InvokeGroupedTopk) instead of the default kernel.
  bool use_vllm_moe{false};
  // Grouped top-k routing (GLM4-MoE): num_expert_group = number of expert groups (n_group),
  // expert_groups_topk = number of groups selected per token (topk_group).
  uint32_t num_expert_group{1};
  uint32_t expert_groups_topk{1};
  // Router scoring function: "softmax" or "sigmoid".
  std::string scoring_func{"softmax"};
  // Expert selection method: "greedy", "fast", or "noaux_tc".
  std::string topk_method{"greedy"};
  // Whether to normalize the selected top-k routing probabilities to sum to 1.
  bool norm_topk_prob{false};
  // Scaling factor applied to routed expert output (e.g. GLM4-MoE 2.5).
  float routed_scaling_factor{1.0f};
  // Whether to apply e_score_correction_bias in top-k selection (DeepSeek-V3 / GLM4-MoE).
  bool use_e_score_correction_bias{false};
  // Whether shared experts run in full (unsharded) mode across all ranks.
  bool enable_full_shared_expert{false};
  // Quantization dtype of expert weights: fp8_weight_dtype for FP8 (FP8_E4M3 / BLOCK_FP8_E4M3),
  // int_weight_dtype for Int4/FP4 (UINT4x2, INT4x2, I4_GROUP, FP4_E2M1FN_x2); TYPE_INVALID if not quantized.
  DataType fp8_weight_dtype{DataType::TYPE_INVALID};
  DataType int_weight_dtype{DataType::TYPE_INVALID};
  // Quantization group size for int4/fp4 expert weights.
  int group_size{0};
  // Whether to multiply each expert's output by its routing weight; Llama4 uses true, most others false.
  bool apply_weight{false};
  // SwiGLU clamp limit; when >0, gate clamped to [..,limit] and up to [-limit,limit] before SiLU*Mul (DeepSeek-V4).
  float swiglu_limit{0.0f};
  // Hash-based gating (DeepSeek-V4): num_hash_layers leading layers use hash routing
  // (layer_idx < num_hash_layers); vocab_size is the model vocab used for hashing.
  uint32_t num_hash_layers{0};
  uint32_t vocab_size{0};
  // SiTU activation params (Kimi K3): when situ_beta > 0, SiTU replaces SwiGLU.
  float situ_beta{0.0f};
  float situ_linear_beta{0.0f};

  std::string ToString() const {
    return fmt::format(
        "moe_scale_norm_mode:{}, max_token_num:{}, layer_idx:{}, "
        "expert_num_per_node:{}, expert_hidden_size:{}, expert_inter_size:{}, "
        "expert_topk:{}, tp_size:{}, use_vllm_moe:{}, num_expert_group:{}, "
        "expert_groups_topk:{}, scoring_func:{}, topk_method:{}, "
        "norm_topk_prob:{}, routed_scaling_factor:{}, "
        "use_e_score_correction_bias:{}, enable_full_shared_expert:{}, "
        "fp8_weight_dtype:{}, int_weight_dtype:{}, group_size:{}, "
        "apply_weight:{}, swiglu_limit:{}, num_hash_layers:{}, "
        "vocab_size:{}, situ_beta:{}, situ_linear_beta:{}",
        static_cast<int>(moe_scale_norm_mode), max_token_num, layer_idx, expert_num_per_node, expert_hidden_size,
        expert_inter_size, expert_topk, tp_size, use_vllm_moe, num_expert_group, expert_groups_topk, scoring_func,
        topk_method, norm_topk_prob, routed_scaling_factor, use_e_score_correction_bias, enable_full_shared_expert,
        static_cast<int>(fp8_weight_dtype), static_cast<int>(int_weight_dtype), group_size, apply_weight, swiglu_limit,
        num_hash_layers, vocab_size, situ_beta, situ_linear_beta);
  }
};

#ifdef ENABLE_CUDA
// Abstract base class for all MoE layer implementations.
// Provides a unified typed Forward interface so callers can invoke it without
// knowing the concrete subtype.
class BaseMoeLayer : public BaseLayer {
 public:
  // Unified typed Forward for all MoE layer variants.
  // eplb_map, e_score_bias, input_ids may be nullptr when unused (e.g., Marlin path).
  virtual Status Forward(const Tensor* hidden_states, const Tensor* gating_output, const Tensor* w1, const Tensor* w2,
                         const Tensor* eplb_map, Tensor* output, Tensor* workspace,
                         const Tensor* e_score_bias = nullptr, const Tensor* input_ids = nullptr,
                         const LayerExecContext& exec_ctx = {}) = 0;

  // Default no-op. Subclasses with expert parallel support override this.
  virtual void SetExpertParallelManager(std::shared_ptr<ExpertParallelManager> expert_parallel_manager) {}

 protected:
  MoeLayerParameters params_;
};
#endif

}  // namespace ksana_llm
