/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "layers/base_moe_layer.h"
#include "models/common_moe/moe_config.h"

namespace ksana_llm {

// Kunlun XPU implementation of MoE (Mixture of Experts) layer.
// Supports:
//   - BF16 experts (standard xblas::moe_fc_fusion)
//   - W4A8 INT4 quantized experts (xblas::moe_fc_fusion with int4_wo_int8)
//   - Softmax scoring (Qwen, DeepSeek base)
//   - Sigmoid scoring with e_score_correction_bias (Kimi-K2.6)
// Implements the BaseMoeLayer typed Forward interface using Kunlun-native kernels
// (moe_softmax_topk_norm, moe_ffn_pre_sorted, moe_fc_fusion (xblas), swiglu,
// moe_ffn_post_fusion), matching NVIDIA's MoeLayer.
class KunlunMoeLayer : public BaseMoeLayer {
 public:
  Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
              std::shared_ptr<Context> context, int rank) override;

  size_t GetWorkspaceSize() override;

  Status SetWorkspaceBuffer(const std::shared_ptr<Tensor>& workspace_buffer) override;

  Status Forward(const Tensor* hidden_states, const Tensor* gating_output, const Tensor* w1, const Tensor* w2,
                 const Tensor* eplb_map, Tensor* output, Tensor* workspace, const Tensor* e_score_bias = nullptr,
                 const Tensor* input_ids = nullptr, const LayerExecContext& exec_ctx = {}) override;

 private:
  template <typename T>
  Status ForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);

 protected:
  // MoE configuration
  size_t max_token_num_;
  int layer_idx_;
  size_t expert_num_per_node_;
  size_t expert_hidden_size_;  // hidden_units
  size_t expert_inter_size_;   // moe_intermediate_size
  size_t expert_topk_;
  bool norm_topk_prob_;
  std::string scoring_func_;
  float routed_scaling_factor_;
  uint32_t num_expert_group_ = 1;
  uint32_t topk_group_ = 1;

  // Quantization control
  bool use_e_score_correction_bias_ = false;
  bool is_w4a8_quantized_ = false;  // Expert weights are INT4x2 (int4 packed in pairs)

  // Expert parallelism
  size_t ep_size_ = 1;
  size_t ep_rank_ = 0;
  size_t expert_num_total_ = 0;  // = expert_num_per_node_ * ep_size_

  // Workspace layout (all offsets relative to workspace_buffer_ base)
  size_t workspace_size_;
};

}  // namespace ksana_llm
