/* Copyright 2024 Tencent Inc.  All rights reserved.
 * ==============================================================================*/
#pragma once
#include "csrc/utils/nvidia/workspace.h"
#include "layers/base_moe_layer.h"
#include "models/base/base_weight.h"
#include "models/common_moe/moe_config.h"

namespace ksana_llm {

#ifdef ENABLE_CUDA
class MarlinMoeLayer : public BaseMoeLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;

  virtual size_t GetWorkspaceSize() override;

  virtual Status Preprocess(const ModelConfig& model_config_, const RuntimeConfig& runtime_config) override;

  // BaseMoeLayer typed Forward override; eplb_map/workspace/e_score_bias/input_ids unused in Marlin path.
  virtual Status Forward(const Tensor* hidden_states, const Tensor* gating_output, const Tensor* w1, const Tensor* w2,
                         const Tensor* eplb_map, Tensor* output, Tensor* workspace,
                         const Tensor* e_score_bias = nullptr, const Tensor* input_ids = nullptr,
                         const LayerExecContext& exec_ctx = {}) override;

 private:
  template <typename T>
  size_t GetWorkspaceSizeT();

 private:
  MoeScaleNormMode moe_scale_norm_mode_;
  size_t max_token_num_;
  size_t expert_num_;
  size_t expert_hidden_size_;
  size_t expert_inter_size_;
  size_t expert_topk_;
  int tp_size_;
  bool use_lora_ = false;
  bool use_vllm_moe_ = false;
  uint32_t num_expert_group_ = 1;
  uint32_t expert_groups_topk_ = 1;
  std::string scoring_func_ = "softmax";
  std::string topk_method_ = "greedy";
  bool norm_topk_prob_ = false;
  float routed_scaling_factor_ = 1.0f;
  bool use_e_score_correction_bias_ = false;
  bool enable_full_shared_expert_ = false;
  size_t group_size_ = 128;
  bool apply_weight_ = false;
  size_t num_bits_ = 4;

  size_t max_ws_bytes_;
  size_t max_gating_size_;
};
#endif
}  // namespace ksana_llm
