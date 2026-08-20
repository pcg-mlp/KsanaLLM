/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include "models/base/base_model_weight_loader.h"
#include "models/base/common_model_weight_loader.h"
#include "models/weight_method/weight_method.h"

namespace ksana_llm {
// Qwen2Moe weight loader
class NewQwen2MoeWeightLoader : public BaseModelWeightLoader {
 public:
  NewQwen2MoeWeightLoader(std::shared_ptr<BaseModelConfig> model_config, std::shared_ptr<Environment> env,
                          std::shared_ptr<Context> context);

  // Do some filter on model weight names.
  virtual Status FilterWeightNames(std::vector<std::string>& weight_names) override;

  // Invoked only once before ProcessModelWeights.
  virtual Status PreProcessModelWeights(const std::unordered_map<std::string, Tensor>& host_model_weights) override;

  // Process weights, such as rename, split, merge, type convert, quantization, etc.
  virtual Status ProcessModelWeights(const std::unordered_map<std::string, Tensor>& host_model_weights, int dev_rank,
                                     std::unordered_map<std::string, Tensor>& device_model_weights,
                                     std::unordered_map<std::string, Tensor>& left_host_weights) override;

  // Invoked only once after ProcessModelWeights.
  virtual Status PostProcessModelWeights(std::unordered_map<std::string, Tensor>& dev_weights_map,
                                         int dev_rank) override;

 public:
  Status process_experts_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                              const std::string& weight_prefix_name, int dev_rank);

  Status process_attn_qkv_bias(std::unordered_map<std::string, Tensor>& device_model_weights,
                               const std::string& weight_prefix_name, int dev_rank);

  // EP-aware per-expert loaders for the BF16 path. Under expert parallelism each rank keeps only its
  // local routed experts and splits the intermediate dim by moe_tensor_para_size (not tensor_parallel_size).
  // Non-routed weights (shared expert / dense MLP) fall back to the standard full-TP split, so behavior
  // is unchanged when expert_para_size==1. These override CommonMethod::load_mlp_gate_up_proj /
  // load_mlp_down_proj for qwen2_moe experts.
  Status load_mlp_gate_up_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                               const std::string& weight_name, const Tensor& weight_tensor, int dev_rank);
  Status load_mlp_down_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                            const std::string& weight_name, const Tensor& weight_tensor, int dev_rank);

 private:
  // Compute this rank's expert-parallel partition (mirrors csrc/models/quant/quant_weight.cpp).
  // Returns the local expert range [expert_start, expert_end), the moe tensor-parallel group size and
  // this rank's index within that group.
  void ComputeExpertPartition(int dev_rank, size_t num_experts, size_t& num_experts_per_rank, size_t& expert_start,
                              size_t& expert_end, size_t& moe_tp_size, size_t& tp_rank) const;

  PipelineConfig pipeline_config_;
  RuntimeConfig runtime_config_;
  std::shared_ptr<CommonModelWeightLoader> common_weight_loader_;
  std::shared_ptr<WeightMethod> weight_method_;

  size_t tp_;
};

}  // namespace ksana_llm