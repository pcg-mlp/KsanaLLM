/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include "configure/schedule_config_parser.h"
#include "models/base/base_model_weight_loader.h"
#include "models/base/common_model_weight_loader.h"
#include "models/weight_method/weight_method.h"

namespace ksana_llm {

// GLM4-MoE weight loader that inherits directly from BaseModelWeightLoader
// (like Qwen2Moe) instead of DeepSeekV3WeightLoader.
// This loader handles:
// 1. Standard GQA attention (not MLA)
// 2. QKV bias and QK norm
// 3. MoE expert weights with gate_up fusion
// 4. Shared expert weights
// 5. MTP (Multi-Token Prediction) layer weights: eh_proj, enorm, hnorm, decoder
class Glm4MoeWeightLoader : public BaseModelWeightLoader {
 public:
  Glm4MoeWeightLoader(std::shared_ptr<BaseModelConfig> model_config, std::shared_ptr<Environment> env,
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

 private:
  // Process QKV bias fusion
  Status process_attn_qkv_bias(std::unordered_map<std::string, Tensor>& device_model_weights,
                               const std::string& weight_prefix_name, int dev_rank);

  PipelineConfig pipeline_config_;
  RuntimeConfig runtime_config_;
  ExpertParallelConfig expert_parallel_config_;
  std::shared_ptr<CommonModelWeightLoader> common_weight_loader_;
  std::shared_ptr<WeightMethod> weight_method_;
  std::shared_ptr<WeightMethod> expert_weight_method_;         // For routed experts (TP=1, no inter_size split)
  std::shared_ptr<WeightMethod> shared_expert_weight_method_;  // For enable_full_shared_expert mode

  size_t tp_;
  bool enable_full_shared_expert_ = false;

  void GetLocalExpertRange(int dev_rank, size_t num_experts, size_t& expert_start, size_t& expert_end) const;
  bool IsLocalExpert(int dev_rank, size_t num_experts, int expert_idx) const;

  // Processes a single decoder layer's weights in PostProcess:
  // QKV fusion + bias, O_proj, MoE (gate/experts/shared_expert) or Dense MLP.
  // Used by both main decoder loop and MTP decoder loop to eliminate duplication.
  Status PostProcessDecoderLayer(std::unordered_map<std::string, Tensor>& dev_weights_map, int layer_idx, int dev_rank,
                                 int first_k_dense_replace);
};

}  // namespace ksana_llm
