/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include "configure/schedule_config_parser.h"
#include "models/base/base_model_weight_loader.h"
#include "models/base/common_model_weight_loader.h"
#include "models/weight_method/weight_method.h"

namespace ksana_llm {

// Weight loader for HunYuan V3 (hy_v3). BF16 and FP8-E4M3 static per-tensor.
//
// hy_v3 = Qwen3-MoE attention (GQA + pre-RoPE QK-Norm, no QKV bias) + DeepSeek-V3
// style MoE gating. Supports:
//   - BF16 checkpoints (weights only, no scales)
//   - FP8-E4M3 static per-tensor (weight_scale + input_scale per-projection scalar)
//   - MTP head at layer index num_layer (num_nextn_predict_layers=1 typical)
class HyV3WeightLoader : public BaseModelWeightLoader {
 public:
  HyV3WeightLoader(std::shared_ptr<BaseModelConfig> model_config, std::shared_ptr<Environment> env,
                   std::shared_ptr<Context> context);

  virtual Status FilterWeightNames(std::vector<std::string>& weight_names) override;

  virtual Status PreProcessModelWeights(const std::unordered_map<std::string, Tensor>& host_model_weights) override;

  virtual Status ProcessModelWeights(const std::unordered_map<std::string, Tensor>& host_model_weights, int dev_rank,
                                     std::unordered_map<std::string, Tensor>& device_model_weights,
                                     std::unordered_map<std::string, Tensor>& left_host_weights) override;

  virtual Status PostProcessModelWeights(std::unordered_map<std::string, Tensor>& dev_weights_map,
                                         int dev_rank) override;

 private:
  // Merge per-expert gate/up and down projections into batched [num_experts, ...] tensors.
  // Dispatches to Fp8E4m3Method::process_experts_proj under FP8, else BF16 path.
  Status process_experts_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                              const std::string& weight_prefix_name, int dev_rank);

  // Post-process one decoder-shaped layer (main-model layers and MTP nextn layers alike):
  // fuse QKV, permute o_proj, fuse gate_up / permute down for MoE (routed experts, shared expert)
  // or dense MLP prefix.
  Status PostProcessDecoderLayer(std::unordered_map<std::string, Tensor>& dev_weights_map, int layer_idx, int dev_rank,
                                 int first_k_dense_replace);

  bool IsFp8() const;

  PipelineConfig pipeline_config_;
  RuntimeConfig runtime_config_;
  std::shared_ptr<CommonModelWeightLoader> common_weight_loader_;
  std::shared_ptr<WeightMethod> weight_method_;
  size_t tp_;
};

}  // namespace ksana_llm
