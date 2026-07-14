/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include "models/base/base_model_weight_loader.h"
#include "models/base/common_model_weight_loader.h"
#include "models/weight_method/weight_method.h"

namespace ksana_llm {
// hunyuan-turbo weight loader
class HunyuanTurboWeightLoader : public BaseModelWeightLoader {
 public:
  HunyuanTurboWeightLoader(std::shared_ptr<BaseModelConfig> model_config, std::shared_ptr<Environment> env,
                           std::shared_ptr<Context> context);
  virtual ~HunyuanTurboWeightLoader() override;

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
  Status load_mlp_fused_gate_up_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                     const std::string& weight_name, const Tensor& weight_tensor, int dev_rank);

  Status load_attn_fused_qkv_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                  const std::string& weight_name, const Tensor& weight_tensor, int dev_rank);

  Status process_mlp_fused_gate_up_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                        const std::string& weight_prefix_name, int dev_rank);

  Status process_attn_fused_qkv_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                     const std::string& weight_prefix_name, int dev_rank);

  void CheckWeightStoreType(bool& is_initialized, bool& is_fused, bool is_fused_expected);

 private:
  PipelineConfig pipeline_config_;
  RuntimeConfig runtime_config_;
  std::shared_ptr<CommonModelWeightLoader> common_weight_loader_;
  std::shared_ptr<WeightMethod> weight_method_;

  size_t tp_;

  bool is_attn_qkv_store_type_initialized_ = false;
  bool is_mlp_gate_up_store_type_initialized_ = false;
  bool is_fused_attn_qkv_proj_;
  bool is_fused_mlp_gate_up_proj_;
};

}  // namespace ksana_llm
