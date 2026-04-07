/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "models/arc_hunyuan_video/arc_hunyuan_video_config.h"
#include "models/base/base_model_weight_loader.h"
#include "models/base/common_model_weight_loader.h"
#include "models/weight_method/weight_method.h"

namespace ksana_llm {
// ArcHunyuanVideo weight loader
class ArcHunyuanVideoWeightLoader : public BaseModelWeightLoader {
 public:
  ArcHunyuanVideoWeightLoader(std::shared_ptr<BaseModelConfig> model_config, std::shared_ptr<Environment> env,
                              std::shared_ptr<Context> context);
  virtual ~ArcHunyuanVideoWeightLoader() override;

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
  // WeightMethod callbacks; do not call directly.

  // Load QKV with head_groups reordering.
  Status load_attn_qkv_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                            const std::string& weight_name, const Tensor& weight_tensor, int dev_rank);

  // Load gate_up_proj with swap.
  Status load_mlp_gate_up_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                               const std::string& weight_name, const Tensor& weight_tensor, int dev_rank);

  // Process QKV (Permute2D, no AutoMerge).
  Status process_attn_qkv_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                               const std::string& weight_prefix_name, int dev_rank);

  // Process gate_up_proj (Permute2D, no AutoMerge).
  Status process_mlp_gate_up_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                  const std::string& weight_prefix_name, int dev_rank);

  // FP8 process QKV (Permute2D + dynamic quant).
  Status process_fp8e4m3_attn_qkv_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                       const std::string& weight_prefix_name, int dev_rank);

  // FP8 process gate_up_proj (Permute2D + dynamic quant).
  Status process_fp8e4m3_mlp_gate_up_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                          const std::string& weight_prefix_name, int dev_rank);

 private:
  PipelineConfig pipeline_config_;
  RuntimeConfig runtime_config_;
  std::shared_ptr<ArcHunyuanVideoConfig> arc_hunyuan_video_config_;
  std::shared_ptr<CommonModelWeightLoader> common_weight_loader_;
  std::shared_ptr<WeightMethod> weight_method_;

  size_t tp_;

  // Whether the model has a separate lm_head weight (V1.1) or uses tied embed_tokens (V2).
  bool has_separate_lm_head_ = false;
};

}  // namespace ksana_llm
