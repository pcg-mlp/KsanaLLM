/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <unordered_set>

#include "models/base/base_model_weight_loader.h"
#include "models/deepseek_v3/deepseek_v3_config.h"
#include "models/deepseek_v3/deepseek_v3_weight_impl.h"

namespace ksana_llm {
// The deepseek loader.
class DeepSeekV3WeightLoader : public BaseModelWeightLoader {
 public:
  DeepSeekV3WeightLoader(std::shared_ptr<BaseModelConfig> model_config, std::shared_ptr<Environment> env,
                         std::shared_ptr<Context> context);
  virtual ~DeepSeekV3WeightLoader() override;

  // Do some filter on model weight names.
  virtual Status FilterWeightNames(std::vector<std::string>& weight_names) override;

  // Process weights, such as rename, split, merge, type convert, quantization, etc.
  virtual Status ProcessModelWeights(const std::unordered_map<std::string, Tensor>& host_model_weights, int dev_rank,
                                     std::unordered_map<std::string, Tensor>& device_model_weights,
                                     std::unordered_map<std::string, Tensor>& left_host_weights) override;

  // Invoked only once after ProcessModelWeights.
  virtual Status PostProcessModelWeights(std::unordered_map<std::string, Tensor>& dev_weights_map,
                                         int dev_rank) override;

 protected:
  Status InitWeightLoaderImpl(DataType weight_data_type);

  // Load EPLB mapping configuration from JSON file
  Status InitExpertMap(const size_t& num_experts, const size_t& expert_start_id, const size_t& expert_end_id,
                       int dev_rank, std::vector<std::vector<int>>& expert_map,
                       std::unordered_map<std::string, Tensor>& device_model_weights);

  // MLA kv_b_proj weight loading: splits kv_b_proj into kv_b_nope_proj + v_head_proj,
  // creates w_uk_t / w_uv / kv_b_combined_proj derived tensors.
  // Handles both non-fp8 (split + permute) and fp8-blockwise (deferred split) paths.
  void LoadMlaKvBProj(const std::string& file_weight_name, const Tensor& host_weight_tensor, int dev_rank,
                      std::shared_ptr<DeepSeekV3Config>& deepseek_v3_config,
                      std::unordered_map<std::string, Tensor>& device_model_weights, size_t head_num,
                      size_t qk_nope_head_dim, size_t qk_rope_head_dim, size_t v_head_dim, size_t kv_lora_rank,
                      size_t attn_tp_size, size_t attn_dev_rank, size_t head_num_tp);

#ifdef ENABLE_KUNLUN
  // W4A8_KLX: Load non-expert weight_scale tensors directly to device.
  // Handles kv_b_proj.weight_scale host-side head-interleaved split and bf16/fp16→fp32 cast.
  void LoadW4A8KlxWeightScale(const std::string& file_weight_name, const Tensor& host_weight_tensor,
                              std::unordered_map<std::string, Tensor>& device_model_weights,
                              std::shared_ptr<DeepSeekV3Config>& deepseek_v3_config, size_t attn_dev_rank,
                              size_t head_num_tp, size_t qk_nope_head_dim, size_t v_head_dim);
#endif

 protected:
  PipelineConfig pipeline_config_;
  RuntimeConfig runtime_config_;
  std::unique_ptr<DeepSeekV3WeightImplBase> weight_impl_;
  // rank -> weight_name
  std::vector<std::unordered_set<std::string>> weights_to_permute_;

  // When true, fc_fusion weights are stored in [N, K] layout (trans_b=True) to match xsglang;
  // fc_batch (w_uk_t/w_uv) weights use [B, N, K] layout.
  // When false (CUDA default), fc_fusion weights are [K, N] and fc_batch uses [B, K, N].
  bool fc_use_transb_layout_ = false;

#ifdef ENABLE_KUNLUN
  // W4A8_KLX: Create weight_scale tensors for fused/renamed weights (fused_lora_a, q_b_nope_rope,
  // kv_b_nope, v_head, gate_up) and return the set of weights already in [k,n] layout.
  std::unordered_set<std::string> CreateW4A8KlxScales(std::unordered_map<std::string, Tensor>& dev_weights_map,
                                                      int dev_rank,
                                                      std::shared_ptr<DeepSeekV3Config>& deepseek_v3_config);
#endif
};
}  // namespace ksana_llm
