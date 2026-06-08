/* Copyright 2026 Tencent Inc.  All rights reserved.
==============================================================================*/
#pragma once

#include <cstdint>

#ifdef ENABLE_CUDA
#  include "csrc/kernels/nvidia/moe/expert_map/expert_map.h"
#  include "csrc/utils/nvidia/workspace.h"
#endif
#include "distributed/expert_parallel/expert_parallel_manager.h"
#include "layers/base_moe_layer.h"
#include "layers/grouped_topk_layer.h"
#include "layers/topk_softplus_sqrt_layer.h"
#include "models/base/base_weight.h"
#include "models/common_moe/moe_config.h"

namespace ksana_llm {

#ifdef ENABLE_CUDA
class MarlinFp4MoeLayer : public BaseMoeLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;

  virtual void SetExpertParallelManager(std::shared_ptr<ExpertParallelManager> expert_parallel_manager) override;

  virtual size_t GetWorkspaceSize() override;

  virtual Status Preprocess(const ModelConfig& model_config, const RuntimeConfig& runtime_config) override;

  // eplb_map/e_score_bias/input_ids are nullptr when unused.
  virtual Status Forward(const Tensor* hidden_states, const Tensor* gating_output, const Tensor* w1, const Tensor* w2,
                         const Tensor* eplb_map, Tensor* output, Tensor* workspace,
                         const Tensor* e_score_bias = nullptr, const Tensor* input_ids = nullptr,
                         const LayerExecContext& exec_ctx = {}) override;

 private:
  template <typename T>
  Status InitT(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
               std::shared_ptr<Context> context, int rank);

  template <typename T>
  size_t GetWorkspaceSizeT();

  template <typename T>
  Status ForwardT(const Tensor* hidden_states, const Tensor* gating_output, const Tensor* w1, const Tensor* w2,
                  const Tensor* eplb_map, Tensor* output, Tensor* workspace, const Tensor* e_score_bias,
                  const Tensor* input_ids);

  template <typename T>
  Status ExecuteGroupedTopk(const Tensor* hidden_states, const Tensor* gating_output, const Tensor* eplb_map,
                            Tensor* workspace, const Tensor* e_score_bias);

  template <typename T>
  Status ExecuteTopkSoftplusSqrt(const Tensor* hidden_states, const Tensor* gating_output, const Tensor* eplb_map,
                                 Tensor* workspace, const Tensor* e_score_bias, const Tensor* input_ids);

  Status Dispatch(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);
  Status Combine(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);

  Status DumpEplbData(Tensor& topk_ids);

 protected:
  bool set_workspace_buffer_info_ = true;

  MoeScaleNormMode moe_scale_norm_mode_;
  size_t max_ws_bytes_ = 0;
  size_t max_token_num_;
  size_t expert_num_per_node_;
  size_t expert_hidden_size_;
  size_t expert_inter_size_;
  size_t expert_topk_;
  int tp_size_;
  int layer_idx_;
  bool use_vllm_moe_ = false;
  uint32_t num_expert_group_ = 1;
  uint32_t expert_groups_topk_ = 1;
  std::string scoring_func_ = "softmax";
  std::string topk_method_ = "greedy";
  bool norm_topk_prob_ = false;
  float routed_scaling_factor_ = 1.0f;
  bool use_e_score_correction_bias_ = false;
  bool enable_full_shared_expert_ = false;
  bool apply_weight_ = false;
  bool has_swiglu_limit_ = false;
  float swiglu_limit_ = 0.0f;

  void* topk_weights_ptr_ = nullptr;
  size_t topk_weights_ptr_size = 0;
  void* topk_ids_ptr_ = nullptr;
  size_t topk_ids_ptr_size = 0;
  void* fused_id_buffer_ = nullptr;
  size_t max_fused_id_buffer_size = 0;
  void* intermediate_cache1_ = nullptr;
  size_t intermediate_cache1_size = 0;
  void* intermediate_cache2_ = nullptr;
  size_t intermediate_cache2_size = 0;
  void* intermediate_cache3_ = nullptr;
  size_t intermediate_cache3_size = 0;
  int* marlin_workspace_ = nullptr;
  size_t marlin_workspace_size = 0;
  float* c_tmp_ = nullptr;
  size_t c_tmp_size = 0;

  size_t global_expert_para_size_;
  bool using_deepep_ = false;

  std::shared_ptr<GroupedTopkLayer> grouped_topk_layer_;
  bool use_topk_softplus_sqrt_ = false;
  bool use_hash_ = false;
  uint32_t num_hash_layers_ = 0;
  uint32_t vocab_size_ = 0;
  size_t softplus_sqrt_workspace_size_ = 0;
  std::shared_ptr<TopKSoftplusSqrtLayer> topk_softplus_sqrt_layer_;

  bool enable_load_eplb_config_ = false;
  bool enable_dump_eplb_data_ = false;
  int eplb_dump_step_ = 0;
  std::string eplb_dump_path_;
  std::shared_ptr<llm_kernels::nvidia::moe::ExpertMap> expert_map_;

  std::shared_ptr<ExpertParallelManager> expert_parallel_manager_;
};
#endif
}  // namespace ksana_llm
