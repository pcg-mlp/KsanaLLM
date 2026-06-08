/* Copyright 2024 Tencent Inc.  All rights reserved.
==============================================================================*/
#pragma once

#ifdef ENABLE_CUDA
#  include "csrc/kernels/nvidia/moe/cutlass_moe/cutlass_moe_wrapper.h"
#  include "csrc/kernels/nvidia/moe/expert_map/expert_map.h"
#  include "csrc/utils/nvidia/workspace.h"
#endif
#include "distributed/expert_parallel/expert_parallel_manager.h"
#include "layers/base_moe_layer.h"
#include "layers/grouped_topk_layer.h"
#include "models/base/base_weight.h"
#include "models/common_moe/moe_config.h"

namespace ksana_llm {

#ifdef ENABLE_CUDA
class CutlassMoeLayer : public BaseMoeLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;

  // Set expert parallel manager (should be called after Init)
  virtual void SetExpertParallelManager(std::shared_ptr<ExpertParallelManager> expert_parallel_manager) override;

  virtual size_t GetWorkspaceSize() override;

  virtual Status Preprocess(const ModelConfig& model_config, const RuntimeConfig& runtime_config) override;

  // eplb_map/e_score_bias are nullptr when unused.
  virtual Status Forward(const Tensor* hidden_states, const Tensor* gating_output, const Tensor* w1, const Tensor* w2,
                         const Tensor* eplb_map, Tensor* output, Tensor* workspace,
                         const Tensor* e_score_bias = nullptr, const Tensor* input_ids = nullptr,
                         const LayerExecContext& exec_ctx = {}) override;

 private:
  inline Status ProcessChunks(const Tensor& input_tensor, Tensor& output_tensor, const size_t total_tokens,
                              const llm_kernels::nvidia::tensorrt_llm::dev::Tensor& fc1_expert_weights_ktensor,
                              const llm_kernels::nvidia::tensorrt_llm::dev::Tensor& fc2_expert_weights_ktensor,
                              const std::vector<llm_kernels::nvidia::tensorrt_llm::dev::Tensor>& quant_scales_ktensors);

  Status ExecuteGroupedTopk(const Tensor* hidden_states, const Tensor* gating_output, const Tensor* eplb_map,
                            Tensor* output, Tensor* workspace, const Tensor* e_score_bias);

  // Used for distributing and gathering topk_ids and hidden_buffer in Expert-Parallel scenarios
  Status Dispatch(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);
  Status Combine(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);

  // Used for Expert-Parallel load balancer.
  Status DumpEplbData(Tensor& topk_ids);

  template <typename T>
  Status InitT(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
               std::shared_ptr<Context> context, int rank);

 private:
  inline std::vector<int64_t> GetBestTactic(const size_t& num_rows);

 protected:
  bool set_workspace_buffer_info_ = true;

  MoeScaleNormMode moe_scale_norm_mode_;
  size_t max_ws_bytes_;
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
  int group_size_;
  bool apply_weight_ = false;

  void* topk_weights_ptr_;
  size_t topk_weights_ptr_size;
  void* topk_ids_ptr_;
  size_t topk_ids_ptr_size;
  void* kernel_workspace_ptr_;
  size_t kernel_workspace_size;

  // LowLatency mode specific buffers
  bool use_low_latency_mode_ = false;
  size_t low_latency_max_tokens_ = 0;
  void* adapted_topk_weights_ptr_ = nullptr;
  void* adapted_topk_ids_ptr_ = nullptr;
  void* recv_expert_count_ptr_ = nullptr;
  size_t adapted_topk_weights_size_ = 0;
  size_t adapted_topk_ids_size_ = 0;
  size_t recv_expert_count_size_ = 0;

  // Adapter function to transform LowLatency dispatch output to FusedMoe kernel input format
  void AdaptLowLatencyDispatchOutput(const int* recv_expert_count, int num_experts_per_node, int tokens_per_expert,
                                     Stream& stream);

  size_t global_expert_para_size_;
  size_t global_expert_para_rank_;

  bool using_deepep_;

  // Used for Expert-Parallel load balancer.
  bool enable_load_eplb_config_ = false;
  bool enable_dump_eplb_data_ = false;
  int eplb_dump_step_ = 0;
  std::string eplb_dump_path_;

  std::shared_ptr<llm_kernels::nvidia::moe::ExpertMap> expert_map_;

  std::vector<std::vector<int64_t>> config_map_;

  std::shared_ptr<GroupedTopkLayer> grouped_topk_layer_;

  std::shared_ptr<llm_kernels::nvidia::CutlassMoeWrapper> cutlass_moe_wrapper_;

  std::shared_ptr<ExpertParallelManager> expert_parallel_manager_;

  // Swiglu-limit contract fields (strict isolation — own members, not from MoeLayer).
  float swiglu_limit_ = 0.0f;
  std::vector<float> swiglu_limit_host_;
  size_t swiglu_limit_ptr_size_ = 0;
  void* swiglu_limit_ptr_ = nullptr;
};
#endif
}  // namespace ksana_llm
