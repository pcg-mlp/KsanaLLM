/* Copyright 2024 Tencent Inc.  All rights reserved.
==============================================================================*/
#pragma once
#include "csrc/utils/nvidia/workspace.h"
#ifdef ENABLE_CUDA
#  include "csrc/kernels/nvidia/cutlass_extensions/gemm_configs.h"
#  include "csrc/kernels/nvidia/moe/expert_map/expert_map.h"
#endif
#include "distributed/expert_parallel/expert_parallel_manager.h"
#include "layers/base_layer.h"
#include "layers/grouped_topk_layer.h"
#include "layers/topk_softplus_sqrt_layer.h"
#include "models/base/base_weight.h"
#include "models/common_moe/moe_config.h"

namespace ksana_llm {

struct WorkspaceInfo {
  void* workspace{};
  void* scale_probs{};
  void* fc2_output{};
  void* src_to_dest_map{};
  void* selected_experts{};
  void* lora_workspace{};
  size_t size{};
  std::vector<size_t> workspace_sizes;
};
#ifdef ENABLE_CUDA
class MoeLayer : public BaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;

  // Set expert parallel manager (should be called after Init)
  void SetExpertParallelManager(std::shared_ptr<ExpertParallelManager> expert_parallel_manager);

  virtual size_t GetWorkspaceSize() override;

  virtual Status SetWorkspaceBuffer(const std::shared_ptr<Tensor>& workspace_buffer) override;

  virtual Status Preprocess(const ModelConfig& model_config_, const RuntimeConfig& runtime_config) override;

  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors,
                         const LayerExecContext& exec_ctx = {}) override;

  // Test-only observability: true iff Init picked the sqrt+softplus gate.
  bool IsUsingTopkSoftplusSqrt() const { return use_topk_softplus_sqrt_; }

 private:
  // 执行 GroupedTopk 计算与 Dispatch 分发
  template <typename T>
  Status ExecuteGroupedTopk(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);

  // Softplus+sqrt gating used by DeepSeek V4 (parallel to ExecuteGroupedTopk).
  template <typename T>
  Status ExecuteTopkSoftplusSqrt(const std::vector<Tensor>& input_tensors,
                                 std::vector<Tensor>& output_tensors);

  // Used for distributing and gathering topk_ids and hidden_buffer in Expert-Parallel scenarios
  Status Dispatch(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);
  Status Combine(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);

  Status DumpEplbData(Tensor& topk_ids);

  template <typename T>
  Status InitT(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
               std::shared_ptr<Context> context, int rank);

  template <typename T>
  size_t GetWorkspaceSizeT();

  template <typename T>
  Status PreprocessT(const ModelConfig& model_config_, const RuntimeConfig& runtime_config);

  template <typename T>
  Status ForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);

 protected:
  bool set_workspace_buffer_info_ = true;
  bool run_topk_ = true;

  MoeScaleNormMode moe_scale_norm_mode_;
  size_t max_ws_bytes_;
  size_t max_token_num_;
  size_t expert_num_per_node_;
  size_t expert_hidden_size_;
  size_t expert_inter_size_;
  size_t expert_topk_;
  size_t global_expert_para_size_;  // expert_para_size* expert_world_size
  int tp_size_;
  int layer_idx_;
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
  float swiglu_limit_ = 0.0f;

  size_t scale_probabilities_size_;
  size_t src_to_dest_map_size_;
  size_t selected_expert_size_;
  size_t lora_workspace_size_;
  size_t moe_workspace_size_;

  bool apply_weight_ = false;

  W4AFP8_MOE_BACKEND w4afp8_moe_backend_;

  void* topk_weights_ptr_;
  size_t topk_weights_ptr_size;
  void* topk_ids_ptr_;
  size_t topk_ids_ptr_size;
  void* intermediate_cache1_;
  size_t intermediate_cache1_size;
  void* intermediate_cache2_;
  size_t intermediate_cache2_size;
  void* intermediate_cache3_;
  size_t intermediate_cache3_size;

  // buffer_of_ids_in_kernel
  void* fused_id_buffer_;
  size_t max_fused_id_buffer_size;

  // shared space size
  size_t intermediate_cache1_and_cache3_size;

  DataType weight_dtype_;
  DataType compute_dtype_;
  std::vector<int> block_shape_;
  void* a1_q_;
  void* a2_q_;
  void* a1_scale_;
  void* a2_scale_;
  size_t a1_q_size;
  size_t a2_q_size;
  size_t a1_scale_size;
  size_t a2_scale_size;
  size_t a1_and_a2_q_size;
  size_t a1_and_a2_scale_size;

  size_t dequant_workspace_size;
  void* dequant_workspace_;

  // Low Latency mode specific buffers
  void* ll_output_ = nullptr;
  size_t ll_output_size = 0;
  void* expert_num_tokens_ = nullptr;
  size_t expert_num_tokens_size = 0;

  // InvokeFusedBatchedMoe intermediate buffers (for LL mode)
  void* batched_intermediate_output_ = nullptr;  // [num_experts, max_num_tokens, n*2]
  size_t batched_intermediate_output_size = 0;
  void* batched_silu_output_ = nullptr;  // [num_experts, max_num_tokens, n]
  size_t batched_silu_output_size = 0;

  // The vector of the best config index for every tokens number
  std::vector<size_t> config_map_;
  WorkspaceInfo workspace_info_;

  // GroupedTopk layer for handling topk computation
  std::shared_ptr<GroupedTopkLayer> grouped_topk_layer_;

  // Set from scoring_func_ == "sqrtsoftplus". When true, topk_softplus_sqrt_layer_
  // is used for gating instead of grouped_topk_layer_.
  bool use_topk_softplus_sqrt_ = false;

  // Derived at Init: layer_idx < num_hash_layers_. Only meaningful when
  // use_topk_softplus_sqrt_ is true.
  bool use_hash_ = false;

  // DeepSeek V4 hash routing: first num_hash_layers_ MoE layers use a
  // per-token-id expert-index lookup table (tid2eid) instead of score-based top-k.
  uint32_t num_hash_layers_ = 0;
  uint32_t vocab_size_ = 0;

  // Size of the inner gate's workspace (only set when use_topk_softplus_sqrt_).
  size_t softplus_sqrt_workspace_size_ = 0;

  // Alternate gate used when use_topk_softplus_sqrt_ is true.
  std::shared_ptr<TopKSoftplusSqrtLayer> topk_softplus_sqrt_layer_;

  std::vector<llm_kernels::nvidia::cutlass_extensions::CutlassGemmConfig> tactics_;

  bool using_deepep_;
  bool low_latency_mode_ = false;
  size_t low_latency_max_tokens_ = 0;

  // Used for Expert-Parallel load balancer.
  bool enable_load_eplb_config_ = false;
  bool enable_dump_eplb_data_ = false;
  int eplb_dump_step_ = 0;
  std::string eplb_dump_path_;

  std::shared_ptr<llm_kernels::nvidia::moe::ExpertMap> expert_map_;

  std::shared_ptr<ExpertParallelManager> expert_parallel_manager_;
};
#endif
}  // namespace ksana_llm
