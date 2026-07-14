/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "layers/grouped_topk_layer.h"

#include "configure/environment.h"
#include "kernels/nvidia/moe_kernel_wrapper.h"
#include "utils/singleton.h"

namespace ksana_llm {

Status GroupedTopkLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                              std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);
  params_ = std::any_cast<const GroupedTopkLayerParameters>(parameters.front());

  expert_para_size_ = runtime_config.parallel_basic_config.expert_parallel_size;
  expert_world_size_ = runtime_config.parallel_basic_config.expert_world_size;

  is_profile_mode_ = runtime_config.is_profile_mode;

  KLLM_LOG_INFO << fmt::format("Rank[{}] GroupedTopkLayer Init with {}", rank_, params_.ToString());
  return Status();
}

Status GroupedTopkLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, input_tensors, output_tensors);
}

template <typename T>
Status GroupedTopkLayer::ForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  // input_tensors:
  // [0] gating_output [num_tokens, num_experts]
  // [1] e_bias (optional, can be nullptr)
  //
  // output_tensors:
  // [0] topk_weights_ptr [num_tokens, topk]
  // [1] topk_ids_ptr [num_tokens, topk]

  void* gating_output = input_tensors[0].GetPtr<void>();
  void* e_bias = nullptr;

  // 根据 use_e_score_correction_bias 和输入张量数量决定是否使用 e_bias
  if (params_.use_e_score_correction_bias && input_tensors.size() > 1) {
    e_bias = input_tensors[1].GetPtr<void>();
  }

  void* topk_weights_ptr = output_tensors[0].GetPtr<void>();
  void* topk_ids_ptr = output_tensors[1].GetPtr<void>();

  int num_tokens = input_tensors[0].shape[0];
  int num_experts = input_tensors[0].shape[1];

  if (num_tokens == 0) {
    return Status();
  }

  // 计算 total_num_experts，考虑专家并行
  // TODO(zezhao): 使用 num_experts / expert_para_size 来替换 total_num_experts. 不再维护 ExpertParallelSize
  int total_num_experts = num_experts;

  InvokeGroupedTopk<T>(gating_output, topk_weights_ptr, topk_ids_ptr, num_tokens, total_num_experts, params_.topk,
                       params_.renormalize, params_.num_expert_group, params_.topk_group, params_.scoring_func, e_bias,
                       params_.routed_scaling_factor, params_.score_threshold, rank_,
                       context_->device->GetPrimaryComputeStream().Get());

  if (is_profile_mode_) {  // profile 模式下，填充固定的 topk_ids
    FillRandomInts(static_cast<int*>(topk_ids_ptr), num_tokens * params_.topk, 0, total_num_experts, rank_,
                   context_->device->GetPrimaryComputeStream().Get());
  }

  return Status();
}

}  // namespace ksana_llm
