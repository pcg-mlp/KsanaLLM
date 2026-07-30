/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "layers/grouped_topk_layer.h"

#include <cstdlib>  // std::getenv

#include "configure/environment.h"
#include "csrc/kernels/nvidia/others/sglang/main/route_radix/route_radix.h"
#include "csrc/kernels/nvidia/others/vllm/main/single_group_topk/single_group_topk.h"
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

  const char* env = std::getenv("KSANA_MOE_GATE_BACKEND");
  if (env != nullptr && std::string(env) == "vllm") {
    backend_ = GroupedTopkBackend::kVllm;
  }

  KLLM_LOG_INFO << fmt::format("Rank[{}] GroupedTopkLayer Init with {}", rank_, params_.ToString());
  return Status();
}

template <typename T>
Status GroupedTopkLayer::ForwardT(const Tensor* gating_output, const Tensor* e_bias, Tensor* topk_weights,
                                  Tensor* topk_ids) {
  void* gating_ptr = gating_output->GetPtr<void>();
  void* e_bias_ptr = (params_.use_e_score_correction_bias && e_bias != nullptr) ? e_bias->GetPtr<void>() : nullptr;
  void* topk_weights_ptr = topk_weights->GetPtr<void>();
  void* topk_ids_ptr = topk_ids->GetPtr<void>();

  int num_tokens = static_cast<int>(gating_output->shape[0]);
  int num_experts = static_cast<int>(gating_output->shape[1]);

  if (num_tokens == 0) {
    return Status();
  }

  // TODO(zezhao): replace total_num_experts with num_experts / expert_para_size
  int total_num_experts = num_experts;

  const bool is_sigmoid_with_bias = (params_.scoring_func == "sigmoid" && e_bias_ptr != nullptr && params_.renormalize);

  if (is_sigmoid_with_bias) {
    // Path 1: SGLang radix (default, only 896/16 hits)
    if (backend_ == GroupedTopkBackend::kSglang && num_experts == 896 && params_.topk == 16) {
      llm_kernels::nvidia::sglang_gate::InvokeSglangRouteRadix<T>(
          static_cast<T*>(gating_ptr), static_cast<float*>(e_bias_ptr), static_cast<float*>(topk_weights_ptr),
          static_cast<int32_t*>(topk_ids_ptr), num_tokens, num_experts, params_.topk, params_.renormalize,
          params_.routed_scaling_factor, true, context_->device->GetPrimaryComputeStream().Get());
      return Status();
    }
    // Path 2: vLLM single_group_topk (K3 only: 896 experts). 强力拦截，确保只对 K3 生效，不影响任何已有实现。
    if (num_experts == 896 &&
        llm_kernels::nvidia::InvokeVllmSingleGroupTopk<T>(
            static_cast<T*>(gating_ptr), static_cast<float*>(topk_weights_ptr), static_cast<int32_t*>(topk_ids_ptr),
            static_cast<const float*>(e_bias_ptr), num_tokens, num_experts, params_.topk, params_.renormalize,
            params_.routed_scaling_factor, true, context_->device->GetPrimaryComputeStream().Get())) {
      return Status();
    }
    // Path 3: fall through to existing InvokeGroupedTopk (Legacy)
  }

  InvokeGroupedTopk<T>(gating_ptr, topk_weights_ptr, topk_ids_ptr, num_tokens, total_num_experts, params_.topk,
                       params_.renormalize, params_.num_expert_group, params_.topk_group, params_.scoring_func,
                       e_bias_ptr, params_.routed_scaling_factor, params_.score_threshold, rank_,
                       context_->device->GetPrimaryComputeStream().Get());

  if (is_profile_mode_) {
    FillRandomInts(static_cast<int*>(topk_ids_ptr), num_tokens * params_.topk, 0, total_num_experts, rank_,
                   context_->device->GetPrimaryComputeStream().Get());
  }

  return Status();
}

Status GroupedTopkLayer::Forward(const Tensor* gating_output, Tensor* topk_weights, Tensor* topk_ids,
                                 const Tensor* e_bias, const LayerExecContext& exec_ctx) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, gating_output, e_bias, topk_weights, topk_ids);
}

}  // namespace ksana_llm
