/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/attn_res_layer.h"

#include "kernels/nvidia/triton_wrapper.h"
#include "utils/logger.h"

namespace ksana_llm {

Status AttnResLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                          std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);
  int parameter_index = 0;
  const float rms_norm_eps = std::any_cast<const float>(parameters[parameter_index++]);
  const float output_rms_norm_eps = std::any_cast<const float>(parameters[parameter_index++]);
  hidden_dim_ = std::any_cast<const uint32_t>(parameters[parameter_index++]);
  constexpr float kFixedEps = 1e-6f;
  KLLM_CHECK_WITH_INFO(rms_norm_eps == kFixedEps && output_rms_norm_eps == kFixedEps,
                       "AttnResLayer requires rms_norm_eps == output_rms_norm_eps == 1e-6 (baked into cubin)");
  return Status();
}

template <typename T>
Status AttnResLayer::ForwardT(const Tensor& prefix, const Tensor* delta, const Tensor& blocks,
                              const Tensor& norm_weight, const Tensor& qk_weight, const Tensor* output_norm_weight,
                              Tensor& output, bool write_block) {
  KLLM_CHECK(prefix.shape.size() == 2);
  KLLM_CHECK(blocks.shape.size() == 3);
  KLLM_CHECK(prefix.shape[1] == hidden_dim_);
  KLLM_CHECK(blocks.shape[0] == prefix.shape[0]);
  KLLM_CHECK(blocks.shape[2] == hidden_dim_);
  if (delta != nullptr) {
    KLLM_CHECK(delta->shape.size() == 2);
    KLLM_CHECK(delta->shape[0] == prefix.shape[0]);
    KLLM_CHECK(delta->shape[1] == hidden_dim_);
  }

  const int num_tokens = static_cast<int>(prefix.shape[0]);
  const int num_blocks = static_cast<int>(blocks.shape[1]);
  void* delta_ptr = (delta != nullptr) ? delta->GetPtr<void>() : nullptr;
  void* output_norm_weight_ptr = (output_norm_weight != nullptr) ? output_norm_weight->GetPtr<void>() : nullptr;

  Singleton<TritonWrapper>::GetInstance()->InvokeAttnResKernel<T>(
      prefix.GetPtr<void>(), delta_ptr, blocks.GetPtr<void>(), norm_weight.GetPtr<void>(), qk_weight.GetPtr<void>(),
      output_norm_weight_ptr, output.GetPtr<void>(), num_tokens, static_cast<int>(hidden_dim_), num_blocks, write_block,
      context_->device->GetPrimaryComputeStream().Get());

  output.shape = {prefix.shape[0], hidden_dim_};
  output.dtype = prefix.dtype;
  return Status();
}

Status AttnResLayer::Forward(const Tensor* prefix, const Tensor* delta, const Tensor* blocks, const Tensor* norm_weight,
                             const Tensor* qk_weight, const Tensor* output_norm_weight, Tensor* output,
                             bool write_block, const LayerExecContext& exec_ctx) {
  KLLM_CHECK_WITH_INFO(inter_data_type_ == DataType::TYPE_BF16, "AttnResLayer only supports BF16.");
  return ForwardT<bfloat16>(*prefix, delta, *blocks, *norm_weight, *qk_weight, output_norm_weight, *output,
                            write_block);
}

}  // namespace ksana_llm
