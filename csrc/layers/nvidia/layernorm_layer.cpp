/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/layernorm_layer.h"

#include "3rdparty/LLM_kernels/csrc/kernels/nvidia/layernorm/rmsnorm_fused_parallel.h"
#include "3rdparty/LLM_kernels/csrc/kernels/nvidia/others/sglang/main/deepseek_v4/rmsnorm.h"
#include "kernels/nvidia/kernel_wrapper.h"

namespace ksana_llm {

Status LayernormLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                            std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);
  int parameter_index = 0;
  rms_norm_eps_ = std::any_cast<const float>(parameters[parameter_index++]);
  KLLM_LOG_DEBUG << fmt::format("rms_norm_eps {}", rms_norm_eps_);

  // Optional: num_heads and head_dim for per-head no-weight RMSNorm
  if (parameter_index < static_cast<int>(parameters.size())) {
    num_heads_ = std::any_cast<const uint32_t>(parameters[parameter_index++]);
    head_dim_ = std::any_cast<const uint32_t>(parameters[parameter_index++]);
  }

  return Status();
}

template <typename T>
Status LayernormLayer::ForwardNoWeightT(const Tensor& input, Tensor& output) {
  KLLM_CHECK_WITH_INFO(num_heads_ > 0 && head_dim_ > 0,
                       "num_heads and head_dim must be set for no-weight per-head RMSNorm");

  const uint32_t batch_size = static_cast<uint32_t>(input.shape[0]);
  const int64_t stride_batch_bytes = static_cast<int64_t>(num_heads_ * head_dim_ * sizeof(T));
  const int64_t stride_head_bytes = static_cast<int64_t>(head_dim_ * sizeof(T));

  cudaStream_t stream = context_->device->GetPrimaryComputeStream().Get();

  if (head_dim_ == 512) {
    llm_kernels::nvidia::rmsnorm_self<T, 512>(input.GetPtr<void>(), output.GetPtr<void>(), stride_batch_bytes,
                                              stride_head_bytes, batch_size, num_heads_, rms_norm_eps_, stream);
  } else {
    KLLM_THROW(fmt::format("Unsupported head_dim {} for no-weight per-head RMSNorm", head_dim_));
  }

  output.shape = input.shape;
  output.dtype = input.dtype;
  return Status();
}

template <typename T>
Status LayernormLayer::ForwardWeightT(const Tensor& input, const Tensor& weight, Tensor& output, const Tensor* bias) {
  if (bias != nullptr) {
    InvokeLayerNorm<T>(input.GetPtr<void>(), weight.GetPtr<void>(), bias->GetPtr<void>(), rms_norm_eps_, input.shape[0],
                       input.shape[1], output.GetPtr<void>(), context_->device->GetPrimaryComputeStream().Get());
  } else {
    InvokeRMSNorm<T>(input.GetPtr<void>(), weight.GetPtr<void>(), rms_norm_eps_, input.shape[0], input.shape[1],
                     output.GetPtr<void>(), context_->device->GetPrimaryComputeStream().Get());
  }
  output.shape = input.shape;
  output.dtype = input.dtype;
  return Status();
}

template <typename T>
Status LayernormLayer::ForwardParallelT(const Tensor& in1, const Tensor& w1, const Tensor& in2, const Tensor& w2,
                                        Tensor& out1, Tensor& out2, int stride1, int stride2) {
  const int token_num = in1.shape[0];
  const int hidden_size1 = in1.shape[1];
  const int hidden_size2 = in2.shape[1];
  const int s1 = (stride1 > 0) ? stride1 : hidden_size1;
  const int s2 = (stride2 > 0) ? stride2 : hidden_size2;
  llm_kernels::nvidia::InvokeParallelRmsNorm<T>(in1.GetPtr<const T>(), w1.GetPtr<const T>(), out1.GetPtr<T>(),
                                                hidden_size1, s1, in2.GetPtr<const T>(), w2.GetPtr<const T>(),
                                                out2.GetPtr<T>(), hidden_size2, s2, token_num, rms_norm_eps_,
                                                context_->device->GetPrimaryComputeStream().Get());
  out1.shape = {static_cast<size_t>(token_num), static_cast<size_t>(hidden_size1)};
  out1.dtype = in1.dtype;
  out2.shape = {static_cast<size_t>(token_num), static_cast<size_t>(hidden_size2)};
  out2.dtype = in2.dtype;
  return Status();
}

Status LayernormLayer::Forward(const Tensor* input, Tensor* output, const LayerExecContext& exec_ctx) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardNoWeightT, *input, *output);
}

Status LayernormLayer::Forward(const Tensor* input, const Tensor* weight, Tensor* output, const Tensor* bias,
                               const LayerExecContext& exec_ctx) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardWeightT, *input, *weight, *output, bias);
}

Status LayernormLayer::Forward(const Tensor* input1, const Tensor* weight1, const Tensor* input2, const Tensor* weight2,
                               Tensor* output1, Tensor* output2, int stride1, int stride2,
                               const LayerExecContext& exec_ctx) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardParallelT, *input1, *weight1, *input2, *weight2, *output1, *output2,
                      stride1, stride2);
}

}  // namespace ksana_llm
