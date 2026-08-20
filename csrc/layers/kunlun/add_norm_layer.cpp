/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/add_norm_layer.h"
#include "device/kunlun/kunlun_device_context_impl.h"

#include "kernels/kunlun/basic_kernel_wrapper.h"

namespace ksana_llm {

Status AddNormLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                          std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);
  int parameter_index = 0;
  rms_norm_eps_ = std::any_cast<const float>(parameters[parameter_index++]);
  KLLM_LOG_DEBUG << fmt::format("rms_norm_eps {}", rms_norm_eps_);
  return Status();
}

template <typename T>
Status AddNormLayer::ForwardT(const Tensor* input, const Tensor* residual, const Tensor* weight, Tensor* output) {
  if (input->GetPtr<void>() != output->GetPtr<void>()) {
    KLLM_THROW("AddNormLayer input and output tensors must be the same because it is an inplace operation.");
  }

  //   input [token_num, hidden_size] modified in-place (= output)
  //   residual [token_num, hidden_size] modified in-place
  //   weight [hidden_size]
  // Step 1: residual[i] += input[i]
  // Step 2: input[i] = rmsnorm(residual[i]) * weight[i]
  xdnn::Context* xctx = static_cast<xdnn::Context*>(
      context_->device->ext->GetXdnnContext(context_->device->GetPrimaryComputeStream().Get()));

  T* input_ptr = const_cast<T*>(input->GetPtr<T>());
  T* residual_ptr = const_cast<T*>(residual->GetPtr<T>());
  // Norm weight is stored in native 16-bit (TW=T) after the 16-bit-norm optimization;
  // read it as T directly — no fp32 cast on load, no GetPtr<float>() here.
  const T* weight_ptr = weight->GetPtr<T>();
  int64_t m = static_cast<int64_t>(input->shape[0]);
  int64_t n = static_cast<int64_t>(input->shape[1]);

  int ret = XpuFusedAddRmsNorm<T, T>(xctx, input_ptr, residual_ptr, weight_ptr, m, n, rms_norm_eps_);
  if (ret != 0) {
    return Status(RET_RUNTIME_FAILED, "XpuFusedAddRmsNorm failed in AddNormLayer.");
  }

  output->shape = input->shape;
  output->dtype = input->dtype;
  return Status();
}

Status AddNormLayer::Forward(const Tensor* input, const Tensor* residual, const Tensor* weight, Tensor* output,
                             const LayerExecContext& exec_ctx) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, input, residual, weight, output);
}

}  // namespace ksana_llm
