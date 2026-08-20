/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/layernorm_layer.h"
#include "device/kunlun/kunlun_device_context_impl.h"

#include "kernels/kunlun/basic_kernel_wrapper.h"

namespace ksana_llm {

Status LayernormLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                            std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);
  int parameter_index = 0;
  rms_norm_eps_ = std::any_cast<const float>(parameters[parameter_index++]);
  KLLM_LOG_DEBUG << fmt::format("rms_norm_eps {}", rms_norm_eps_);
  return Status();
}

template <typename T>
Status LayernormLayer::ForwardNoWeightT(const Tensor& input, Tensor& output) {
  // No-weight per-head RMSNorm is not implemented on the Kunlun XPU backend.
  return Status(RET_RUNTIME_FAILED, "No-weight per-head RMSNorm not supported in LayernormLayer on Kunlun.");
}

template <typename T>
Status LayernormLayer::ForwardWeightT(const Tensor& input, const Tensor& weight, Tensor& output, const Tensor* bias) {
  // input:  [token_num, hidden_size]
  // weight: [hidden_size]
  // bias:   [hidden_size] (optional; nullptr for RMSNorm)
  // output: [token_num, hidden_size]
  namespace xpukernel = xpytorch::xpu::api;
  void* stream = context_->device->GetPrimaryComputeStream().Get();
  xdnn::Context* xctx = static_cast<xdnn::Context*>(context_->device->ext->GetXdnnContext(stream));
  auto* xpuctx = static_cast<xpukernel::Context*>(context_->device->ext->GetXpuKernelContext(stream));

  const T* input_ptr = input.GetPtr<T>();
  T* output_ptr = output.GetPtr<T>();
  // Norm affine weights are stored in native 16-bit (TW=T) after the 16-bit-norm optimization;
  // the kernel reads them as T directly — no fp32 cast on load, no GetPtr<float>() here.
  const T* scale = weight.GetPtr<T>();
  int64_t m = static_cast<int64_t>(input.shape[0]);
  int64_t n = static_cast<int64_t>(input.shape[1]);

  if (bias != nullptr) {
    // LayerNorm with bias
    const T* bias_ptr = bias->GetPtr<T>();
    int ret = XpuLayerNorm<T, T>(xpuctx, input_ptr, output_ptr, m, n, rms_norm_eps_, scale, bias_ptr,
                                 /*mean=*/nullptr, /*var=*/nullptr);
    if (ret != 0) {
      return Status(RET_RUNTIME_FAILED, "XpuLayerNorm failed in LayernormLayer.");
    }
  } else {
    // RMS Norm (no bias)
    int ret = XpuRmsLayerNorm<T, T, T>(xctx, input_ptr, output_ptr, m, n, rms_norm_eps_, scale,
                                       /*bias=*/nullptr, /*var=*/nullptr);
    if (ret != 0) {
      return Status(RET_RUNTIME_FAILED, "XpuRmsLayerNorm failed in LayernormLayer.");
    }
  }

  output.shape = input.shape;
  output.dtype = input.dtype;
  return Status();
}

template <typename T>
Status LayernormLayer::ForwardParallelT(const Tensor& in1, const Tensor& w1, const Tensor& in2, const Tensor& w2,
                                        Tensor& out1, Tensor& out2, int stride1, int stride2) {
  // Parallel two-path norm is not implemented on the Kunlun XPU backend.
  return Status(RET_RUNTIME_FAILED, "Parallel RMSNorm not supported in LayernormLayer on Kunlun.");
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
