/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/activation_layer.h"
#include "device/kunlun/kunlun_device_context_impl.h"

#include "kernels/kunlun/basic_kernel_wrapper.h"
#include "kernels/kunlun/moe_kernel_wrapper.h"

namespace ksana_llm {

Status ActivationLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                             std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);
  activation_type_ = std::any_cast<const ActivationType>(parameters[0]);
  return Status();
}

template <typename T>
Status ActivationLayer::ForwardT(const Tensor* input, Tensor* output, const Tensor* gated_weight, const Tensor* bias,
                                 const Tensor* gated_bias) {
  namespace xpukernel = xpytorch::xpu::api;
  void* stream = context_->device->GetPrimaryComputeStream().Get();
  xdnn::Context* xctx = static_cast<xdnn::Context*>(context_->device->ext->GetXdnnContext(stream));
  auto* xpuctx = static_cast<xpukernel::Context*>(context_->device->ext->GetXpuKernelContext(stream));

  const T* input_ptr = input->GetPtr<T>();
  T* output_ptr = output->GetPtr<T>();
  int64_t m = static_cast<int64_t>(input->shape[0]);
  int64_t n = static_cast<int64_t>(input->shape[1]);
  int ret = 0;

  switch (activation_type_) {
    case ActivationType::Gelu: {
      // Pure gelu without gating
      ret = XpuGatedGelu<T>(xpuctx, input_ptr, /*gated_weight=*/nullptr, output_ptr, m, n);
      break;
    }
    case ActivationType::Geglu: {
      // Gated gelu: gelu(input) * gated_weight
      const T* gated_weight_ptr = (gated_weight != nullptr) ? gated_weight->GetPtr<T>() : nullptr;
      ret = XpuGatedGelu<T>(xpuctx, input_ptr, gated_weight_ptr, output_ptr, m, n);
      break;
    }
    case ActivationType::Swiglu: {
      // Gated silu: silu(input) * gated_weight
      const T* gated_weight_ptr = (gated_weight != nullptr) ? gated_weight->GetPtr<T>() : nullptr;
      if (gated_weight_ptr) {
        ret = XpuSiluMul<T>(xpuctx, input_ptr, gated_weight_ptr, output_ptr, m, n);
      } else {
        // Fallback: row-based split [m, 2n] -> [m, n]
        ret = XpuRowBasedSiluMul<T>(xctx, input_ptr, output_ptr, m, n);
      }
      break;
    }
    case ActivationType::Relu: {
      // For Relu, use silu as approximation on XPU (Stage 2 placeholder).
      // TECHDEBT: stage3-relu: add XpuRelu kernel wrapper when available.
      ret = XpuSilu<T>(xpuctx, input_ptr, output_ptr, m * n);
      break;
    }
    default:
      return Status(RET_RUNTIME_FAILED, "Unsupported activation type in ActivationLayer.");
  }

  if (ret != 0) {
    return Status(RET_RUNTIME_FAILED, "Activation kernel failed in ActivationLayer.");
  }

  output->shape = input->shape;
  output->dtype = input->dtype;
  return Status();
}

Status ActivationLayer::Forward(const Tensor* input, Tensor* output, const Tensor* gated_weight, const Tensor* bias,
                                const Tensor* gated_bias, const LayerExecContext& exec_ctx) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, input, output, gated_weight, bias, gated_bias);
}

// ── SigmoidLayer ────────────────────────────────────────────────────────────

template <typename T>
Status SigmoidLayer::ForwardT(Tensor* inout) {
  // Sigmoid operates in-place on inout.
  xdnn::Context* xctx = static_cast<xdnn::Context*>(
      context_->device->ext->GetXdnnContext(context_->device->GetPrimaryComputeStream().Get()));

  T* data = inout->GetPtr<T>();
  int64_t len = static_cast<int64_t>(inout->shape[0] * inout->shape[1]);

  int ret = XpuSigmoid<T>(xctx, data, data, len);
  if (ret != 0) {
    return Status(RET_RUNTIME_FAILED, "XpuSigmoid failed in SigmoidLayer.");
  }
  return Status();
}

Status SigmoidLayer::Forward(Tensor* inout, const LayerExecContext& exec_ctx) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, inout);
}

}  // namespace ksana_llm
