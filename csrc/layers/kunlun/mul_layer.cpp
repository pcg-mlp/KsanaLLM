/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/mul_layer.h"

#include "device/kunlun/kunlun_device_context_impl.h"
#include "kernels/kunlun/moe_kernel_wrapper.h"

namespace ksana_llm {

template <typename T>
Status MulLayer::ForwardT(const Tensor& a, const Tensor& b, Tensor& output) {
  // Pointwise multiplication: output = a * b
  // For shared expert gating in Qwen2-MoE:
  //   a is gate sigmoid [m, 1], b is shared expert output [m, n]
  xdnn::Context* xctx = static_cast<xdnn::Context*>(
      context_->device->ext->GetXdnnContext(context_->device->GetPrimaryComputeStream().Get()));

  int64_t input_0_m = static_cast<int64_t>(a.shape[0]);
  int64_t input_0_n = static_cast<int64_t>(a.shape[1]);
  int64_t input_1_m = static_cast<int64_t>(b.shape[0]);
  int64_t input_1_n = static_cast<int64_t>(b.shape[1]);

  const T* a_ptr = a.GetPtr<T>();
  const T* b_ptr = b.GetPtr<T>();
  T* out = output.GetPtr<T>();

  int ret;
  if (input_0_m == input_1_m && input_0_n == input_1_n) {
    // Element-wise mul: same shape
    ret = XpuMul<T>(xctx, a_ptr, b_ptr, out, input_0_m * input_0_n);
  } else {
    // Broadcast mul via kernel wrapper (handles xdnn type bridging)
    std::vector<int64_t> x_shape = {input_1_m, input_1_n};
    std::vector<int64_t> y_shape = {input_0_m, input_0_n};
    ret = XpuBroadcastMul<T>(xctx, b_ptr, a_ptr, out, x_shape, y_shape);
  }

  if (ret != 0) {
    return Status(RET_RUNTIME_FAILED, "MulLayer kernel failed on Kunlun.");
  }

  size_t output_m = static_cast<size_t>((input_0_m >= input_1_m) ? input_0_m : input_1_m);
  size_t output_n = static_cast<size_t>((input_0_n >= input_1_n) ? input_0_n : input_1_n);
  output.shape = {output_m, output_n};
  output.dtype = b.dtype;
  return Status();
}

Status MulLayer::Forward(const Tensor* a, const Tensor* b, Tensor* output, const LayerExecContext& exec_ctx) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, *a, *b, *output);
}

}  // namespace ksana_llm
