/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/add_layer.h"
#include "device/kunlun/kunlun_device_context_impl.h"

#include "kernels/kunlun/basic_kernel_wrapper.h"

namespace ksana_llm {

Status AddLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);
  return Status();
}

template <typename T>
Status AddLayer::ForwardT(const Tensor* a, const Tensor* b, Tensor* output) {
  namespace xpukernel = xpytorch::xpu::api;
  auto* xpuctx = static_cast<xpukernel::Context*>(
      context_->device->ext->GetXpuKernelContext(context_->device->GetPrimaryComputeStream().Get()));

  const T* a_ptr = a->GetPtr<T>();
  const T* b_ptr = b->GetPtr<T>();
  T* out = output->GetPtr<T>();

  int64_t len = static_cast<int64_t>(a->GetElementNumber());

  if (a->shape[0] == b->shape[0]) {
    // Same shape: element-wise add
    int ret = XpuAdd<T>(xpuctx, a_ptr, b_ptr, out, len);
    if (ret != 0) {
      return Status(RET_RUNTIME_FAILED, "XpuAdd failed in AddLayer.");
    }
  } else {
    // Broadcast bias add: b is bias [1, hidden_size] broadcast along dim 0
    // out[i,:] = a[i,:] + b[0,:]
    int64_t m = static_cast<int64_t>(a->shape[0]);
    int64_t n = static_cast<int64_t>(a->shape[1]);
    for (int64_t i = 0; i < m; ++i) {
      int ret = XpuAdd<T>(xpuctx, a_ptr + i * n, b_ptr, out + i * n, n);
      if (ret != 0) {
        return Status(RET_RUNTIME_FAILED, "XpuAdd broadcast failed in AddLayer.");
      }
    }
  }

  output->shape = a->shape;
  output->dtype = a->dtype;
  return Status();
}

Status AddLayer::Forward(const Tensor* a, const Tensor* b, Tensor* output, const LayerExecContext& exec_ctx) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, a, b, output);
}

}  // namespace ksana_llm
