/* Copyright 2026 Tencent Inc.  All rights reserved.

   Kunlun XPU DeviceContext extension stub.
==============================================================================*/
#include "device/kunlun/kunlun_device_context_impl.h"

#include <xpu/xdnn.h>
#include "runtime/xpu_context.h"          // xmlir_rt::getXpuKernelContext
#include "xpu/refactor/xpukernel_math.h"  // xpytorch::xpu::api::Context

#include "device/kunlun/xpu_utils.h"

namespace ksana_llm {

template <>
void DeviceContextT<DEVICE_TYPE_KUNLUN>::InitializeExtension() {
  ext = new KunlunDeviceContextExtension<DEVICE_TYPE_KUNLUN>(this);
  ext->Initialize();
}

template <>
void DeviceContextT<DEVICE_TYPE_KUNLUN>::DestroyExtension() {
  ext->Destroy();
  delete ext;
}

template <int T>
void KunlunDeviceContextExtension<T>::Initialize() {
  xdnn::Context* ctx = xdnn::create_context();
  if (ctx == nullptr) {
    KLLM_LOG_ERROR << "xdnn::create_context() returned nullptr";
    KLLM_THROW("Failed to create xdnn context");
  }
  xdnn_ctx_ = static_cast<void*>(ctx);

  // Cache the xpukernel context (process-level singleton from xmlir_rt) —
  // required by all xpytorch::xpu::api kernels (argmax, scatter, gather, etc.).
  xpukernel_ctx_ = xmlir_rt::getXpuKernelContext(/*useCpu=*/false);
}

template <int T>
void* KunlunDeviceContextExtension<T>::GetXdnnContext(void* stream) {
  auto* ctx = static_cast<xdnn::Context*>(xdnn_ctx_);
  ctx->set_stream(static_cast<XPUStream>(stream), false);
  return xdnn_ctx_;
}

template <int T>
void KunlunDeviceContextExtension<T>::Destroy() {
  if (xdnn_ctx_) {
    auto* ctx = static_cast<xdnn::Context*>(xdnn_ctx_);
    xdnn::destroy_context(ctx);
    xdnn_ctx_ = nullptr;
  }
  if (bkcl_param_.bkcl_comm != nullptr) {
    DestroyBKCLParam(bkcl_param_);
  }
}

template void* KunlunDeviceContextExtension<DEVICE_TYPE_KUNLUN>::GetXdnnContext(void*);

template <int T>
void* KunlunDeviceContextExtension<T>::GetXpuKernelContext(void* stream) {
  auto* ctx = static_cast<xpytorch::xpu::api::Context*>(xpukernel_ctx_);
  ctx->set_stream(static_cast<XPUStream>(stream), false);
  return xpukernel_ctx_;
}

template void* KunlunDeviceContextExtension<DEVICE_TYPE_KUNLUN>::GetXpuKernelContext(void*);

}  // namespace ksana_llm
