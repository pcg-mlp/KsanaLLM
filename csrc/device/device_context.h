/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "device/device_context_impl.h"

#ifdef ENABLE_CUDA
#  include "device/nvidia/nvidia_device_context_impl.h"
#endif

#ifdef ENABLE_ACL
#  include "device/ascend/ascend_device_context_impl.h"
#endif

#ifdef ENABLE_TOPS
#  include "device/zixiao/zixiao_device_context_impl.h"
#endif

#ifdef ENABLE_KUNLUN
#  include "device/kunlun/kunlun_device_context_impl.h"
#endif

namespace ksana_llm {

// The context for different device type.
using DeviceContext = DeviceContextT<ACTIVE_DEVICE_TYPE>;

// Helper: get platform-specific kernel context as void* for cross-platform kernel calls.
// On Kunlun: binds the given stream to the persistent xdnn::Context and returns it.
// On other platforms: returns nullptr.
inline void* GetDeviceCtxPtr(DeviceContext* device, Stream& stream) {
#ifdef ENABLE_KUNLUN
  return device->ext->GetXdnnContext(stream.Get());
#else
  (void)device;
  (void)stream;
  return nullptr;
#endif
}

// Helper: get xpukernel (xpytorch::xpu::api) context for xpytorch-path kernels.
// On Kunlun: returns the cached xpukernel::Context* with stream bound.
// On other platforms: returns nullptr.
inline void* GetXpuKernelCtxPtr(DeviceContext* device, Stream& stream) {
#ifdef ENABLE_KUNLUN
  return device->ext->GetXpuKernelContext(stream.Get());
#else
  (void)device;
  (void)stream;
  return nullptr;
#endif
}

}  // namespace ksana_llm
