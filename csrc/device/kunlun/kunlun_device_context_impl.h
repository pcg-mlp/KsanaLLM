/* Copyright 2026 Tencent Inc.  All rights reserved.

   Kunlun XPU DeviceContext extension. Mirrors zixiao_device_context_impl.h.
==============================================================================*/
#pragma once

#include "device/device_context_impl.h"
#include "device/kunlun/bkcl_utils.h"

namespace ksana_llm {

template <int T>
struct KunlunDeviceContextExtension {
 public:
  explicit KunlunDeviceContextExtension(DeviceContextT<T>* base_ptr) { base_ptr_ = base_ptr; }

  void Initialize();
  void Destroy();

  // Persistent xdnn::Context for all Kunlun kernel calls (analogous to NVIDIA cublasHandle).
  // stream: the XPUStream to bind before returning. Every call must specify a stream
  // to prevent stale stream references (xsglang-style per-call set_stream pattern).
  // Note(zhongzhicao): xdnn::Context依赖的xpu/xdnn.h头文件也包含json库，会导致与nlohmann::json冲突,
  // 所以这里用void*代替XPUStream和xdnn::Context*。
  void* GetXdnnContext(void* stream);

  // Persistent xpukernel::Context for xpytorch/xpukernel API calls (argmax, scatter,
  // gather, sort, etc.). Like xdnn_ctx_, this is a process-level singleton obtained
  // from xmlir_rt, cached here for unified context management.
  void* GetXpuKernelContext(void* stream);

  // BKCL collective communication support
  BKCLParam& GetBKCLParam() { return bkcl_param_; }
  void SetBKCLUniqueId(const BKCLUniqueId& uid) { bkcl_param_.bkcl_uid = uid; }
  void InitBkcl(int rank, int world_size) {
    bkcl_param_.rank = rank;
    bkcl_param_.world_size = world_size;
    if (world_size > 1) {
      BKCL_CHECK(bkcl_init_rank(&bkcl_param_.bkcl_comm, rank, world_size, &bkcl_param_.bkcl_uid));
      KLLM_LOG_INFO << fmt::format("BKCL init rank {}/{} success", rank, world_size);
    }
  }

  // Kunlun does not support multicast/NVLink features
  const bool IsMulticastSupported() const { return false; }
  const bool IsP2PAccessSupported() const { return false; }
  const bool IsFullNvLink() const { return false; }

 private:
  DeviceContextT<T>* base_ptr_ = nullptr;
  void* xdnn_ctx_ = nullptr;
  void* xpukernel_ctx_ = nullptr;  // cached xpytorch::xpu::api::Context*

  std::vector<int32_t> rank_ids_;
  BKCLParam bkcl_param_;
};

template <>
struct DeviceExtensionTypeTraits<DEVICE_TYPE_KUNLUN> {
  typedef KunlunDeviceContextExtension<DEVICE_TYPE_KUNLUN> value_type;
};

template <>
void DeviceContextT<DEVICE_TYPE_KUNLUN>::InitializeExtension();

template <>
void DeviceContextT<DEVICE_TYPE_KUNLUN>::DestroyExtension();

}  // namespace ksana_llm
