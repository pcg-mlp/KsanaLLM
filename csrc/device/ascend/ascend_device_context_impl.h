/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "runtime/device_context_impl.h"
#include "utils/ascend/hccl_utils.h"

namespace ksana_llm {

// The class used for ascend extension.
template <int T>
struct AscendDeviceContextExtension {
 public:
  explicit AscendDeviceContextExtension(DeviceContextT<T>* base_ptr) { base_ptr_ = base_ptr; }

  const bool IsMulticastSupported() const { return false; }

  // Initialize and destroy extension.
  void Initialize();
  void Destroy();

  std::vector<HcclComm>& GetHCCLComm() { return hccl_params_; }

 private:
  DeviceContextT<T>* base_ptr_ = nullptr;

  // init nccl handle
  void InitHcclParam();

  // hccl comms
  std::vector<HcclComm> hccl_params_;
  std::vector<int32_t> rank_ids_;
};

template <>
struct DeviceExtensionTypeTraits<DEVICE_TYPE_ASCEND> {
  typedef AscendDeviceContextExtension<DEVICE_TYPE_ASCEND> value_type;
};

// Initialize extension.
template <>
void DeviceContextT<DEVICE_TYPE_ASCEND>::InitializeExtension();

// Destroy extension.
template <>
void DeviceContextT<DEVICE_TYPE_ASCEND>::DestroyExtension();

}  // namespace ksana_llm
