/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "runtime/device_context_impl.h"

namespace ksana_llm {

// The class used for ascend extension.
template <int T>
struct ZiXiaoDeviceContextExtension {
 public:
  explicit ZiXiaoDeviceContextExtension(DeviceContextT<T>* base_ptr) { base_ptr_ = base_ptr; }

  // Initialize and destroy extension.
  void Initialize();
  void Destroy();

 private:
  DeviceContextT<T>* base_ptr_ = nullptr;

  std::vector<int32_t> rank_ids_;
};

template <>
struct DeviceExtensionTypeTraits<DEVICE_TYPE_ZIXIAO> {
  typedef ZiXiaoDeviceContextExtension<DEVICE_TYPE_ZIXIAO> value_type;
};

// Initialize extension.
template <>
void DeviceContextT<DEVICE_TYPE_ZIXIAO>::InitializeExtension();

// Destroy extension.
template <>
void DeviceContextT<DEVICE_TYPE_ZIXIAO>::DestroyExtension();

}  // namespace ksana_llm
