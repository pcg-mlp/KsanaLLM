/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "device/zixiao/zixiao_device_context_impl.h"

#include <numeric>

#include "utils/zixiao/tops_utils.h"

namespace ksana_llm {

template <>
void DeviceContextT<DEVICE_TYPE_ZIXIAO>::InitializeExtension() {
  ext = new ZiXiaoDeviceContextExtension<DEVICE_TYPE_ZIXIAO>(this);
  ext->Initialize();
}

template <>
void DeviceContextT<DEVICE_TYPE_ZIXIAO>::DestroyExtension() {
  ext->Destroy();
  delete ext;
}

template <int T>
void ZiXiaoDeviceContextExtension<T>::Initialize() {}

template <int T>
void ZiXiaoDeviceContextExtension<T>::Destroy() {
  KLLM_LOG_DEBUG << "Destroy zixiao context.";
  if (base_ptr_->world_size_ <= 1) {
    return;
  }
}

}  // namespace ksana_llm
