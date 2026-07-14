/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "device/ascend/ascend_device_context_impl.h"

#include <numeric>

#include "utils/ascend/acl_utils.h"

namespace ksana_llm {

template <>
void DeviceContextT<DEVICE_TYPE_ASCEND>::InitializeExtension() {
  ext = new AscendDeviceContextExtension<DEVICE_TYPE_ASCEND>(this);
  ext->Initialize();
}

template <>
void DeviceContextT<DEVICE_TYPE_ASCEND>::DestroyExtension() {
  ext->Destroy();
  delete ext;
}

template <int T>
void AscendDeviceContextExtension<T>::InitHcclParam() {
  KLLM_LOG_DEBUG << "Init ascend hccl param.";
  if (base_ptr_->world_size_ <= 1) {
    return;
  }
  rank_ids_.resize(base_ptr_->tensor_parallel_size_, int32_t(0));
  std::iota(rank_ids_.begin() + 1, rank_ids_.end(), int32_t(1));
  hccl_params_.resize(base_ptr_->tensor_parallel_size_);

  HCCL_CHECK(HcclCommInitAll(base_ptr_->tensor_parallel_size_, rank_ids_.data(), hccl_params_.data()));
}

template <int T>
void AscendDeviceContextExtension<T>::Initialize() {
  InitHcclParam();
}

template <int T>
void AscendDeviceContextExtension<T>::Destroy() {
  KLLM_LOG_DEBUG << "Destroy ascend context.";
  if (base_ptr_->world_size_ <= 1) {
    return;
  }

  for (int worker_id = 0; worker_id < base_ptr_->world_size_; ++worker_id) {
    HCCL_CHECK(HcclCommDestroy(hccl_params_[worker_id]));
  }
}

}  // namespace ksana_llm
