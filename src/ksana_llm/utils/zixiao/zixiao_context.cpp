/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/utils/zixiao/zixiao_context.h"

#include <numeric>

#include "ksana_llm/utils/zixiao/tops_utils.h"

namespace ksana_llm {

template <>
void ContextT<DEVICE_TYPE_ZIXIAO>::InitializeExtension() {
  ext = new ZiXiaoContextExtension<DEVICE_TYPE_ZIXIAO>(this);
  ext->Initialize();
}

template <>
void ContextT<DEVICE_TYPE_ZIXIAO>::DestroyExtension() {
  ext->Destroy();
  delete ext;
}

template <int T>
void ZiXiaoContextExtension<T>::Initialize() {}

template <int T>
void ZiXiaoContextExtension<T>::Destroy() {
  KLLM_LOG_DEBUG << "Destroy zixiao context.";
  if (base_ptr_->tensor_parallel_size_ <= 1) {
    return;
  }
}

}  // namespace ksana_llm
