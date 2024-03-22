/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/utils/common_context.h"
#include "ksana_llm/utils/device_types.h"

namespace ksana_llm {

// The class used for ascend extension.
template <int T>
struct AscendContextExtension {
  AscendContextExtension(ContextT<T>* base_ptr) {
    base_ptr_ = base_ptr;
  }

 private:
  ContextT<T>* base_ptr_ = nullptr;
};

template <>
struct ExtensionTypeTraits<DEVICE_TYPE_ASCEND> {
  typedef AscendContextExtension<DEVICE_TYPE_ASCEND> value_type;
};

// 构造扩展类对象
template<>
void ContextT<DEVICE_TYPE_ASCEND>::InitializeExtension();

// 销毁扩展类对象
template<>
void ContextT<DEVICE_TYPE_ASCEND>::DestroyExtension();

} // namespace ksana_llm
