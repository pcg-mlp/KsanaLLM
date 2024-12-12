/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/utils/common_context.h"

namespace ksana_llm {

// The class used for ascend extension.
template <int T>
struct ZiXiaoContextExtension {
 public:
  explicit ZiXiaoContextExtension(ContextT<T>* base_ptr) { base_ptr_ = base_ptr; }

  // Initialize and destroy extension.
  void Initialize();
  void Destroy();

 private:
  ContextT<T>* base_ptr_ = nullptr;

  std::vector<int32_t> rank_ids_;
};

template <>
struct ExtensionTypeTraits<DEVICE_TYPE_ZIXIAO> {
  typedef ZiXiaoContextExtension<DEVICE_TYPE_ZIXIAO> value_type;
};

// 构造扩展类对象
template <>
void ContextT<DEVICE_TYPE_ZIXIAO>::InitializeExtension();

// 销毁扩展类对象
template <>
void ContextT<DEVICE_TYPE_ZIXIAO>::DestroyExtension();

}  // namespace ksana_llm
