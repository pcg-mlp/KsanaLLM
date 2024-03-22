/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/utils/ascend_context.h"

namespace ksana_llm {

template<>
void ContextT<DEVICE_TYPE_ASCEND>::InitializeExtension() {
  ext = new AscendContextExtension<DEVICE_TYPE_ASCEND>(this);
}

template<>
void ContextT<DEVICE_TYPE_ASCEND>::DestroyExtension() {
  delete ext;
}


} // namespace ksana_llm
