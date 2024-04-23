/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/utils/common_context.h"

#ifdef ENABLE_CUDA
#  include "ksana_llm/utils/nvidia/nvidia_context.h"
#endif

#ifdef ENABLE_ACL
#  include "ksana_llm/utils/ascend/ascend_context.h"
#endif

namespace ksana_llm {

// The context for different device type.
using Context = ContextT<ACTIVE_DEVICE_TYPE>;

}  // namespace ksana_llm
