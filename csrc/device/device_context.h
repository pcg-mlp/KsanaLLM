/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "device/device_context_impl.h"

#ifdef ENABLE_CUDA
#  include "device/nvidia/nvidia_device_context_impl.h"
#endif

#ifdef ENABLE_ACL
#  include "device/ascend/ascend_device_context_impl.h"
#endif

#ifdef ENABLE_TOPS
#  include "device/zixiao/zixiao_device_context_impl.h"
#endif

namespace ksana_llm {

// The context for different device type.
using DeviceContext = DeviceContextT<ACTIVE_DEVICE_TYPE>;

}  // namespace ksana_llm
