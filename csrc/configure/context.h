/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "device/device_context.h"
#include "configure/global_context.h"

namespace ksana_llm {

struct Context {
  // The global part.
  std::shared_ptr<GlobalContext> global;

  // The device part, such as streams.
  std::shared_ptr<DeviceContext> device;
};

}  // namespace ksana_llm
