/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <unordered_map>
#include <utility>
#include <vector>

#include "numerous_llm/runtime/context.h"
// #include "numerous_llm/runtime/infer_request.h"
#include "numerous_llm/utils/status.h"

namespace numerous_llm {

// The worker executed on every device.
class Worker {
 public:
  Worker() {}
  ~Worker() {}
  // Execute model inference.
  Status Execute(Context& ctx);
};

}  // namespace numerous_llm
