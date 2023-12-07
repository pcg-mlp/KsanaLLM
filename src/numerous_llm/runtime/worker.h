/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/runtime/context.h"
#include "numerous_llm/utils/status.h"

#include <unordered_map>
#include <utility>
#include <vector>

namespace numerous_llm {

// The worker executed on every device.
class Worker {
 public:
  // Execute model inference.
  Status Execute(Context& ctx, std::vector<std::pair<ModelInstance, std::vector<InferRequest>>>& reqs);
};

}  // namespace numerous_llm
