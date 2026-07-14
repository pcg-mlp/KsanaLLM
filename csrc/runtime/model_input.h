/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <unordered_map>

#include "runtime/model_instance.h"
#include "runtime/schedule_output.h"

namespace ksana_llm {

class NewModelInput {
 public:
  RunMode run_mode;

  // The schedule output.
  std::shared_ptr<ScheduleOutput> schedule_output;

  std::vector<ForwardRequest> reqs;
};

}  // namespace ksana_llm
