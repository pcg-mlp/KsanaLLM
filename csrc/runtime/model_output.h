/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "runtime/schedule_output.h"

namespace ksana_llm {

class NewModelOutput {
 public:
  RunMode run_mode;

  // The schedule output.
  std::shared_ptr<ScheduleOutput> schedule_output;

  // hidden buffer for MTP, only used in sampling node
  HiddenUnitDeviceBuffer* hidden_unit;
};

}  // namespace ksana_llm
