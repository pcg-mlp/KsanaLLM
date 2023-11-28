/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "numerous_llm/batch_manager/scheduler/priority/base_priority.h"
#include "numerous_llm/batch_manager/scheduler/strategy/base_granularity.h"

class BatchScheduler {
public:
  BatchScheduler();
  ~BatchScheduler();

private:
  // The scheduler priority.
  std::shared_ptr<BasePriority> priority_;

  // The scheduler granularity.
  std::shared_ptr<BaseGranularity> granularity_;
};
