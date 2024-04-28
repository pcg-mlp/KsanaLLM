/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <vector>

#include "ksana_llm/batch_manager/batch_scheduler/state/batch_state.h"
#include "ksana_llm/runtime/infer_request.h"

namespace ksana_llm {

class BaseScheduleStrategy {
  public:
    BaseScheduleStrategy(const BatchSchedulerConfig &batch_scheduler_config, std::shared_ptr<Context> context,
                         std::shared_ptr<BatchState> batch_state)
        : batch_state_(batch_state), batch_scheduler_config_(batch_scheduler_config), context_(context) {}

    // Get the next infer reqs that ready to run.
    virtual void Schedule() = 0;

  protected:
    // The batch state informations, include some queues and mutexes.
    std::shared_ptr<BatchState> batch_state_ = nullptr;

    // the config and context.
    BatchSchedulerConfig batch_scheduler_config_;
    std::shared_ptr<Context> context_;
};

}  // namespace ksana_llm
