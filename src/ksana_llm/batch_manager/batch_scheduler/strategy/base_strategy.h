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
    BaseScheduleStrategy(const BatchSchedulerConfig &batch_scheduler_config, int tp_num,
                         std::shared_ptr<BatchState> batch_state)
        : batch_state_(batch_state), batch_scheduler_config_(batch_scheduler_config), tp_num_(tp_num) {}

    // Get the next infer reqs that ready to run.
    virtual void Schedule() = 0;

  protected:
    // The batch state informations, include some queues and mutexes.
    std::shared_ptr<BatchState> batch_state_ = nullptr;

    // the config and context.
    BatchSchedulerConfig batch_scheduler_config_;
    int tp_num_;
};

}  // namespace ksana_llm
