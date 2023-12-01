/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <vector>

#include "numerous_llm/batch_manager/batch_scheduler/priority/base_priority.h"
#include "numerous_llm/batch_manager/batch_scheduler/strategy/base_strategy.h"
#include "numerous_llm/runtime/infer_request.h"

#include "numerous_llm/utils/environment.h"
#include "numerous_llm/utils/channel.h"

namespace numerous_llm {

class BatchScheduler {
public:
  explicit BatchScheduler(const BatchSchedulerConfig &batch_scheduler_config);
  ~BatchScheduler();

  // Get the next infer reqs that ready to run.
  Status Schedule(std::vector<InferRequest> &scheduled_reqs);

  // Add infer request to waiting list.
  Status AddInferRequest(const InferRequest &infer_request);

  // Stop channel
  Status StopChannel();

private:
  // The scheduler priority.
  std::shared_ptr<BasePriority> priority_;

  // The scheduler granularity.
  std::shared_ptr<BaseGranularity> granularity_;

  // The three queue of current scheduler.
  Channel<InferRequest> waiting_queue_;
  Channel<InferRequest> running_queue_;
  Channel<InferRequest> swapped_queue_;
  Channel<InferRequest> finish_queue_;
};

} // namespace numerous_llm
