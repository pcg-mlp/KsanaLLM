/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/batch_manager/batch_scheduler/batch_scheduler.h"
#include "numerous_llm/utils/logger.h"

namespace numerous_llm {

BatchScheduler::BatchScheduler(
    const BatchSchedulerConfig &batch_scheduler_config) {}

Status BatchScheduler::AddInferRequest(const InferRequest &infer_request) {
  waiting_queue_.push_back(infer_request);
  return Status();
}

Status BatchScheduler::Schedule(std::vector<InferRequest> &scheduled_reqs) {
  scheduled_reqs = waiting_queue_;
  waiting_queue_.clear();
  return Status();
}

} // namespace numerous_llm
