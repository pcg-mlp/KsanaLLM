/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/batch_manager/batch_scheduler/batch_scheduler.h"

#include <utility>

#include "src/numerous_llm/utils/channel.h"

namespace numerous_llm {

BatchScheduler::BatchScheduler(
    const BatchSchedulerConfig &batch_scheduler_config) {}

Status BatchScheduler::AddInferRequest(const InferRequest &infer_request) {
  InferRequest infer_requestl_inner = infer_request;
  waiting_queue_.Write(std::move(infer_request));
  return Status();
}

Status BatchScheduler::Schedule(std::vector<InferRequest> &scheduled_reqs) {
  // Fetch one

  // TODO(karlluo): strategy for schedule how many request
  InferRequest infer_req;
  waiting_queue_.Read(&infer_req);

  return Status();
}

} // namespace numerous_llm
