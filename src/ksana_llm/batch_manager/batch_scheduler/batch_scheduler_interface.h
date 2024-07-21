/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <vector>

#include "ksana_llm/runtime/infer_request.h"

namespace ksana_llm {

class BatchSchedulerInterface {
 public:
  virtual ~BatchSchedulerInterface() {}

  // Get the next infer reqs that ready to run.
  virtual std::vector<std::shared_ptr<InferRequest>> &Schedule() = 0;

  // Add infer request to waiting list.
  virtual Status AddInferRequest(std::vector<std::shared_ptr<InferRequest>> &infer_request_group) = 0;

  // Check whether the waiting buffer is empty.
  virtual bool WaitingBufferEmpty() = 0;

  // Check whether the swapped queue is empty.
  virtual bool SwappedQueueEmtpy() = 0;
};

}  // namespace ksana_llm
