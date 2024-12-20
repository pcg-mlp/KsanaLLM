/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <vector>

#include "ksana_llm/cache_manager/cache_manager_interface.h"
#include "ksana_llm/data_hub/schedule_output.h"
#include "ksana_llm/runtime/infer_request.h"

namespace ksana_llm {

class BatchSchedulerInterface {
 public:
  virtual ~BatchSchedulerInterface() {}

  // Get the next infer reqs that ready to run.
  virtual ScheduleOutput* Schedule() = 0;

  // Add infer request to waiting list.
  virtual Status AddInferRequest(std::vector<std::shared_ptr<InferRequest>> &infer_request_group) = 0;

  // Set the cache manager instance of batch scheduler.
  virtual void SetCacheManager(std::shared_ptr<CacheManagerInterface> cache_manager) = 0;

  // Get cache manager
  virtual std::shared_ptr<CacheManagerInterface> &GetCacheManager() = 0;

  // Whether the scheduler is idle, that is, waiting buffer and swapped queue is both empty.
  virtual bool IsIdle() = 0;
};

}  // namespace ksana_llm
