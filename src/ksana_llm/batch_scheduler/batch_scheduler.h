/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <mutex>
#include <vector>

#include "ksana_llm/batch_scheduler/batch_scheduler_interface.h"
#include "ksana_llm/batch_scheduler/state/batch_state.h"
#include "ksana_llm/batch_scheduler/strategy/strategy_factory.h"

#include "ksana_llm/cache_manager/cache_manager_interface.h"

#include "ksana_llm/runtime/threadpool.h"
#include "ksana_llm/utils/context.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/tokenizer.h"

namespace ksana_llm {

class BatchScheduler : public BatchSchedulerInterface {
 public:
  BatchScheduler(const BatchSchedulerConfig &batch_scheduler_config, int tp_num);
  ~BatchScheduler() {}

  // Get the next infer reqs that ready to run.
  std::vector<std::shared_ptr<InferRequest>> &Schedule();

  // Add infer request to waiting list.
  Status AddInferRequest(std::vector<std::shared_ptr<InferRequest>> &infer_request_group);

  // Set the cache manager instance of batch scheduler.
  void SetCacheManager(std::shared_ptr<CacheManagerInterface> cache_manager);

  std::shared_ptr<CacheManagerInterface> &GetCacheManager();

  // Whether the scheduler is idle, that is, waiting buffer and swapped queue is both empty.
  bool IsIdle();

  // Set the tokenizer of batch scheduler.
  void SetTokenizer(std::shared_ptr<Tokenizer> tokenizer);

 private:
  // Add infer requests to waiting buffer queue, and reject requests if the queue is full.
  Status EnqueueWaitingBufferQueue(std::vector<std::shared_ptr<InferRequest>> &infer_request_group);

  // True if request length exceed the max input length.
  inline bool CheckRequestExceedLength(const std::shared_ptr<InferRequest> req);

 private:
  // The config of batch scheduler.
  BatchSchedulerConfig batch_scheduler_config_;

  // The batch state informations, include some queues and mutexes.
  std::shared_ptr<BatchState> batch_state_ = nullptr;

  // The batch strategy implementation.
  std::shared_ptr<BaseScheduleStrategy> schedule_strategy_ = nullptr;
};

}  // namespace ksana_llm
