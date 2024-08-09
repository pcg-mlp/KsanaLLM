/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/batch_scheduler/strategy/base_strategy.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

// The auto-prefix-caching continuous batching implementation.
class ContinuousBatchingStrategy : public BaseScheduleStrategy {
 public:
  ContinuousBatchingStrategy(const BatchSchedulerConfig &batch_scheduler_config, int tp_num,
                             std::shared_ptr<BatchState> batch_state);

  virtual ~ContinuousBatchingStrategy() {}

  virtual void Schedule() override;

 private:
  // True if request timeout.
  inline bool CheckRequestTimeout(const std::shared_ptr<InferRequest> req);

  // True if request finished, that is, arrive max output len or encounter eos.
  inline bool CheckRequestFinish(const std::shared_ptr<InferRequest> req);

  // Schedule the running/swapped/waiting queue.
  void ProcessRunningQueue();
  void ProcessSwappedQueue();
  void ProcessWaitingQueue();

 private:
  // Wait pending swap out/in requests done, and merge these requests.
  // If blocking is false, the function will return immediately even no request finished.
  // If early_stop is false, the function return until all requests finished.
  Status MergePendingSwapinRequests(bool blocking, bool early_stop);
  Status MergePendingSwapoutRequests(bool blocking, bool early_stop);
};

}  // namespace ksana_llm
