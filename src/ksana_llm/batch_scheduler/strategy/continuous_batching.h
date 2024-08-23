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

  // Destroy the request and add it to the begining of waiting queue to recompute.
  void RecomputeRequest(std::shared_ptr<InferRequest> req);

  // Set the finish status of the request to finished, timeout or aborted.
  void StopRequest(std::shared_ptr<InferRequest> req, Status req_status);

  // Update cache manager, process finished and timeout requests.
  void UpdateRunningRequests(size_t &total_needed_block_num);

  // Try to allocate request blocks. If failed, try the allocation again after all swapout finished.
  Status AllocateRequestBlocksWithRetry(std::shared_ptr<InferRequest> req, size_t &total_needed_block_num,
                                        size_t &step_block_num, bool &allocate_block_succ, bool &skip_swapout_check);

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
