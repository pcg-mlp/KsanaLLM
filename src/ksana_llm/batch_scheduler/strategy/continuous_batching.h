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

  // Returns true if the request needs to jump forward with a constant string.
  void CheckJumpForwardRequest(std::shared_ptr<InferRequest> req);

  // Expand structured output with a constant string (dependent on the execution of Retokenizationr by Tokenizer).
  void ExtendTokensWithRetokenization(std::shared_ptr<InferRequest> req);

  // Expand structured output with a constant string (independent of Retokenizationr execution by Tokenizer).
  void ExtendTokensWithoutRetokenization(std::shared_ptr<InferRequest> req);

  // Destroy the request and add it to the begining of waiting queue to recompute.
  void RecomputeRequest(std::shared_ptr<InferRequest> req);

  // Set the finish status of the request to finished, timeout or aborted.
  void StopRequest(std::shared_ptr<InferRequest> req, Status req_status);

  // Check the running queue to determine whether it exceeds the max_step_tokens.
  void CheckRunningQueueStepTokens(size_t& step_token_num);

  // Update cache manager, process finished and timeout requests.
  void UpdateRunningRequests(size_t &total_needed_block_num);

  // Try to allocate request blocks. If failed, try the allocation again after all swapout finished.
  Status AllocateRequestBlocksWithRetry(std::shared_ptr<InferRequest> req, size_t &total_needed_block_num,
                                        size_t &step_block_num, bool &allocate_block_succ, bool &skip_swapout_check);

  /**
   * Processes a request to determine the appropriate number of tokens to split or fuse based on the current
   * batching strategy configuration. This function adjusts the number of output tokens in the request to match
   * the calculated split or fuse token count, and updates the shared and unique block counts accordingly.
   *
   * The function aims to optimize the processing of requests by dynamically adjusting the number of tokens
   * to be processed together, based on the configured thresholds and the current state of the request and
   * batch scheduler.
   */
  bool ProcessSplitFuseToken(std::shared_ptr<InferRequest> req, size_t &shared_block_num, size_t &unique_block_num,
                             size_t &shared_token_num, size_t step_token_num, size_t decode_request_num);

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
