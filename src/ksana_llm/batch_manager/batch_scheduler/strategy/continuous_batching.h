/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/batch_manager/batch_scheduler/strategy/base_strategy.h"

#include "ksana_llm/runtime/threadpool.h"

namespace ksana_llm {

// The auto-batching scheduler implementation.
class ContinuousBatchingStrategy : public BaseScheduleStrategy {
 public:
  ContinuousBatchingStrategy(const BatchSchedulerConfig &batch_scheduler_config, int tp_num,
                             std::shared_ptr<BatchState> batch_state);

  virtual ~ContinuousBatchingStrategy();

  virtual void Schedule() override;

 private:
  // True if request timeout.
  inline bool CheckRequestTimeout(const std::shared_ptr<InferRequest> req);

  // True if request finished, that is, arrive max output len or encounter eos.
  inline bool CheckRequestFinish(const std::shared_ptr<InferRequest> req);

  // True if all request finished, that is, arrive max output len or encounter eos.
  inline bool CheckBeamSearchRequestFinish(const std::shared_ptr<InferRequest> req);

 private:
  // Schedule the running/swapped/waiting queue.
  void ScheduleRunning(size_t &step_token_num_sum, bool &skip_other);
  void ScheduleSwapped(size_t &step_token_num_sum, size_t &curr_block_num_sum, bool &skip_other);
  void ScheduleWaiting(size_t &step_token_num_sum, size_t &curr_block_num_sum, bool &skip_other);
  void SchedulePending();

  // Execute swap in separate threadpool.
  void SwapOutAsync(std::shared_ptr<InferRequest> req, const int host_block_num_to_add);
  void SwapInAsync(std::shared_ptr<InferRequest> req);

  bool CheckBeamSearch(std::shared_ptr<InferRequest> req);

  // Prepare the running/swapped/waiting queue.
  void PrepareRunningRequests(std::vector<size_t> &step_token_num_list, std::vector<size_t> &step_block_num_list,
                              std::vector<size_t> &curr_block_num_list);
  void PrepareSwappedRequests(std::vector<size_t> &step_token_num_list, std::vector<size_t> &curr_block_num_list,
                              bool skip_collect = false);
  void PrepareWaitingRequests(std::vector<size_t> &step_token_num_list, std::vector<size_t> &total_block_num_list,
                              bool skip_collect = false);

  // Merge pending swap out/in requests.
  void MergePendingSwapoutRequest();

  // Iterate all request in swapin_pending_queue, take request which swap_pending is false(already finished swapin)
  // into running_queue
  void MergePendingSwapinRequests();

  // Merge Recompute Queue to waiting queue
  void MergeRecomputeQueue();

  // Wait pending swap out/in done.
  void WaitPendingSwapoutDone();
  void WaitPendingSwapinDone();

  // Get the pending block number used by async swapin.
  size_t GetSwapinPendingBlockNumber();

  // Process the running/swapped/waiting queue.
  void ProcessRunningRequests(const std::vector<size_t> &step_token_num_list,
                              const std::vector<size_t> &step_block_num_list,
                              const std::vector<size_t> &curr_block_num_list, std::vector<int> &swapped_indexes,
                              size_t &step_token_num_sum);
  void ProcessSwappedRequests(const std::vector<size_t> &step_token_num_list,
                              const std::vector<size_t> &curr_block_num_list, std::vector<int> &running_indexes,
                              size_t &step_token_num_sum, size_t &total_block_num_sum);
  void ProcessWaitingRequests(const std::vector<size_t> &step_token_num_list,
                              const std::vector<size_t> &total_block_num_list, std::vector<int> &running_indexes,
                              size_t &step_token_num_sum, size_t &total_block_num_sum);

 private:
  // The recompute queue, used when prompt that could not be swapped.
  std::vector<std::shared_ptr<InferRequest>> recompute_queue_;

  // The pending queue used for swap in/out.
  std::vector<std::shared_ptr<InferRequest>> swapin_pending_queue_;
  std::vector<std::shared_ptr<InferRequest>> swapout_pending_queue_;

  // Threadpool used to swap in/out.
  std::shared_ptr<ThreadPool> threadpool_ = nullptr;

  // Preallocate vectors, for speedup.
  std::vector<size_t> running_step_token_num_list_;
  std::vector<size_t> running_step_block_num_list_;
  std::vector<size_t> running_curr_block_num_list_;
  std::vector<int> running_swapped_indexes_;

  std::vector<size_t> swapped_step_token_num_list_;
  std::vector<size_t> swapped_curr_block_num_list_;
  std::vector<int> swapped_running_indexes_;

  std::vector<size_t> waiting_step_token_num_list_;
  std::vector<size_t> waiting_total_block_num_list_;
  std::vector<int> waiting_running_indexes_;
};

}  // namespace ksana_llm
