/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <mutex>
#include <vector>

#include "ksana_llm/batch_manager/batch_scheduler/priority/base_priority.h"
#include "ksana_llm/batch_manager/batch_scheduler/strategy/base_strategy.h"
#include "ksana_llm/runtime/infer_request.h"
#include "ksana_llm/runtime/threadpool.h"
#include "ksana_llm/utils/context.h"
#include "ksana_llm/utils/environment.h"

namespace ksana_llm {

class BatchScheduler {
  public:
    BatchScheduler(const BatchSchedulerConfig &batch_scheduler_config, std::shared_ptr<Context> context);
    ~BatchScheduler();

    // Get the next infer reqs that ready to run.
    std::vector<std::shared_ptr<InferRequest>> &Schedule();

    // Add infer request to waiting list.
    Status AddInferRequest(std::shared_ptr<InferRequest> infer_request);

    // Check whether the waiting buffer is empty.
    bool WaitingBufferEmpty();

    // Check whether the swapped queue is empty.
    bool SwappedQueueEmtpy();

  private:
    // True if request timeout.
    inline bool CheckRequestTimeout(const std::shared_ptr<InferRequest> req);

    // True if waiting queue is already full.
    inline bool CheckWaitingQueueFull();

    // True if request length exceed the max input length.
    inline bool CheckRequestExceedLength(const std::shared_ptr<InferRequest> req);

    // True if request finished, that is, arrive max output len or encounter eos.
    inline bool CheckRequestFinish(const std::shared_ptr<InferRequest> req);

    // Reset necessary informations for scheduling.
    inline void ResetInfoBeforeSchedule();

    // Schedule the running/swapped/waiting queue.
    void ScheduleRunning(size_t &step_token_num_sum, bool &skip_other);
    void ScheduleSwapped(size_t &step_token_num_sum, size_t &curr_block_num_sum, bool &skip_other);
    void ScheduleWaiting(size_t &step_token_num_sum, size_t &curr_block_num_sum, bool &skip_other);
    void SchedulePending();

    // Execute swap in separate threadpool.
    void SwapOutAsync(std::shared_ptr<InferRequest> req);
    void SwapInAsync(std::shared_ptr<InferRequest> req);

    // Prepare the running/swapped/waiting queue.
    void PrepareRunningRequests(std::vector<size_t> &step_token_num_list, std::vector<size_t> &step_block_num_list,
                                std::vector<size_t> &curr_block_num_list);
    void PrepareSwappedRequests(std::vector<size_t> &step_token_num_list, std::vector<size_t> &curr_block_num_list,
                                bool skip_collect = false);
    void PrepareWaitingRequests(std::vector<size_t> &step_token_num_list, std::vector<size_t> &total_block_num_list,
                                bool skip_collect = false);

    // Merge waiting buffer to waiting queue.
    void MergeWaitingBufferQueue();

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
    BatchSchedulerConfig batch_scheduler_config_;

    // The current timestamp for current schedule loop.
    unsigned long schedule_time_in_ms_;

    std::shared_ptr<Context> context_;

    // To guard queue.
    std::mutex queue_mutex_;

    // Protect the queue buffer.
    std::mutex queue_buffer_mutex_;

    // The scheduler priority.
    std::shared_ptr<BasePriority> priority_;

    // The scheduler granularity.
    std::shared_ptr<BaseGranularity> granularity_;

    // The three queue of current scheduler.
    std::vector<std::shared_ptr<InferRequest>> waiting_queue_;
    std::vector<std::shared_ptr<InferRequest>> running_queue_;
    std::vector<std::shared_ptr<InferRequest>> swapped_queue_;

    // The buffer queue used to save input request temporary.
    std::vector<std::shared_ptr<InferRequest>> waiting_buffer_queue_;
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
