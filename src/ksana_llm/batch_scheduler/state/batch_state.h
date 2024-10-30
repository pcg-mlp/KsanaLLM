/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <deque>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#include "ksana_llm/runtime/infer_request.h"

namespace ksana_llm {

struct BatchState {
  explicit BatchState(const BatchSchedulerConfig& batch_scheduler_config)
      : batch_scheduler_config_(batch_scheduler_config) {
    running_queue.reserve(batch_scheduler_config_.max_batch_size);
    waiting_buffer_queue.reserve(batch_scheduler_config_.max_waiting_queue_len);
  }

  void MergeWaitingBufferQueue() {
    std::lock_guard<std::mutex> guard(queue_buffer_mutex);

    for (auto& infer_request : waiting_buffer_queue) {
      if (waiting_queue.size() < batch_scheduler_config_.max_waiting_queue_len) {
        waiting_queue.push_back(infer_request);
      } else {
        KLLM_LOG_DEBUG << "waiting queue is full, req " << infer_request->req_id << " failed.";

        // Reject this request.
        infer_request->finish_status = Status(RET_EXCEED_CAPACITY, "waiting queue is full.");
        infer_request->finished = true;
        infer_request->Notify();
      }
    }
    waiting_buffer_queue.clear();
  }

  void MergeRunningBufferQueue() {
    std::lock_guard<std::mutex> guard(queue_buffer_mutex);

    running_queue.insert(running_queue.end(), running_buffer_queue.begin(), running_buffer_queue.end());
    running_buffer_queue.clear();
  }

  void ResetInfoBeforeSchedule() {
    schedule_time_in_ms = GetCurrentTimeInMs();
    step_sched_finish = false;
  }

  // The config of batch scheduler.
  BatchSchedulerConfig batch_scheduler_config_;

  // The waiting queue, double end queue.
  std::deque<std::shared_ptr<InferRequest>> waiting_queue;

  // The buffer queue used to save input request temporary.
  std::vector<std::shared_ptr<InferRequest>> waiting_buffer_queue;

  // The running queue, vector.
  std::vector<std::shared_ptr<InferRequest>> running_queue;

  // The buffer queue used to save finished swapin request temporary.
  std::vector<std::shared_ptr<InferRequest>> running_buffer_queue;

  // The swapped queue, sorted map.
  std::map<int, std::shared_ptr<InferRequest>> swapped_queue;

  // The pending requests used for swap in/out, unordered.
  std::unordered_map<int, std::shared_ptr<InferRequest>> swapin_pending_requests_;
  std::unordered_map<int, std::shared_ptr<InferRequest>> swapout_pending_requests_;

  // To guard queue.
  std::mutex queue_mutex;

  // Protect the queue buffer.
  std::mutex queue_buffer_mutex;

  // The current timestamp for current schedule loop.
  uint64_t schedule_time_in_ms;

  // Whether current scheduler step have finished.
  bool step_sched_finish = false;
};

}  // namespace ksana_llm
