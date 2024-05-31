/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <vector>

#include "ksana_llm/runtime/infer_request.h"

namespace ksana_llm {

struct BatchState {
  BatchState(const BatchSchedulerConfig &batch_scheduler_config) {
    waiting_queue.reserve(batch_scheduler_config.max_waiting_queue_len);
    running_queue.reserve(batch_scheduler_config.max_batch_size);
    swapped_queue.reserve(batch_scheduler_config.max_batch_size);

    waiting_buffer_queue.reserve(batch_scheduler_config.max_waiting_queue_len);
  }

  void MergeWaitingBufferQueue() {
    std::lock_guard<std::mutex> guard(queue_buffer_mutex);

    waiting_queue.insert(waiting_queue.end(), waiting_buffer_queue.begin(), waiting_buffer_queue.end());
    waiting_buffer_queue.clear();
  }

  void ResetInfoBeforeSchedule() { schedule_time_in_ms = GetCurrentTimeInMs(); }

  // The three queue of current scheduler.
  std::vector<std::shared_ptr<InferRequest>> waiting_queue;
  std::vector<std::shared_ptr<InferRequest>> running_queue;
  std::vector<std::shared_ptr<InferRequest>> swapped_queue;

  // The buffer queue used to save input request temporary.
  std::vector<std::shared_ptr<InferRequest>> waiting_buffer_queue;

  // To guard queue.
  std::mutex queue_mutex;

  // Protect the queue buffer.
  std::mutex queue_buffer_mutex;

  // The current timestamp for current schedule loop.
  unsigned long schedule_time_in_ms;
};

}  // namespace ksana_llm
