/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <vector>

#include "numerous_llm/batch_manager/batch_scheduler/priority/base_priority.h"
#include "numerous_llm/batch_manager/batch_scheduler/strategy/base_strategy.h"
#include "numerous_llm/runtime/context.h"
#include "numerous_llm/runtime/infer_request.h"
#include "numerous_llm/utils/environment.h"

namespace numerous_llm {

class BatchScheduler {
 public:
  BatchScheduler(const BatchSchedulerConfig &batch_scheduler_config, std::shared_ptr<Context> context);
  ~BatchScheduler();

  // Get the next infer reqs that ready to run.
  std::vector<std::shared_ptr<InferRequest>> &Schedule();

  // Add infer request to waiting list.
  Status AddInferRequest(std::shared_ptr<InferRequest> infer_request);

 private:
  // True if request timeout.
  inline bool CheckRequestTimeout(const std::shared_ptr<InferRequest> req);

  // True if waiting queue is already full.
  inline bool CheckWaitingQueueFull();

  // True if request finished, that is, arrive max output len or encounter eos.
  inline bool CheckRequestFinish(const std::shared_ptr<InferRequest> req);

  // Reset necessary informations for scheduling.
  inline void ResetSchedule();

  // Schedule the running/swapped/waiting queue.
  void ScheduleRunning(size_t &total_token_num, size_t &total_block_num, bool &schedule_step_finish,
                       size_t max_free_block_num);

  void ScheduleSwapped(size_t &total_token_num, size_t &total_block_num, bool &schedule_step_finish,
                       size_t max_free_block_num);

  void ScheduleWaiting(size_t &total_token_num, size_t &total_block_num, bool &schedule_step_finish,
                       size_t max_free_block_num);

 private:
  BatchSchedulerConfig batch_schedule_config_;

  // The current timestamp for current schedule loop.
  unsigned long schedule_time_in_ms_;

  std::shared_ptr<Context> context_;

  // To guard queue.
  std::mutex queue_mutex_;

  // The scheduler priority.
  std::shared_ptr<BasePriority> priority_;

  // The scheduler granularity.
  std::shared_ptr<BaseGranularity> granularity_;

  // The three queue of current scheduler.
  std::vector<std::shared_ptr<InferRequest>> waiting_queue_;
  std::vector<std::shared_ptr<InferRequest>> running_queue_;
  std::vector<std::shared_ptr<InferRequest>> swapped_queue_;
};

}  // namespace numerous_llm
