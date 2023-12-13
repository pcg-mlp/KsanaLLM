/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/batch_manager/batch_scheduler/batch_scheduler.h"

#include <utility>
#include <vector>

#include "numerous_llm/runtime/infer_request.h"
#include "numerous_llm/utils/channel.h"
#include "numerous_llm/utils/logger.h"
#include "numerous_llm/utils/ret_code.h"
#include "numerous_llm/utils/string_utils.h"

namespace numerous_llm {

BatchScheduler::BatchScheduler(const BatchSchedulerConfig &batch_scheduler_config)
    : batch_schedule_config_(batch_scheduler_config) {}

BatchScheduler::~BatchScheduler() {}

Status BatchScheduler::AddInferRequest(std::shared_ptr<InferRequest> infer_request) {
  NLLM_LOG_INFO << "Add infer req.";
  if (CheckWaitingQueueFull()) {
    infer_request->finish_status = Status(RET_EXCEED_CAPACITY, "waiting queue is full.");

    std::lock_guard<std::mutex> guard(queue_mutex_);
    finish_queue_.push_back(infer_request);
    return infer_request->finish_status;
  }

  std::lock_guard<std::mutex> guard(queue_mutex_);
  waiting_queue_.push_back(infer_request);
  return Status();
}

bool BatchScheduler::CheckRequestTimeout(const std::shared_ptr<InferRequest> req) {
  return schedule_time_in_ms_ - req->timestamp_in_ms >= batch_schedule_config_.timeout_in_ms;
}

bool BatchScheduler::CheckWaitingQueueFull() {
  return waiting_queue_.size() >= batch_schedule_config_.max_waiting_queue_len;
}

bool BatchScheduler::CheckRequestFinish(const std::shared_ptr<InferRequest> req) { return false; }

void BatchScheduler::ResetSchedule() { schedule_time_in_ms_ = GetCurrentTimeInMs(); }

void BatchScheduler::ScheduleRunning(size_t &total_token_num, size_t &total_block_num, bool &schedule_step_finish,
                                     size_t max_free_block_num) {
  for (auto it = running_queue_.begin(); it != running_queue_.end();) {
    auto req = *it;

    // Check if finished.
    if (CheckRequestFinish(req)) {
      req->finish_status = Status(RET_SUCCESS);
      finish_queue_.push_back(req);
      running_queue_.erase(it);
      req->waiter->Notify();
      continue;
    }

    // Check timeout
    if (CheckRequestTimeout(req)) {
      req->finish_status = Status(RET_TIMEOUT, "running timeout.");
      finish_queue_.push_back(req);
      running_queue_.erase(it);
      req->waiter->Notify();
      continue;
    }

    // Check total token number and block number.
    total_token_num += req->GetStepTokenNumber();
    total_block_num += req->GetStepBlockNumber();
    if (total_token_num > batch_schedule_config_.max_token_number || total_block_num > max_free_block_num) {
      req->SwapOutAsync();
      swapped_queue_.push_back(req);
      running_queue_.erase(it);
      schedule_step_finish = true;
      continue;
    }

    // Swap left running reqs.
    if (schedule_step_finish) {
      req->SwapOutAsync();
      swapped_queue_.push_back(req);
      running_queue_.erase(it);
      continue;
    }

    // continue running.
    ++it;
  }
}

void BatchScheduler::ScheduleSwapped(size_t &total_token_num, size_t &total_block_num, bool &schedule_step_finish,
                                     size_t max_free_block_num) {
  for (auto it = swapped_queue_.begin(); it != swapped_queue_.end();) {
    auto req = *it;

    // Check timeout, no finished req in swapped queue.
    if (CheckRequestTimeout(req)) {
      req->finish_status = Status(RET_TIMEOUT, "running timeout.");
      finish_queue_.push_back(req);
      swapped_queue_.erase(it);
      req->waiter->Notify();
      continue;
    }

    // All the bocks must be swapped in for swapped reqs.
    total_token_num += req->GetStepTokenNumber();
    total_block_num += req->GetTotalBlockNumber();
    if (total_token_num <= batch_schedule_config_.max_token_number && total_block_num <= max_free_block_num) {
      running_queue_.push_back(req);
      swapped_queue_.erase(it);
      continue;
    }

    // stay swapped.
    schedule_step_finish = true;
    ++it;
  }
}

void BatchScheduler::ScheduleWaiting(size_t &total_token_num, size_t &total_block_num, bool &schedule_step_finish,
                                     size_t max_free_block_num) {
  for (auto it = waiting_queue_.begin(); it != waiting_queue_.end();) {
    auto &req = *it;

    // Check timeout
    if (CheckRequestTimeout(req)) {
      req->finish_status = Status(RET_TIMEOUT, "running timeout.");
      finish_queue_.push_back(req);
      waiting_queue_.erase(it);
      req->waiter->Notify();
      continue;
    }

    total_token_num += req->GetStepTokenNumber();
    total_block_num += req->GetTotalBlockNumber();
    if (total_token_num <= batch_schedule_config_.max_token_number && total_block_num <= max_free_block_num) {
      running_queue_.push_back(req);
      waiting_queue_.erase(it);
      continue;
    }

    // stay waiting.
    schedule_step_finish = true;
    break;
  }
}

std::vector<std::shared_ptr<InferRequest>> &BatchScheduler::Schedule() {
  std::lock_guard<std::mutex> guard(queue_mutex_);

  // TODO(yancyliu): Get from block manager.
  size_t max_free_block_num = 1024;

  size_t total_token_num = 0;
  size_t total_block_num = 0;

  bool schedule_step_finish = false;
  ScheduleRunning(total_token_num, total_block_num, schedule_step_finish, max_free_block_num);

  if (!schedule_step_finish) {
    ScheduleSwapped(total_token_num, total_block_num, schedule_step_finish, max_free_block_num);
  }

  if (!schedule_step_finish) {
    ScheduleWaiting(total_token_num, total_block_num, schedule_step_finish, max_free_block_num);
  }

  return running_queue_;
}

}  // namespace numerous_llm
