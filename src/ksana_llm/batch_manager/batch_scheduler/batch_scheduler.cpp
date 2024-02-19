/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/batch_manager/batch_scheduler/batch_scheduler.h"

#include <memory>
#include <utility>
#include <vector>

#include "ksana_llm/block_manager/block_manager.h"
#include "ksana_llm/block_manager/memory_block.h"
#include "ksana_llm/runtime/context.h"
#include "ksana_llm/runtime/infer_request.h"
#include "ksana_llm/utils/channel.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/memory_utils.h"
#include "ksana_llm/utils/ret_code.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/string_utils.h"

namespace ksana_llm {

BatchScheduler::BatchScheduler(const BatchSchedulerConfig &batch_scheduler_config, std::shared_ptr<Context> context)
    : batch_schedule_config_(batch_scheduler_config), context_(context) {}

BatchScheduler::~BatchScheduler() {}

Status BatchScheduler::AddInferRequest(std::shared_ptr<InferRequest> infer_request) {
  NLLM_LOG_DEBUG << "batch scheduler add infer req " << infer_request->req_id << ".";
  if (CheckWaitingQueueFull()) {
    infer_request->finish_status = Status(RET_EXCEED_CAPACITY, "waiting queue is full.");
    infer_request->finished = true;
    infer_request->Notify();

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

bool BatchScheduler::CheckRequestFinish(const std::shared_ptr<InferRequest> req) {
  if (req->infer_stage == InferStage::STATE_DECODE) {
    if (req->output_tokens.size() > req->input_tokens.size() &&
        ((req->output_tokens.back()) == req->model_instance->GetModelConfig().end_id ||
         req->output_tokens.size() >= req->model_instance->GetMaxTokenNum())) {
      return true;
    }
  }
  return false;
}

void BatchScheduler::ResetSchedule() { schedule_time_in_ms_ = GetCurrentTimeInMs(); }

void BatchScheduler::ScheduleRunning(size_t &total_token_num, size_t &total_block_num, bool &schedule_step_finish,
                                     size_t &max_free_block_num, size_t &total_swapout_num) {
  for (auto it = running_queue_.begin(); it != running_queue_.end();) {
    auto req = *it;
    NLLM_LOG_DEBUG << "try req " << req->req_id << " in running_queue_";

    req->ResetInferStage();

    // Check if finished.
    if (CheckRequestFinish(req)) {
      NLLM_LOG_DEBUG << "req " << req->req_id << " finished.";

      max_free_block_num += req->GetCurrentBlockNumber();

      req->finish_status = Status(RET_SUCCESS);
      it = running_queue_.erase(it);
      req->finished = true;
      req->Notify();
      continue;
    }

    // Check timeout
    if (CheckRequestTimeout(req)) {
      NLLM_LOG_DEBUG << "req " << req->req_id << " timeout in running.";

      max_free_block_num += req->GetCurrentBlockNumber();

      req->finish_status = Status(RET_TIMEOUT, "running timeout.");
      it = running_queue_.erase(it);
      req->finished = true;
      req->Notify();
      continue;
    }

    // Notify streaming iterator if needed.
    req->NotifyStep();

    if (req->infer_stage == InferStage::STAGE_CONTEXT) {
      NLLM_LOG_DEBUG << "req " << req->req_id << " change from context decode to decode";
      req->infer_stage = InferStage::STATE_DECODE;
    }

    // Swap left running reqs if schedule step finished.
    if (schedule_step_finish) {
      NLLM_LOG_DEBUG << "req " << req->req_id << " following swapped out.";
      NLLM_LOG_DEBUG << "Reason: total_token_num:" << total_token_num
                     << ", max_token_number:" << batch_schedule_config_.max_token_number
                     << ", total_block_num:" << total_block_num << ", max_free_block_num:" << max_free_block_num;

      total_swapout_num += 1;
      max_free_block_num += req->GetCurrentBlockNumber();

      NLLM_LOG_DEBUG << "Result: total_token_num:" << total_token_num
                     << ", max_token_number:" << batch_schedule_config_.max_token_number
                     << ", total_block_num:" << total_block_num << ", max_free_block_num:" << max_free_block_num;

      req->SwapOutAsync();
      swapped_queue_.push_back(req);
      it = running_queue_.erase(it);
      continue;
    }

    // Check total token number and block number.
    size_t step_token_num_wanted = req->GetStepTokenNumber();
    size_t step_block_num_wanted = req->GetStepBlockNumber();

    total_token_num += step_token_num_wanted;
    total_block_num += step_block_num_wanted;

    if (total_token_num > batch_schedule_config_.max_token_number || total_block_num > max_free_block_num ||
        running_queue_.size() > batch_schedule_config_.max_running_queue_len) {
      NLLM_LOG_DEBUG << "req " << req->req_id << " swapped out.";
      NLLM_LOG_DEBUG << "Reason: total_token_num:" << total_token_num
                     << ", max_token_number:" << batch_schedule_config_.max_token_number
                     << ", total_block_num:" << total_block_num << ", max_free_block_num:" << max_free_block_num;

      total_token_num -= step_token_num_wanted;
      total_block_num -= step_block_num_wanted;

      total_swapout_num += 1;
      max_free_block_num += req->GetCurrentBlockNumber();

      NLLM_LOG_DEBUG << "Result: total_token_num:" << total_token_num
                     << ", max_token_number:" << batch_schedule_config_.max_token_number
                     << ", total_block_num:" << total_block_num << ", max_free_block_num:" << max_free_block_num;

      req->SwapOutAsync();
      swapped_queue_.push_back(req);
      it = running_queue_.erase(it);
      schedule_step_finish = true;
      continue;
    }

    // Allocate blocks and continue running.
    NLLM_LOG_DEBUG << "req " << req->req_id << " continue running.";
    if (step_block_num_wanted > 0) {
      for (size_t i = 0; i < context_->GetTensorParallelSize(); ++i) {
        std::vector<int> blocks;
        GetBlockManager()->SetDeviceId(i);
        GetBlockManager()->AllocateBlocks(step_block_num_wanted, blocks);
        req->kv_cache_blocks[i].insert(req->kv_cache_blocks[i].end(), blocks.begin(), blocks.end());
      }
    }
    ++it;
  }
}

void BatchScheduler::ScheduleSwapped(size_t &total_token_num, size_t &total_block_num, bool &schedule_step_finish,
                                     size_t &max_free_block_num) {
  for (auto it = swapped_queue_.begin(); it != swapped_queue_.end();) {
    auto req = *it;
    NLLM_LOG_DEBUG << "Try req " << req->req_id << " in swapped_queue_";

    // Check timeout, no finished req in swapped queue.
    if (CheckRequestTimeout(req)) {
      NLLM_LOG_DEBUG << "req " << req->req_id << " timeout in swapped.";

      max_free_block_num += req->GetCurrentBlockNumber();

      // Drop the swapped blocks.
      req->DropSwappedAsync();

      req->finish_status = Status(RET_TIMEOUT, "running timeout.");
      it = swapped_queue_.erase(it);
      req->finished = true;
      req->Notify();
      continue;
    }

    // Stay swapped and step to next.
    if (schedule_step_finish) {
      NLLM_LOG_DEBUG << "req " << req->req_id << " stay swapped.";
      ++it;
      continue;
    }

    // All the bocks must be swapped in for swapped reqs.
    // For swapped req, all blocks should be swapped in, and then allocate new block if necessary.
    size_t step_token_num_wanted = req->GetStepTokenNumber();
    size_t total_block_num_wanted = req->GetTotalBlockNumber();

    total_token_num += step_token_num_wanted;
    total_block_num += total_block_num_wanted;

    if (total_token_num > batch_schedule_config_.max_token_number || total_block_num > max_free_block_num ||
        running_queue_.size() >= batch_schedule_config_.max_running_queue_len) {
      // stay swapped.
      NLLM_LOG_DEBUG << "swapped req " << req->req_id << " stay swapped.";
      NLLM_LOG_DEBUG << "Reason: total_token_num:" << total_token_num
                     << ", max_token_number:" << batch_schedule_config_.max_token_number
                     << ", total_block_num:" << total_block_num << ", max_free_block_num:" << max_free_block_num;

      total_token_num -= step_token_num_wanted;
      total_block_num -= total_block_num_wanted;

      NLLM_LOG_DEBUG << "Result: total_token_num:" << total_token_num
                     << ", max_token_number:" << batch_schedule_config_.max_token_number
                     << ", total_block_num:" << total_block_num << ", max_free_block_num:" << max_free_block_num;

      schedule_step_finish = true;
      ++it;
      continue;
    }

    size_t step_block_num_wanted = req->GetStepBlockNumber();

    NLLM_LOG_DEBUG << "swapped req " << req->req_id << " swap in and ready to run.";
    req->SwapInAsync();
    if (step_block_num_wanted > 0) {
      for (int i = 0; i < context_->GetTensorParallelSize(); ++i) {
        std::vector<int> blocks;
        GetBlockManager()->SetDeviceId(i);
        GetBlockManager()->AllocateBlocks(step_block_num_wanted, blocks);
        req->kv_cache_blocks[i].insert(req->kv_cache_blocks[i].end(), blocks.begin(), blocks.end());
      }
    }

    running_queue_.push_back(req);
    it = swapped_queue_.erase(it);
  }
}

void BatchScheduler::ScheduleWaiting(size_t &total_token_num, size_t &total_block_num, bool &schedule_step_finish,
                                     size_t &max_free_block_num) {
  for (auto it = waiting_queue_.begin(); it != waiting_queue_.end();) {
    auto &req = *it;
    NLLM_LOG_DEBUG << "Try req " << req->req_id << " in waiting_queue_";

    // Check timeout
    if (CheckRequestTimeout(req)) {
      NLLM_LOG_DEBUG << "req " << req->req_id << " timeout in waiting.";

      // No blocks allocated yet.
      req->finish_status = Status(RET_TIMEOUT, "running timeout.");
      it = waiting_queue_.erase(it);
      req->finished = true;
      req->Notify();
      continue;
    }

    // Stay waiting and step to next.
    if (schedule_step_finish) {
      NLLM_LOG_DEBUG << "req " << req->req_id << " stay waiting.";
      ++it;
      continue;
    }

    size_t step_token_num_wanted = req->GetStepTokenNumber();
    size_t total_block_num_wanted = req->GetTotalBlockNumber();

    total_token_num += step_token_num_wanted;
    total_block_num += total_block_num_wanted;

    if (total_token_num > batch_schedule_config_.max_token_number || total_block_num > max_free_block_num ||
        running_queue_.size() >= batch_schedule_config_.max_running_queue_len) {
      // stay waiting.
      NLLM_LOG_DEBUG << "req " << req->req_id << " stay waiting.";
      NLLM_LOG_DEBUG << "Reason: total_token_num:" << total_token_num
                     << ", max_token_number:" << batch_schedule_config_.max_token_number
                     << ", total_block_num:" << total_block_num << ", max_free_block_num:" << max_free_block_num;

      total_token_num -= step_token_num_wanted;
      total_block_num -= total_block_num_wanted;

      NLLM_LOG_DEBUG << "Result: total_token_num:" << total_token_num
                     << ", max_token_number:" << batch_schedule_config_.max_token_number
                     << ", total_block_num:" << total_block_num << ", max_free_block_num:" << max_free_block_num;

      schedule_step_finish = true;
      ++it;
      continue;
    }

    if (total_block_num_wanted > 0) {
      for (size_t i = 0; i < context_->GetTensorParallelSize(); ++i) {
        std::vector<int> blocks;
        GetBlockManager()->SetDeviceId(i);
        GetBlockManager()->AllocateBlocks(total_block_num_wanted, blocks);
        req->kv_cache_blocks[i].insert(req->kv_cache_blocks[i].end(), blocks.begin(), blocks.end());
      }
    }

    NLLM_LOG_DEBUG << "req " << req->req_id << " ready to run.";
    running_queue_.push_back(req);
    it = waiting_queue_.erase(it);
  }
}

std::vector<std::shared_ptr<InferRequest>> &BatchScheduler::Schedule() {
  NLLM_LOG_DEBUG << "Try a scheduler loop.";
  ResetSchedule();

  std::lock_guard<std::mutex> guard(queue_mutex_);

  size_t total_token_num = 0;
  size_t total_block_num = 0;
  size_t max_free_block_num = GetBlockManager()->GetFreeBlockNumber();

  size_t total_swapout_num = 0;
  bool schedule_step_finish = false;
  ScheduleRunning(total_token_num, total_block_num, schedule_step_finish, max_free_block_num, total_swapout_num);

  // Reschedule if reqs swapped, because more blocks is available.
  if (total_swapout_num > 0) {
    schedule_step_finish = false;
  }

  ScheduleSwapped(total_token_num, total_block_num, schedule_step_finish, max_free_block_num);
  ScheduleWaiting(total_token_num, total_block_num, schedule_step_finish, max_free_block_num);

  NLLM_LOG_DEBUG << "batch scheduler result: " << running_queue_.size();
  return running_queue_;
}

}  // namespace ksana_llm
