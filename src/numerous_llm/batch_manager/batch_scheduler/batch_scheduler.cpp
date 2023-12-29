/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/batch_manager/batch_scheduler/batch_scheduler.h"

#include <memory>
#include <utility>
#include <vector>

#include "numerous_llm/block_manager/block_manager.h"
#include "numerous_llm/block_manager/memory_block.h"
#include "numerous_llm/runtime/context.h"
#include "numerous_llm/runtime/infer_request.h"
#include "numerous_llm/utils/channel.h"
#include "numerous_llm/utils/logger.h"
#include "numerous_llm/utils/memory_utils.h"
#include "numerous_llm/utils/ret_code.h"
#include "numerous_llm/utils/singleton.h"
#include "numerous_llm/utils/string_utils.h"

namespace numerous_llm {

BatchScheduler::BatchScheduler(const BatchSchedulerConfig &batch_scheduler_config, std::shared_ptr<Context> context)
    : batch_schedule_config_(batch_scheduler_config), context_(context) {}

BatchScheduler::~BatchScheduler() {}

Status BatchScheduler::AddInferRequest(std::shared_ptr<InferRequest> infer_request) {
  NLLM_LOG_INFO << "batch scheduler add infer request.";
  if (CheckWaitingQueueFull()) {
    infer_request->finish_status = Status(RET_EXCEED_CAPACITY, "waiting queue is full.");

    infer_request->waiter->Notify();
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
                                     size_t max_free_block_num) {
  for (auto it = running_queue_.begin(); it != running_queue_.end();) {
    auto req = *it;
    NLLM_LOG_INFO << "try req " << req->infer_id << " in running_queue_";

    req->ResetInferStage();

    // Check if finished.
    if (CheckRequestFinish(req)) {
      NLLM_LOG_INFO << "req " << req->infer_id << " finished.";
      req->finish_status = Status(RET_SUCCESS);
      it = running_queue_.erase(it);
      req->waiter->Notify();
      continue;
    }

    // Check timeout
    if (CheckRequestTimeout(req)) {
      NLLM_LOG_INFO << "req " << req->infer_id << " timeout in running.";
      req->finish_status = Status(RET_TIMEOUT, "running timeout.");
      it = running_queue_.erase(it);
      req->waiter->Notify();
      continue;
    }

    if (req->infer_stage == InferStage::STAGE_CONTEXT) {
      NLLM_LOG_INFO << "req " << req->infer_id << " change from context decode to decode";
      req->infer_stage = InferStage::STATE_DECODE;
    }

    // Swap left running reqs if schedule step finished.
    if (schedule_step_finish) {
      NLLM_LOG_INFO << "req " << req->infer_id << " swapped out.";
      req->SwapOutAsync();
      swapped_queue_.push_back(req);
      it = running_queue_.erase(it);
      continue;
    }

    // Check total token number and block number.
    total_token_num += req->GetStepTokenNumber();
    size_t block_num_wanted = req->GetStepBlockNumber();
    total_block_num += block_num_wanted;

    if (total_token_num > batch_schedule_config_.max_token_number || total_block_num > max_free_block_num) {
      NLLM_LOG_INFO << "req " << req->infer_id << " swapped out.";
      req->SwapOutAsync();
      swapped_queue_.push_back(req);
      it = running_queue_.erase(it);
      schedule_step_finish = true;
      continue;
    }

    // Allocate blocks and continue running.
    NLLM_LOG_INFO << "req " << req->infer_id << " continue running.";
    if (block_num_wanted > 0) {
      for (size_t i = 0; i < context_->GetTensorParallelSize(); ++i) {
        std::vector<int> blocks;
        GetBlockManager()->SetDeviceId(i);
        GetBlockManager()->AllocateBlocks(block_num_wanted, blocks);
        req->kv_cache_blocks[i].insert(req->kv_cache_blocks[i].end(), blocks.begin(), blocks.end());
      }
    }
    ++it;
  }
}

void BatchScheduler::ScheduleSwapped(size_t &total_token_num, size_t &total_block_num, bool &schedule_step_finish,
                                     size_t max_free_block_num) {
  for (auto it = swapped_queue_.begin(); it != swapped_queue_.end();) {
    auto req = *it;
    NLLM_LOG_INFO << "Try req " << req->infer_id << " in swapped_queue_";

    // Check timeout, no finished req in swapped queue.
    if (CheckRequestTimeout(req)) {
      NLLM_LOG_INFO << "req " << req->infer_id << " timeout in swapped.";

      // Drop the swapped blocks.
      req->DropSwappedAsync();

      req->finish_status = Status(RET_TIMEOUT, "running timeout.");
      it = swapped_queue_.erase(it);
      req->waiter->Notify();
      continue;
    }

    // Stay swapped and step to next.
    if (schedule_step_finish) {
      NLLM_LOG_INFO << "req " << req->infer_id << " stay swapped.";
      ++it;
      continue;
    }

    // All the bocks must be swapped in for swapped reqs.
    // For swapped req, all blocks should be swapped in, and then allocate new block if necessary.
    total_token_num += req->GetStepTokenNumber();
    total_block_num += req->GetTotalBlockNumber();

    if (total_token_num > batch_schedule_config_.max_token_number || total_block_num > max_free_block_num) {
      // stay swapped.
      NLLM_LOG_INFO << "swapped req " << req->infer_id << " stay swapped.";
      schedule_step_finish = true;
      ++it;
      continue;
    }

    size_t block_num_wanted = req->GetStepBlockNumber();

    NLLM_LOG_INFO << "swapped req " << req->infer_id << " swap in and ready to run.";
    req->SwapInAsync();
    if (block_num_wanted > 0) {
      for (size_t i = 0; i < context_->GetTensorParallelSize(); ++i) {
        std::vector<int> blocks;
        GetBlockManager()->SetDeviceId(i);
        GetBlockManager()->AllocateBlocks(block_num_wanted, blocks);
        req->kv_cache_blocks[i].insert(req->kv_cache_blocks[i].end(), blocks.begin(), blocks.end());
      }
    }

    running_queue_.push_back(req);
    it = swapped_queue_.erase(it);
  }
}

void BatchScheduler::ScheduleWaiting(size_t &total_token_num, size_t &total_block_num, bool &schedule_step_finish,
                                     size_t max_free_block_num) {
  for (auto it = waiting_queue_.begin(); it != waiting_queue_.end();) {
    auto &req = *it;
    NLLM_LOG_INFO << "Try req " << req->infer_id << " in waiting_queue_";

    // Check timeout
    if (CheckRequestTimeout(req)) {
      NLLM_LOG_INFO << "req " << req->infer_id << " timeout in waiting.";
      req->finish_status = Status(RET_TIMEOUT, "running timeout.");
      it = waiting_queue_.erase(it);
      req->waiter->Notify();
      continue;
    }

    // Stay waiting and step to next.
    if (schedule_step_finish) {
      NLLM_LOG_INFO << "req " << req->infer_id << " stay waiting.";
      ++it;
      continue;
    }

    total_token_num += req->GetStepTokenNumber();
    size_t block_num_wanted = req->GetTotalBlockNumber();
    total_block_num += block_num_wanted;

    if (total_token_num > batch_schedule_config_.max_token_number || total_block_num > max_free_block_num) {
      // stay waiting.
      NLLM_LOG_INFO << "req " << req->infer_id << " stay waiting.";
      NLLM_LOG_INFO << "Reason: total_token_num:" << total_token_num
                    << ", max_token_number:" << batch_schedule_config_.max_token_number
                    << ", total_block_num:" << total_block_num << ", max_free_block_num:" << max_free_block_num;
      schedule_step_finish = true;
      ++it;
      continue;
    }

    if (block_num_wanted > 0) {
      for (size_t i = 0; i < context_->GetTensorParallelSize(); ++i) {
        std::vector<int> blocks;
        GetBlockManager()->SetDeviceId(i);
        GetBlockManager()->AllocateBlocks(block_num_wanted, blocks);
        req->kv_cache_blocks[i].insert(req->kv_cache_blocks[i].end(), blocks.begin(), blocks.end());
      }
    }

    NLLM_LOG_INFO << "req " << req->infer_id << " ready to run.";
    running_queue_.push_back(req);
    it = waiting_queue_.erase(it);
  }
}

std::vector<std::shared_ptr<InferRequest>> &BatchScheduler::Schedule() {
  NLLM_LOG_INFO << "Try a scheduler loop.";
  ResetSchedule();

  std::lock_guard<std::mutex> guard(queue_mutex_);

  size_t total_token_num = 0;
  size_t total_block_num = 0;
  size_t max_free_block_num = GetBlockManager()->GetFreeBlockNumber();

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
