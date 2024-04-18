/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/batch_manager/batch_scheduler/batch_scheduler.h"

#include <algorithm>
#include <future>
#include <memory>
#include <numeric>
#include <thread>
#include <utility>
#include <vector>

#include "ksana_llm/block_manager/block_manager.h"
#include "ksana_llm/block_manager/memory_block.h"
#include "ksana_llm/profiler/reporter.h"
#include "ksana_llm/runtime/infer_request.h"
#include "ksana_llm/utils/channel.h"
#include "ksana_llm/utils/context.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/memory_utils.h"
#include "ksana_llm/utils/ret_code.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/string_utils.h"

namespace ksana_llm {

BatchScheduler::BatchScheduler(const BatchSchedulerConfig &batch_scheduler_config, std::shared_ptr<Context> context)
    : batch_scheduler_config_(batch_scheduler_config), context_(context) {
  // Config validation.
  NLLM_CHECK_WITH_INFO(batch_scheduler_config_.max_step_tokens > batch_scheduler_config_.max_token_len,
                       FormatStr("The max_step_tokens must large than max_token_len, %d vs %d.",
                                 batch_scheduler_config_.max_step_tokens, batch_scheduler_config_.max_token_len));

  threadpool_ = std::make_shared<ThreadPool>(batch_scheduler_config.swap_threadpool_size);
  threadpool_->Start();

  waiting_queue_.reserve(batch_scheduler_config_.max_waiting_queue_len);
  running_queue_.reserve(batch_scheduler_config_.max_batch_size);
  swapped_queue_.reserve(batch_scheduler_config_.max_batch_size);

  waiting_buffer_queue_.reserve(batch_scheduler_config_.max_waiting_queue_len);
  recompute_queue_.reserve(batch_scheduler_config_.max_waiting_queue_len);
  swapin_pending_queue_.reserve(batch_scheduler_config_.max_batch_size);
  swapout_pending_queue_.reserve(batch_scheduler_config_.max_batch_size);

  running_step_token_num_list_.reserve(batch_scheduler_config_.max_batch_size);
  running_step_block_num_list_.reserve(batch_scheduler_config_.max_batch_size);
  running_curr_block_num_list_.reserve(batch_scheduler_config_.max_batch_size);
  running_swapped_indexes_.reserve(batch_scheduler_config_.max_batch_size);

  swapped_step_token_num_list_.reserve(batch_scheduler_config_.max_batch_size);
  swapped_curr_block_num_list_.reserve(batch_scheduler_config_.max_batch_size);
  swapped_running_indexes_.reserve(batch_scheduler_config_.max_batch_size);

  waiting_step_token_num_list_.reserve(batch_scheduler_config_.max_waiting_queue_len);
  waiting_total_block_num_list_.reserve(batch_scheduler_config_.max_waiting_queue_len);
  waiting_running_indexes_.reserve(batch_scheduler_config_.max_waiting_queue_len);
}

BatchScheduler::~BatchScheduler() { threadpool_->Stop(); }

void BatchScheduler::SwapOutAsync(std::shared_ptr<InferRequest> req) {
  req->swap_pending = true;
  req->swap_future = threadpool_->Submit([=]() {
    NLLM_LOG_DEBUG << "Start to async swapout req " << req->req_id << ", block size:" << req->GetCurrentBlockNumber();
    {
      REPORT_TIME_US(batch_scheduler_swapout_us);
      req->SwapOutAsync();
      req->swap_pending = false;
    }
    NLLM_LOG_DEBUG << "Finish to async swapout req " << req->req_id;
  });
}

void BatchScheduler::SwapInAsync(std::shared_ptr<InferRequest> req) {
  req->swap_pending = true;
  req->swap_future = threadpool_->Submit([=]() {
    NLLM_LOG_DEBUG << "Start to async swapin req " << req->req_id << ", block size:" << req->GetCurrentBlockNumber();
    {
      REPORT_TIME_US(batch_scheduler_swapin_us);
      req->SwapInAsync();
      req->swap_pending = false;
    }
    NLLM_LOG_DEBUG << "Finish to async swapin req " << req->req_id;
  });
}

Status BatchScheduler::AddInferRequest(std::shared_ptr<InferRequest> infer_request) {
  NLLM_LOG_DEBUG << "batch scheduler add infer req " << infer_request->req_id << ", max_new_tokens "
                 << infer_request->sampling_config.max_new_tokens;
  if (CheckWaitingQueueFull()) {
    NLLM_LOG_DEBUG << "waiting queue is full, req " << infer_request->req_id << " failed.";

    infer_request->finish_status = Status(RET_EXCEED_CAPACITY, "waiting queue is full.");
    infer_request->finished = true;
    infer_request->Notify();

    return infer_request->finish_status;
  }

  if (CheckRequestExceedLength(infer_request)) {
    NLLM_LOG_DEBUG << "input len is too long, req " << infer_request->req_id << " failed.";

    infer_request->finish_status = Status(RET_EXCEED_LENGTH, "input length exceed max_token_len.");
    infer_request->finished = true;
    infer_request->Notify();

    return infer_request->finish_status;
  }

  std::lock_guard<std::mutex> guard(queue_buffer_mutex_);
  waiting_buffer_queue_.push_back(infer_request);
  return Status();
}

bool BatchScheduler::WaitingBufferEmpty() {
  std::lock_guard<std::mutex> guard(queue_buffer_mutex_);
  return waiting_buffer_queue_.empty();
}

bool BatchScheduler::SwappedQueueEmtpy() {
  std::lock_guard<std::mutex> guard(queue_mutex_);
  return swapped_queue_.empty();
}

void BatchScheduler::MergeWaitingBufferQueue() {
  std::lock_guard<std::mutex> guard(queue_buffer_mutex_);

  waiting_queue_.insert(waiting_queue_.end(), waiting_buffer_queue_.begin(), waiting_buffer_queue_.end());
  waiting_buffer_queue_.clear();
}

void BatchScheduler::MergeRecomputeQueue() {
  if (recompute_queue_.empty()) {
    return;
  }
  waiting_queue_.insert(waiting_queue_.begin(), recompute_queue_.begin(), recompute_queue_.end());
  recompute_queue_.clear();
}

void BatchScheduler::MergePendingSwapoutRequest() {
  for (auto it = swapout_pending_queue_.begin(); it != swapout_pending_queue_.end();) {
    auto &req = *it;
    if (!req->swap_pending) {
      NLLM_LOG_DEBUG << "Merge swapout req " << req->req_id << " to swapped.";
      swapped_queue_.insert(swapped_queue_.begin(), req);
      it = swapout_pending_queue_.erase(it);
      continue;
    }

    NLLM_LOG_DEBUG << "swapout req " << req->req_id << " is pending, skip merge";
    ++it;
  }
}

void BatchScheduler::MergePendingSwapinRequests() {
  for (auto it = swapin_pending_queue_.begin(); it != swapin_pending_queue_.end();) {
    auto &req = *it;
    if (!req->swap_pending) {
      NLLM_LOG_DEBUG << "Merge swapin req " << req->req_id << " to running.";
      running_queue_.push_back(req);
      it = swapin_pending_queue_.erase(it);
      continue;
    }

    NLLM_LOG_DEBUG << "swapin req " << req->req_id << " is pending, skip merge";
    ++it;
  }
}

void BatchScheduler::WaitPendingSwapoutDone() {
  for (auto it = swapout_pending_queue_.begin(); it != swapout_pending_queue_.end(); ++it) {
    auto &req = *it;
    if (req->swap_pending) {
      NLLM_LOG_DEBUG << "Wait until swapout req " << req->req_id << " done.";
      try {
        req->swap_future.get();
      } catch (const std::exception &e) {
        NLLM_LOG_FATAL << "Exception in swapout, info: " << e.what();
      }
    }
  }
}

void BatchScheduler::WaitPendingSwapinDone() {
  for (auto it = swapin_pending_queue_.begin(); it != swapin_pending_queue_.end(); ++it) {
    auto &req = *it;
    if (req->swap_pending) {
      NLLM_LOG_DEBUG << "Wait until swapin req " << req->req_id << " done.";
      try {
        req->swap_future.get();
      } catch (const std::exception &e) {
        NLLM_LOG_FATAL << "Exception in swapin, info: " << e.what();
      }
    }
  }
}

size_t BatchScheduler::GetSwapinPendingBlockNumber() {
  size_t pending_block_num = 0;
  for (auto it = swapin_pending_queue_.begin(); it != swapin_pending_queue_.end(); ++it) {
    auto &req = *it;
    if (req->swap_pending) {
      pending_block_num += req->GetCurrentBlockNumber();
    }
  }
  return pending_block_num;
}

bool BatchScheduler::CheckRequestTimeout(const std::shared_ptr<InferRequest> req) {
  return schedule_time_in_ms_ >= req->timestamp_in_ms + batch_scheduler_config_.waiting_timeout_in_ms;
}

bool BatchScheduler::CheckWaitingQueueFull() {
  return waiting_queue_.size() >= batch_scheduler_config_.max_waiting_queue_len;
}

inline bool BatchScheduler::CheckRequestExceedLength(const std::shared_ptr<InferRequest> req) {
  return req->input_tokens.size() > batch_scheduler_config_.max_token_len;
}

bool BatchScheduler::CheckRequestFinish(const std::shared_ptr<InferRequest> req) {
  if (req->infer_stage == InferStage::STATE_DECODE) {
    if (req->output_tokens.size() > req->input_tokens.size() &&
        (req->output_tokens.back() == req->model_instance->GetModelConfig().end_id ||
         (req->sampling_config.max_new_tokens > 0 &&
          req->output_tokens.size() >= req->input_tokens.size() + req->sampling_config.max_new_tokens) ||
         req->output_tokens.size() >= batch_scheduler_config_.max_token_len)) {
      return true;
    }
  }
  return false;
}

void BatchScheduler::ResetInfoBeforeSchedule() { schedule_time_in_ms_ = GetCurrentTimeInMs(); }

void BatchScheduler::PrepareRunningRequests(std::vector<size_t> &step_token_num_list,
                                            std::vector<size_t> &step_block_num_list,
                                            std::vector<size_t> &curr_block_num_list) {
  for (auto it = running_queue_.begin(); it != running_queue_.end();) {
    auto &req = *it;
    NLLM_LOG_DEBUG << "prepare req " << req->req_id << " in running_queue_";

    req->AdjustInferStage();

    // Check if finished.
    if (CheckRequestFinish(req)) {
      NLLM_LOG_DEBUG << "req " << req->req_id << " finished.";

      req->finish_status = Status(RET_SUCCESS);
      req->FreeBlocks();
      req->finished = true;
      req->Notify();
      it = running_queue_.erase(it);
      continue;
    }

    // Check timeout
    if (CheckRequestTimeout(req)) {
      NLLM_LOG_DEBUG << "req " << req->req_id << " timeout in running.";

      req->finish_status = Status(RET_TIMEOUT, "running timeout.");
      req->finished = true;
      req->FreeBlocks();
      req->Notify();
      it = running_queue_.erase(it);
      continue;
    }

    // Notify streaming iterator if needed.
    req->NotifyStep();

    step_token_num_list.push_back(req->GetStepTokenNumber());
    step_block_num_list.push_back(req->GetStepBlockNumber());
    curr_block_num_list.push_back(req->GetCurrentBlockNumber());

    ++it;
  }
}

void BatchScheduler::PrepareSwappedRequests(std::vector<size_t> &step_token_num_list,
                                            std::vector<size_t> &curr_block_num_list, bool skip_collect) {
  for (auto it = swapped_queue_.begin(); it != swapped_queue_.end();) {
    auto &req = *it;
    NLLM_LOG_DEBUG << "prepare req " << req->req_id << " in swapped_queue_";

    // Check timeout, no finished req in swapped queue.
    if (CheckRequestTimeout(req)) {
      NLLM_LOG_DEBUG << "req " << req->req_id << " timeout in swapped.";

      req->DropSwappedAsync();

      req->finish_status = Status(RET_TIMEOUT, "running timeout.");
      req->finished = true;
      req->Notify();
      it = swapped_queue_.erase(it);
      continue;
    }

    if (!skip_collect) {
      step_token_num_list.push_back(req->GetStepTokenNumber());
      curr_block_num_list.push_back(req->GetCurrentBlockNumber());
    }
    ++it;
  }
}

void BatchScheduler::PrepareWaitingRequests(std::vector<size_t> &step_token_num_list,
                                            std::vector<size_t> &total_block_num_list, bool skip_collect) {
  for (auto it = waiting_queue_.begin(); it != waiting_queue_.end();) {
    auto &req = *it;
    NLLM_LOG_DEBUG << "prepare req " << req->req_id << " in waiting_queue_";

    // Check timeout
    if (CheckRequestTimeout(req)) {
      NLLM_LOG_DEBUG << "req " << req->req_id << " timeout in waiting.";

      req->finish_status = Status(RET_TIMEOUT, "running timeout.");
      req->finished = true;
      req->Notify();
      it = waiting_queue_.erase(it);
      continue;
    }

    if (!skip_collect) {
      step_token_num_list.push_back(req->GetStepTokenNumber());
      total_block_num_list.push_back(req->GetTotalBlockNumber());
    }
    ++it;
  }
}

void BatchScheduler::ProcessRunningRequests(const std::vector<size_t> &step_token_num_list,
                                            const std::vector<size_t> &step_block_num_list,
                                            const std::vector<size_t> &curr_block_num_list,
                                            std::vector<int> &swapped_indexes, size_t &step_token_num_sum) {
  size_t total_free_block_num = GetBlockManager()->GetDeviceFreeBlockNumber();
  size_t total_pending_block_num = GetSwapinPendingBlockNumber();
  if (total_free_block_num > total_pending_block_num) {
    total_free_block_num -= total_pending_block_num;
  } else {
    WaitPendingSwapinDone();
    total_free_block_num = GetBlockManager()->GetDeviceFreeBlockNumber();
    total_pending_block_num = GetSwapinPendingBlockNumber();
    if (total_free_block_num > total_pending_block_num) {
      total_free_block_num -= total_pending_block_num;
    } else {
      total_free_block_num = 0;
    }
  }

  size_t step_block_num_sum = std::reduce(step_block_num_list.begin(), step_block_num_list.end());

  step_token_num_sum = std::reduce(step_token_num_list.begin(), step_token_num_list.end());

  size_t step_batch_size = running_queue_.size();
  size_t swapout_block_threshold = step_batch_size * batch_scheduler_config_.swapout_block_threshold;

  // Swapout the last requests.
  swapped_indexes.clear();
  while (step_token_num_sum > batch_scheduler_config_.max_step_tokens ||
         step_batch_size > batch_scheduler_config_.max_batch_size ||
         step_block_num_sum + swapout_block_threshold > total_free_block_num) {
    --step_batch_size;
    swapout_block_threshold = step_batch_size * batch_scheduler_config_.swapout_block_threshold;

    step_token_num_sum -= step_token_num_list[step_batch_size];
    step_block_num_sum -= step_block_num_list[step_batch_size];
    total_free_block_num += curr_block_num_list[step_batch_size];

    swapped_indexes.push_back(step_batch_size);
  }

  // Make if [N - 1, N, N + 1] order.
  std::reverse(swapped_indexes.begin(), swapped_indexes.end());
}

void BatchScheduler::ProcessSwappedRequests(const std::vector<size_t> &step_token_num_list,
                                            const std::vector<size_t> &curr_block_num_list,
                                            std::vector<int> &running_indexes, size_t &step_token_num_sum,
                                            size_t &curr_block_num_sum) {
  size_t total_free_block_num = GetBlockManager()->GetDeviceFreeBlockNumber();
  size_t total_pending_block_num = GetSwapinPendingBlockNumber();
  if (total_free_block_num > total_pending_block_num) {
    total_free_block_num -= total_pending_block_num;
  } else {
    WaitPendingSwapinDone();
    total_free_block_num = GetBlockManager()->GetDeviceFreeBlockNumber();
    total_pending_block_num = GetSwapinPendingBlockNumber();
    if (total_free_block_num > total_pending_block_num) {
      total_free_block_num -= total_pending_block_num;
    } else {
      total_free_block_num = 0;
    }
  }

  size_t step_batch_size = running_queue_.size();
  size_t swapin_block_threshold = step_batch_size * batch_scheduler_config_.swapin_block_threshold;

  running_indexes.clear();
  for (size_t i = 0; i < swapped_queue_.size(); ++i) {
    ++step_batch_size;
    swapin_block_threshold = step_batch_size * batch_scheduler_config_.swapin_block_threshold;

    step_token_num_sum += step_token_num_list[i];
    curr_block_num_sum += curr_block_num_list[i];

    if (step_token_num_sum < batch_scheduler_config_.max_step_tokens &&
        step_batch_size < batch_scheduler_config_.max_batch_size &&
        curr_block_num_sum + swapin_block_threshold < total_free_block_num) {
      running_indexes.push_back(i);
      continue;
    }

    // Stop swapin.
    step_token_num_sum -= step_token_num_list[i];
    curr_block_num_sum -= curr_block_num_list[i];
    break;
  }
}

void BatchScheduler::ProcessWaitingRequests(const std::vector<size_t> &step_token_num_list,
                                            const std::vector<size_t> &total_block_num_list,
                                            std::vector<int> &running_indexes, size_t &step_token_num_sum,
                                            size_t &total_block_num_sum) {
  size_t total_free_block_num = GetBlockManager()->GetDeviceFreeBlockNumber();
  size_t total_pending_block_num = GetSwapinPendingBlockNumber();
  if (total_free_block_num > total_pending_block_num) {
    total_free_block_num -= total_pending_block_num;
  } else {
    WaitPendingSwapinDone();
    total_free_block_num = GetBlockManager()->GetDeviceFreeBlockNumber();
    total_pending_block_num = GetSwapinPendingBlockNumber();
    if (total_free_block_num > total_pending_block_num) {
      total_free_block_num -= total_pending_block_num;
    } else {
      total_free_block_num = 0;
    }
  }

  size_t step_batch_size = running_queue_.size() + swapin_pending_queue_.size();
  size_t launch_block_threshold = step_batch_size * batch_scheduler_config_.launch_block_threshold;

  running_indexes.clear();
  for (size_t i = 0; i < waiting_queue_.size(); ++i) {
    ++step_batch_size;
    launch_block_threshold = step_batch_size * batch_scheduler_config_.launch_block_threshold;
    step_token_num_sum += step_token_num_list[i];
    total_block_num_sum += total_block_num_list[i];

    if (step_token_num_sum < batch_scheduler_config_.max_step_tokens &&
        step_batch_size < batch_scheduler_config_.max_batch_size &&
        total_block_num_sum + launch_block_threshold < total_free_block_num) {
      running_indexes.push_back(i);
      continue;
    }

    // Stay waiting.
    step_token_num_sum -= step_token_num_list[i];
    total_block_num_sum -= total_block_num_list[i];
    break;
  }
}

void BatchScheduler::ScheduleRunning(size_t &step_token_num_sum, bool &skip_other) {
  if (batch_scheduler_config_.preempt_mode == SWAP) {
    MergePendingSwapinRequests();
  }
  if (running_queue_.empty()) {
    NLLM_LOG_DEBUG << "Empty running queue after MergePendingSwapinRequests, skip.";
    return;
  }

  // Check timeout and finish status.
  running_step_token_num_list_.clear();
  running_step_block_num_list_.clear();
  running_curr_block_num_list_.clear();
  PrepareRunningRequests(running_step_token_num_list_, running_step_block_num_list_, running_curr_block_num_list_);
  if (running_queue_.empty()) {
    NLLM_LOG_DEBUG << "Empty running queue after PrepareRunningRequests, skip.";
    return;
  }

  int retry_times = 0;
  size_t step_token_num_sum_tmp;
  do {
    running_swapped_indexes_.clear();
    step_token_num_sum_tmp = step_token_num_sum;
    ProcessRunningRequests(running_step_token_num_list_, running_step_block_num_list_, running_curr_block_num_list_,
                           running_swapped_indexes_, step_token_num_sum_tmp);
    NLLM_LOG_DEBUG << "Process running requests, swapped size:" << running_swapped_indexes_.size();
    if (!running_queue_.empty() && running_queue_.size() == running_swapped_indexes_.size()) {
      if (!swapout_pending_queue_.empty()) {
        REPORT_TIME_US(batch_scheduler_wait_swapout_us);
        WaitPendingSwapoutDone();
        ++retry_times;
      }
    }
  } while (retry_times == 1);
  step_token_num_sum = step_token_num_sum_tmp;
  skip_other = !running_swapped_indexes_.empty();

  size_t visit_idx = 0;
  constexpr size_t MAX_SIZE_T = std::numeric_limits<size_t>::max();
  size_t swapout_pos = running_swapped_indexes_.empty() ? MAX_SIZE_T : running_swapped_indexes_.front();
  size_t host_free_num = GetBlockManager()->GetHostFreeBlockNumber();
  for (size_t idx = 0; idx < running_queue_.size();) {
    auto &req = running_queue_[idx];
    if (visit_idx < swapout_pos) {
      NLLM_LOG_DEBUG << "running req " << req->req_id << " continue running.";
      size_t step_block_num = running_step_block_num_list_[visit_idx];
      if (step_block_num > 0) {
        for (size_t i = 0; i < context_->GetTensorParallelSize(); ++i) {
          std::vector<int> blocks;
          GetBlockManager()->SetDeviceId(i);
          Status status = GetBlockManager()->AllocateBlocks(step_block_num, blocks);
          if (!status.OK()) {
            if (!swapout_pending_queue_.empty()) {
              NLLM_LOG_DEBUG << "No more blocks, waiting util all swapout reqs done.";
              WaitPendingSwapoutDone();
              status = GetBlockManager()->AllocateBlocks(step_block_num, blocks);
              if (!status.OK()) {
                NLLM_LOG_ERROR << "No more blocks after all swapout reqs done, exit.";
                abort();
              }
            } else {
              NLLM_LOG_ERROR << "No more blocks, and no pending swapout reqs, exit.";
              abort();
            }
          }
          req->kv_cache_blocks[i].insert(req->kv_cache_blocks[i].end(), blocks.begin(), blocks.end());
        }
      }
      ++idx;
    } else {
      bool swap_success = false;
      if (batch_scheduler_config_.preempt_mode == SWAP) {
        NLLM_LOG_DEBUG << "running req " << req->req_id << " swapout async.";
        size_t swap_size = req->kv_cache_blocks[0].size();
        swap_success = host_free_num > swap_size;
        if (swap_success) {
          host_free_num -= swap_size;
          SwapOutAsync(req);
          swapout_pending_queue_.insert(swapout_pending_queue_.begin(), req);
          running_queue_.erase(running_queue_.begin() + idx);
        }
      }

      if (!swap_success || batch_scheduler_config_.preempt_mode == RECOMPUTE) {
        NLLM_LOG_DEBUG << "running req " << req->req_id << " recompute.";
        req->FreeBlocks();
        req->kv_cache_blocks.resize(context_->GetTensorParallelSize());
        req->infer_stage = InferStage::STAGE_CONTEXT;
        req->step = 0;
        recompute_queue_.push_back(req);
        running_queue_.erase(running_queue_.begin() + idx);
      }
    }
    ++visit_idx;
  }
}

void BatchScheduler::ScheduleSwapped(size_t &step_token_num_sum, size_t &curr_block_num_sum, bool &skip_other) {
  MergePendingSwapoutRequest();
  if (swapped_queue_.empty()) {
    NLLM_LOG_DEBUG << "Empty swapped queue after merge, skip.";
    return;
  }

  swapped_step_token_num_list_.clear();
  swapped_curr_block_num_list_.clear();
  PrepareSwappedRequests(swapped_step_token_num_list_, swapped_curr_block_num_list_, skip_other);
  if (skip_other || swapped_queue_.empty()) {
    NLLM_LOG_DEBUG << "Empty swapped queue after prepare, skip.";
    return;
  }

  int retry_times = 0;
  size_t step_token_num_sum_tmp;
  size_t curr_block_num_sum_tmp;
  do {
    swapped_running_indexes_.clear();
    step_token_num_sum_tmp = step_token_num_sum;
    curr_block_num_sum_tmp = curr_block_num_sum;
    ProcessSwappedRequests(swapped_step_token_num_list_, swapped_curr_block_num_list_, swapped_running_indexes_,
                           step_token_num_sum_tmp, curr_block_num_sum_tmp);
    NLLM_LOG_DEBUG << "Process swapped requests, running size:" << swapped_running_indexes_.size();
    if (running_queue_.empty() && !swapped_queue_.empty() && swapped_running_indexes_.empty()) {
      if (!swapout_pending_queue_.empty()) {
        REPORT_TIME_US(batch_scheduler_wait_swapout_us);
        WaitPendingSwapoutDone();
        ++retry_times;
      }
    }
  } while (retry_times == 1);
  step_token_num_sum = step_token_num_sum_tmp;
  curr_block_num_sum = curr_block_num_sum_tmp;

  size_t visit_idx = 0;
  size_t swapin_pos = swapped_running_indexes_.empty() ? 0 : swapped_running_indexes_.back() + 1;
  for (size_t idx = 0; idx < swapped_queue_.size();) {
    auto &req = swapped_queue_[idx];
    if (visit_idx < swapin_pos) {
      NLLM_LOG_DEBUG << "swapped req " << req->req_id << " swapin async.";
      SwapInAsync(req);
      swapin_pending_queue_.push_back(req);
      swapped_queue_.erase(swapped_queue_.begin() + idx);
      ++visit_idx;
      continue;
    }
    break;
  }

  // Skip launch waiting if swapped list is not empty.
  skip_other = !swapped_queue_.empty();
}

void BatchScheduler::ScheduleWaiting(size_t &step_token_num_sum, size_t &curr_block_num_sum, bool &skip_other) {
  MergeRecomputeQueue();

  MergeWaitingBufferQueue();
  if (waiting_queue_.empty()) {
    NLLM_LOG_DEBUG << "Empty waiting queue after merge, skip.";
    return;
  }

  waiting_step_token_num_list_.clear();
  waiting_total_block_num_list_.clear();
  PrepareWaitingRequests(waiting_step_token_num_list_, waiting_total_block_num_list_, skip_other);
  if (skip_other || waiting_queue_.empty()) {
    NLLM_LOG_DEBUG << "Empty waiting queue after prepare, skip.";
    return;
  }

  int retry_times = 0;
  size_t step_token_num_sum_tmp;
  size_t curr_block_num_sum_tmp;
  do {
    waiting_running_indexes_.clear();
    step_token_num_sum_tmp = step_token_num_sum;
    curr_block_num_sum_tmp = curr_block_num_sum;
    ProcessWaitingRequests(waiting_step_token_num_list_, waiting_total_block_num_list_, waiting_running_indexes_,
                           step_token_num_sum_tmp, curr_block_num_sum_tmp);
    NLLM_LOG_DEBUG << "Process waiting requests, running size:" << waiting_running_indexes_.size();
    if (running_queue_.empty() && !waiting_queue_.empty() && waiting_running_indexes_.empty()) {
      if (!swapout_pending_queue_.empty()) {
        REPORT_TIME_US(batch_scheduler_wait_swapout_us);
        WaitPendingSwapoutDone();
        ++retry_times;
      }
    }
  } while (retry_times == 1);
  step_token_num_sum = step_token_num_sum_tmp;
  curr_block_num_sum = curr_block_num_sum_tmp;

  size_t visit_idx = 0;
  size_t launch_pos = waiting_running_indexes_.empty() ? 0 : waiting_running_indexes_.back() + 1;
  for (size_t idx = 0; idx < waiting_queue_.size();) {
    auto &req = waiting_queue_[idx];
    if (visit_idx < launch_pos) {
      NLLM_LOG_DEBUG << "waiting req " << req->req_id << " launch.";
      size_t total_block_num = waiting_total_block_num_list_[visit_idx];
      if (total_block_num > 0) {
        for (int i = 0; i < context_->GetTensorParallelSize(); ++i) {
          std::vector<int> blocks;
          GetBlockManager()->SetDeviceId(i);
          GetBlockManager()->AllocateBlocks(total_block_num, blocks);
          req->kv_cache_blocks[i].insert(req->kv_cache_blocks[i].end(), blocks.begin(), blocks.end());
        }
      }

      running_queue_.push_back(req);
      waiting_queue_.erase(waiting_queue_.begin() + idx);
      ++visit_idx;
      continue;
    }
    break;
  }
}

void BatchScheduler::SchedulePending() {
  // Process pending only if running queue is empty.
  if (running_queue_.empty()) {
    if (!swapin_pending_queue_.empty()) {
      {
        REPORT_TIME_US(batch_scheduler_wait_swapin_us);
        WaitPendingSwapinDone();
      }
      MergePendingSwapinRequests();
      if (!swapin_pending_queue_.empty()) {
        NLLM_CHECK_WITH_INFO(false, "Wait and merge pending swapin error.");
      }
    }

    // Waiting for swapout finished, then batch manager will invoke the next schedule step.
    if (running_queue_.empty()) {
      if (!swapout_pending_queue_.empty()) {
        {
          REPORT_TIME_US(batch_scheduler_wait_swapout_us);
          WaitPendingSwapoutDone();
        }
        MergePendingSwapoutRequest();
        if (!swapout_pending_queue_.empty()) {
          NLLM_CHECK_WITH_INFO(false, "Wait and merge pending swapout error.");
        }
      }
    }
  }
}

std::vector<std::shared_ptr<InferRequest>> &BatchScheduler::Schedule() {
  NLLM_LOG_DEBUG << "Try scheduler loop.";
  std::lock_guard<std::mutex> guard(queue_mutex_);
  ResetInfoBeforeSchedule();

  bool skip_other = false;
  size_t step_token_num_sum = 0;
  size_t curr_block_num_sum = 0;

  ScheduleRunning(step_token_num_sum, skip_other);
  if (batch_scheduler_config_.preempt_mode == SWAP) {
    ScheduleSwapped(step_token_num_sum, curr_block_num_sum, skip_other);
  }
  ScheduleWaiting(step_token_num_sum, curr_block_num_sum, skip_other);
  SchedulePending();

  REPORT_METRIC(batch_scheduler_running, running_queue_.size());
  REPORT_METRIC(batch_scheduler_waiting, waiting_queue_.size());
  REPORT_METRIC(batch_scheduler_swapped, swapped_queue_.size());

  REPORT_METRIC(batch_scheduler_pending_swapin, swapin_pending_queue_.size());
  REPORT_METRIC(batch_scheduler_pending_swapout, swapout_pending_queue_.size());

  REPORT_METRIC(block_manager_free, GetBlockManager()->GetDeviceFreeBlockNumber());
  REPORT_METRIC(block_manager_used, GetBlockManager()->GetDeviceUsedBlockNumber());

  NLLM_LOG_DEBUG << "batch scheduler result: " << running_queue_.size();
  return running_queue_;
}

}  // namespace ksana_llm
