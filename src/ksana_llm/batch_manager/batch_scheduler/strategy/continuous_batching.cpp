/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/batch_manager/batch_scheduler/strategy/continuous_batching.h"

#include "base_strategy.h"
#include "ksana_llm/profiler/reporter.h"

namespace ksana_llm {

ContinuousBatchingStrategy::ContinuousBatchingStrategy(const BatchSchedulerConfig &batch_scheduler_config,
                                                       int tp_num,
                                                       std::shared_ptr<BatchState> batch_state)
    : BaseScheduleStrategy(batch_scheduler_config, tp_num, batch_state) {
  threadpool_ = std::make_shared<ThreadPool>(batch_scheduler_config.swap_threadpool_size);
  threadpool_->Start();

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

ContinuousBatchingStrategy::~ContinuousBatchingStrategy() { threadpool_->Stop(); }

void ContinuousBatchingStrategy::SwapOutAsync(std::shared_ptr<InferRequest> req, const int host_block_num_to_add) {
  req->swap_pending = true;
  req->swap_future = threadpool_->Submit([=]() {
    NLLM_LOG_DEBUG << "Start to async swapout req " << req->req_id << ", block size:" << req->GetCurrentBlockNumber();
    {
      REPORT_TIME_US(batch_scheduler_swapout_us);
      req->SwapOutAsync(host_block_num_to_add);
      req->swap_pending = false;
    }
    NLLM_LOG_DEBUG << "Finish to async swapout req " << req->req_id;
  });
}

void ContinuousBatchingStrategy::SwapInAsync(std::shared_ptr<InferRequest> req) {
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

bool ContinuousBatchingStrategy::CheckRequestTimeout(const std::shared_ptr<InferRequest> req) {
  return batch_state_->schedule_time_in_ms >= req->timestamp_in_ms + batch_scheduler_config_.waiting_timeout_in_ms;
}

bool ContinuousBatchingStrategy::CheckBeamSearchRequestFinish(const std::shared_ptr<InferRequest> req) {
  bool req_finish = CheckRequestFinish(req);
  if (req->sampling_config.num_beams > 1) {
    if (req_finish) {
      // Add a minimum value to avoid continuing calculations on completed requests.
      req->cumulative_score += (std::numeric_limits<float>::max() / -2.0);
    }
    for (auto &beam_req : req->req_group) {
      if (!CheckRequestFinish(beam_req)) return false;
    }
  }
  return req_finish;
}

bool ContinuousBatchingStrategy::CheckRequestFinish(const std::shared_ptr<InferRequest> req) {
  if (req->infer_stage == InferStage::STATE_DECODE) {
    std::vector<int> &stop_token_ids = req->sampling_config.stop_token_ids;
    if (req->output_tokens.size() > req->input_tokens.size() &&
        (req->output_tokens.back() == req->end_id ||
         std::find(stop_token_ids.begin(), stop_token_ids.end(), req->output_tokens.back()) != stop_token_ids.end() ||
         (req->sampling_config.max_new_tokens > 0 &&
          req->output_tokens.size() >= req->input_tokens.size() + req->sampling_config.max_new_tokens) ||
         req->output_tokens.size() >= batch_scheduler_config_.max_token_len)) {
      return true;
    }
  }
  return false;
}

void ContinuousBatchingStrategy::PrepareRunningRequests(std::vector<size_t> &step_token_num_list,
                                                        std::vector<size_t> &step_block_num_list,
                                                        std::vector<size_t> &curr_block_num_list) {
  for (auto it = batch_state_->running_queue.begin(); it != batch_state_->running_queue.end();) {
    auto &req = *it;
    NLLM_LOG_DEBUG << "prepare req " << req->req_id << " in running_queue_";

    req->AdjustInferStage();

    // Check if finished.
    if (CheckBeamSearchRequestFinish(req)) {
      NLLM_LOG_DEBUG << "req " << req->req_id << " finished.";

      req->finish_status = Status(RET_SUCCESS);
      req->FreeBlocks();
      req->finished = true;
      req->Notify();
      it = batch_state_->running_queue.erase(it);
      continue;
    }

    // Check timeout
    if (CheckRequestTimeout(req)) {
      NLLM_LOG_DEBUG << "req " << req->req_id << " timeout in running.";

      req->finish_status = Status(RET_TIMEOUT, "running timeout.");
      req->finished = true;
      req->FreeBlocks();
      req->Notify();
      it = batch_state_->running_queue.erase(it);
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

void ContinuousBatchingStrategy::PrepareSwappedRequests(std::vector<size_t> &step_token_num_list,
                                                        std::vector<size_t> &curr_block_num_list, bool skip_collect) {
  for (auto it = batch_state_->swapped_queue.begin(); it != batch_state_->swapped_queue.end();) {
    auto &req = *it;
    NLLM_LOG_DEBUG << "prepare req " << req->req_id << " in swapped_queue_";

    // Check timeout, no finished req in swapped queue.
    if (CheckRequestTimeout(req)) {
      NLLM_LOG_DEBUG << "req " << req->req_id << " timeout in swapped.";

      req->DropSwappedAsync();

      req->finish_status = Status(RET_TIMEOUT, "running timeout.");
      req->finished = true;
      req->Notify();
      it = batch_state_->swapped_queue.erase(it);
      continue;
    }

    if (!skip_collect) {
      step_token_num_list.push_back(req->GetStepTokenNumber());
      curr_block_num_list.push_back(req->GetCurrentBlockNumber());
    }
    ++it;
  }
}

void ContinuousBatchingStrategy::PrepareWaitingRequests(std::vector<size_t> &step_token_num_list,
                                                        std::vector<size_t> &total_block_num_list, bool skip_collect) {
  for (auto it = batch_state_->waiting_queue.begin(); it != batch_state_->waiting_queue.end();) {
    auto &req = *it;
    NLLM_LOG_DEBUG << "prepare req " << req->req_id << " in waiting_queue_";

    // Check timeout
    if (CheckRequestTimeout(req)) {
      NLLM_LOG_DEBUG << "req " << req->req_id << " timeout in waiting.";

      req->finish_status = Status(RET_TIMEOUT, "running timeout.");
      req->finished = true;
      req->Notify();
      it = batch_state_->waiting_queue.erase(it);
      continue;
    }

    if (!skip_collect) {
      step_token_num_list.push_back(req->GetStepTokenNumber());
      total_block_num_list.push_back(req->GetTotalBlockNumber());
    }
    ++it;
  }
}

void ContinuousBatchingStrategy::MergePendingSwapoutRequest() {
  for (auto it = swapout_pending_queue_.begin(); it != swapout_pending_queue_.end();) {
    auto &req = *it;
    if (!req->swap_pending) {
      NLLM_LOG_DEBUG << "Merge swapout req " << req->req_id << " to swapped.";
      batch_state_->swapped_queue.insert(batch_state_->swapped_queue.begin(), req);
      it = swapout_pending_queue_.erase(it);
      continue;
    }

    NLLM_LOG_DEBUG << "swapout req " << req->req_id << " is pending, skip merge";
    ++it;
  }
}

void ContinuousBatchingStrategy::MergePendingSwapinRequests() {
  for (auto it = swapin_pending_queue_.begin(); it != swapin_pending_queue_.end();) {
    auto &req = *it;
    if (!req->swap_pending) {
      NLLM_LOG_DEBUG << "Merge swapin req " << req->req_id << " to running.";
      if (CheckBeamSearch(req)) batch_state_->running_queue.push_back(req);
      it = swapin_pending_queue_.erase(it);
      continue;
    }

    NLLM_LOG_DEBUG << "swapin req " << req->req_id << " is pending, skip merge";
    ++it;
  }
}

void ContinuousBatchingStrategy::MergeRecomputeQueue() {
  if (recompute_queue_.empty()) {
    return;
  }
  batch_state_->waiting_queue.insert(batch_state_->waiting_queue.begin(), recompute_queue_.begin(),
                                     recompute_queue_.end());
  recompute_queue_.clear();
}

void ContinuousBatchingStrategy::WaitPendingSwapoutDone() {
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

void ContinuousBatchingStrategy::WaitPendingSwapinDone() {
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

size_t ContinuousBatchingStrategy::GetSwapinPendingBlockNumber() {
  size_t pending_block_num = 0;
  for (auto it = swapin_pending_queue_.begin(); it != swapin_pending_queue_.end(); ++it) {
    auto &req = *it;
    if (req->swap_pending) {
      pending_block_num += req->GetCurrentBlockNumber();
    }
  }
  return pending_block_num;
}

void ContinuousBatchingStrategy::ProcessRunningRequests(const std::vector<size_t> &step_token_num_list,
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

  size_t step_batch_size = batch_state_->running_queue.size();
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

void ContinuousBatchingStrategy::ProcessSwappedRequests(const std::vector<size_t> &step_token_num_list,
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

  size_t step_batch_size = batch_state_->running_queue.size();
  size_t swapin_block_threshold = step_batch_size * batch_scheduler_config_.swapin_block_threshold;

  running_indexes.clear();
  for (size_t i = 0; i < batch_state_->swapped_queue.size(); ++i) {
    ++step_batch_size;
    swapin_block_threshold = step_batch_size * batch_scheduler_config_.swapin_block_threshold;

    step_token_num_sum += step_token_num_list[i];
    curr_block_num_sum += curr_block_num_list[i];

    if (step_token_num_sum < batch_scheduler_config_.max_step_tokens &&
        step_batch_size <= batch_scheduler_config_.max_batch_size &&
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

void ContinuousBatchingStrategy::ProcessWaitingRequests(const std::vector<size_t> &step_token_num_list,
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

  size_t step_batch_size = batch_state_->running_queue.size() + swapin_pending_queue_.size();
  size_t launch_block_threshold = step_batch_size * batch_scheduler_config_.launch_block_threshold;

  running_indexes.clear();
  for (size_t i = 0; i < batch_state_->waiting_queue.size(); ++i) {
    // When the prompt_probs_offset is greater than 0, the size of logits to be calculated is prompt_probs_offset.
    if (batch_state_->waiting_queue[i]->prompt_probs_offset > 0) {
      step_batch_size += batch_state_->waiting_queue[i]->prompt_probs_offset;
    } else {
      ++step_batch_size;
    }
    launch_block_threshold = step_batch_size * batch_scheduler_config_.launch_block_threshold;
    step_token_num_sum += step_token_num_list[i];
    total_block_num_sum += total_block_num_list[i];

    if (step_token_num_sum < batch_scheduler_config_.max_step_tokens &&
        step_batch_size <= batch_scheduler_config_.max_batch_size &&
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

bool ContinuousBatchingStrategy::CheckBeamSearch(std::shared_ptr<InferRequest> req) {
  if (req->sampling_config.num_beams > 1) {
    for (auto &beam_req : req->req_group) {
      if (req->output_tokens.size() > beam_req->output_tokens.size()) {
        NLLM_LOG_DEBUG << "CheckBeamSearch false";
        return false;
      }
    }
  }
  NLLM_LOG_DEBUG << "CheckBeamSearch true";
  return true;
}

void ContinuousBatchingStrategy::ScheduleRunning(size_t &step_token_num_sum, bool &skip_other) {
  if (batch_scheduler_config_.preempt_mode == SWAP) {
    MergePendingSwapinRequests();
  }
  if (batch_state_->running_queue.empty()) {
    NLLM_LOG_DEBUG << "Empty running queue after MergePendingSwapinRequests, skip.";
    return;
  }

  // Check timeout and finish status.
  running_step_token_num_list_.clear();
  running_step_block_num_list_.clear();
  running_curr_block_num_list_.clear();
  PrepareRunningRequests(running_step_token_num_list_, running_step_block_num_list_, running_curr_block_num_list_);
  if (batch_state_->running_queue.empty()) {
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
    if (!batch_state_->running_queue.empty() && batch_state_->running_queue.size() == running_swapped_indexes_.size()) {
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
  for (size_t idx = 0; idx < batch_state_->running_queue.size();) {
    auto &req = batch_state_->running_queue[idx];
    // Allocate blocks according to actual needs to avoid duplicate allocation of hosts and devices.
    size_t step_block_num = req->GetTotalBlockNumber() - req->GetCurrentBlockNumber();
    if (visit_idx < swapout_pos && CheckBeamSearch(req)) {
      NLLM_LOG_DEBUG << "running req " << req->req_id << " continue running.";
      if (step_block_num > 0) {
        for (int i = 0; i < tp_num_; ++i) {
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
        size_t swap_size = req->kv_cache_blocks[0].size() + step_block_num;
        swap_success = host_free_num > swap_size;
        if (swap_success) {
          host_free_num -= swap_size;
          // Assuming that a new block needs to be allocated when the request is going to infer the next token,
          // but the request is SwapOut and happens to be SwapIn in the SchedulePending.
          // In this case, No new blocks will be allocated before the next token infers,
          // causing the block access to be out of bounds during inference.
          // Therefore, in SwapOut, host blocks are allocated in advance, and during SwapIn,
          // they are automatically swapped with device blocks to ensure sufficient block
          // for requests after SwapIn in SchedulePending to infer next token.
          SwapOutAsync(req, step_block_num);
          swapout_pending_queue_.insert(swapout_pending_queue_.begin(), req);
          batch_state_->running_queue.erase(batch_state_->running_queue.begin() + idx);
        }
      }

      if (!swap_success || batch_scheduler_config_.preempt_mode == RECOMPUTE) {
        NLLM_LOG_DEBUG << "running req " << req->req_id << " recompute.";
        req->FreeBlocks();
        req->kv_cache_blocks.resize(tp_num_);
        req->infer_stage = InferStage::STAGE_CONTEXT;
        req->step = 0;
        recompute_queue_.push_back(req);
        batch_state_->running_queue.erase(batch_state_->running_queue.begin() + idx);
      }
    }
    ++visit_idx;
  }
}

void ContinuousBatchingStrategy::ScheduleSwapped(size_t &step_token_num_sum, size_t &curr_block_num_sum,
                                                 bool &skip_other) {
  MergePendingSwapoutRequest();
  if (batch_state_->swapped_queue.empty()) {
    NLLM_LOG_DEBUG << "Empty swapped queue after merge, skip.";
    return;
  }

  swapped_step_token_num_list_.clear();
  swapped_curr_block_num_list_.clear();
  PrepareSwappedRequests(swapped_step_token_num_list_, swapped_curr_block_num_list_, skip_other);
  if (skip_other || batch_state_->swapped_queue.empty()) {
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
    if (batch_state_->running_queue.empty() && !batch_state_->swapped_queue.empty() &&
        swapped_running_indexes_.empty()) {
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
  for (size_t idx = 0; idx < batch_state_->swapped_queue.size();) {
    auto &req = batch_state_->swapped_queue[idx];
    if (visit_idx < swapin_pos) {
      NLLM_LOG_DEBUG << "swapped req " << req->req_id << " swapin async.";
      SwapInAsync(req);
      swapin_pending_queue_.push_back(req);
      batch_state_->swapped_queue.erase(batch_state_->swapped_queue.begin() + idx);
      ++visit_idx;
      continue;
    }
    break;
  }

  // Skip launch waiting if swapped list is not empty.
  skip_other = !batch_state_->swapped_queue.empty();
}

void ContinuousBatchingStrategy::ScheduleWaiting(size_t &step_token_num_sum, size_t &curr_block_num_sum,
                                                 bool &skip_other) {
  MergeRecomputeQueue();

  if (batch_state_->waiting_queue.empty()) {
    NLLM_LOG_DEBUG << "Empty waiting queue after merge, skip.";
    return;
  }

  waiting_step_token_num_list_.clear();
  waiting_total_block_num_list_.clear();
  PrepareWaitingRequests(waiting_step_token_num_list_, waiting_total_block_num_list_, skip_other);
  if (skip_other || batch_state_->waiting_queue.empty()) {
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
    if (batch_state_->running_queue.empty() && !batch_state_->waiting_queue.empty() &&
        waiting_running_indexes_.empty()) {
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
  for (size_t idx = 0; idx < batch_state_->waiting_queue.size();) {
    auto &req = batch_state_->waiting_queue[idx];
    if (!CheckBeamSearch(req)) continue;
    if (visit_idx < launch_pos) {
      NLLM_LOG_DEBUG << "waiting req " << req->req_id << " launch.";
      size_t total_block_num = waiting_total_block_num_list_[visit_idx];
      if (total_block_num > 0) {
        for (int i = 0; i < tp_num_; ++i) {
          std::vector<int> blocks;
          GetBlockManager()->SetDeviceId(i);
          GetBlockManager()->AllocateBlocks(total_block_num, blocks);
          req->kv_cache_blocks[i].insert(req->kv_cache_blocks[i].end(), blocks.begin(), blocks.end());
        }
      }

      batch_state_->running_queue.push_back(req);
      batch_state_->waiting_queue.erase(batch_state_->waiting_queue.begin() + idx);
      ++visit_idx;
      continue;
    }
    break;
  }
}

void ContinuousBatchingStrategy::SchedulePending() {
  // Process pending only if running queue is empty.
  if (batch_state_->running_queue.empty()) {
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
    if (batch_state_->running_queue.empty()) {
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

void ContinuousBatchingStrategy::Schedule() {
  batch_state_->ResetInfoBeforeSchedule();
  batch_state_->MergeWaitingBufferQueue();

  bool skip_other = false;
  size_t step_token_num_sum = 0;
  size_t curr_block_num_sum = 0;

  ScheduleRunning(step_token_num_sum, skip_other);
  if (batch_scheduler_config_.preempt_mode == SWAP) {
    ScheduleSwapped(step_token_num_sum, curr_block_num_sum, skip_other);
  }
  ScheduleWaiting(step_token_num_sum, curr_block_num_sum, skip_other);
  SchedulePending();

  REPORT_METRIC(batch_scheduler_pending_swapin, swapin_pending_queue_.size());
  REPORT_METRIC(batch_scheduler_pending_swapout, swapout_pending_queue_.size());
}

}  // namespace ksana_llm
