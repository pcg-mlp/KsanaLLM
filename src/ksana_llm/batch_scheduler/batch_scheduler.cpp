/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/batch_scheduler/batch_scheduler.h"

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
#include "ksana_llm/profiler/trace_event_recorder.h"
#include "ksana_llm/runtime/infer_request.h"
#include "ksana_llm/utils/channel.h"
#include "ksana_llm/utils/context.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/memory_utils.h"
#include "ksana_llm/utils/ret_code.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/string_utils.h"
#include "ksana_llm/utils/tokenizer.h"

namespace ksana_llm {

BatchScheduler::BatchScheduler(const BatchSchedulerConfig& batch_scheduler_config, int tp_num)
    : batch_scheduler_config_(batch_scheduler_config) {
  // Config validation.
  KLLM_CHECK_WITH_INFO(batch_scheduler_config_.max_step_tokens >= batch_scheduler_config_.max_token_len,
                       FormatStr("The max_step_tokens must larger or equal than max_token_len, %d vs %d.",
                                 batch_scheduler_config_.max_step_tokens, batch_scheduler_config_.max_token_len));

  batch_state_ = std::make_shared<BatchState>(batch_scheduler_config_);
  schedule_strategy_ = ScheduleStrategyFactory::CreateScheduleStrategy(batch_scheduler_config_, tp_num, batch_state_);
}

void BatchScheduler::SetCacheManager(std::shared_ptr<CacheManagerInterface> cache_manager) {
  schedule_strategy_->SetCacheManager(cache_manager);
}

void BatchScheduler::SetTokenizer(std::shared_ptr<Tokenizer> tokenizer) { schedule_strategy_->SetTokenizer(tokenizer); }

std::shared_ptr<CacheManagerInterface>& BatchScheduler::GetCacheManager() {
  return schedule_strategy_->GetCacheManager();
}

Status BatchScheduler::AddInferRequest(std::vector<std::shared_ptr<InferRequest>>& infer_request_group) {
  std::shared_ptr<InferRequest>& infer_request = infer_request_group[0];
  KLLM_LOG_DEBUG << "batch scheduler add infer req " << infer_request->req_id << ", max_new_tokens "
                 << infer_request->sampling_config.max_new_tokens;

  if (CheckRequestExceedLength(infer_request)) {
    KLLM_LOG_DEBUG << "input len or logits_custom_length is too long, req " << infer_request->req_id << " failed.";

    auto finish_status = Status(RET_EXCEED_LENGTH, "input length or logits_custom_length exceeds the limit.");
    infer_request->finish_status = finish_status;
    for (auto& infer_request : infer_request_group) {
      infer_request->finished = true;

      RECORD_TRACE_EVENT_TAG("Input2Long", TraceEventType::DropReq, std::to_string(infer_request->req_id),
                             TRACE_THREAD_NAME_PREFILL_DECODE);
    }
    infer_request->Notify();
    return finish_status;
  }

  return EnqueueWaitingBufferQueue(infer_request_group);
}

bool BatchScheduler::IsIdle() {
  bool waiting_buffer_emtpy = false;
  {
    std::lock_guard<std::mutex> guard(batch_state_->queue_buffer_mutex);
    waiting_buffer_emtpy = batch_state_->waiting_buffer_queue.empty();
  }

  bool batch_state_queue_empty = false;
  {
    std::lock_guard<std::mutex> guard(batch_state_->queue_mutex);
    batch_state_queue_empty = batch_state_->swapped_queue.empty() && batch_state_->waiting_queue.empty();
  }

  return (waiting_buffer_emtpy && batch_state_queue_empty);
}

Status BatchScheduler::EnqueueWaitingBufferQueue(std::vector<std::shared_ptr<InferRequest>>& infer_request_group) {
  std::lock_guard<std::mutex> guard(batch_state_->queue_buffer_mutex);

  if (batch_state_->waiting_buffer_queue.size() + infer_request_group.size() >
      batch_scheduler_config_.max_waiting_queue_len) {
    std::shared_ptr<InferRequest>& infer_request = infer_request_group[0];
    KLLM_LOG_DEBUG << "waiting queue is full, req " << infer_request->req_id << " failed.";

    auto finish_status = Status(RET_EXCEED_CAPACITY, "waiting queue is full.");
    infer_request->finish_status = finish_status;
    for (auto& infer_request : infer_request_group) {
      infer_request->finished = true;
      RECORD_TRACE_EVENT_TAG("WaitingQFull", TraceEventType::DropReq, std::to_string(infer_request->req_id),
                             TRACE_THREAD_NAME_PREFILL_DECODE);
    }
    infer_request->Notify();
    return finish_status;
  }

  for (const auto& infer_request : infer_request_group) {
    batch_state_->waiting_buffer_queue.push_back(infer_request);
  }
  return Status();
}

inline bool BatchScheduler::CheckRequestExceedLength(const std::shared_ptr<InferRequest> req) {
  return req->input_tokens.size() > batch_scheduler_config_.max_token_len ||
         req->logits_custom_length > std::min(req->input_tokens.size(), batch_scheduler_config_.max_batch_size);
}

std::vector<std::shared_ptr<InferRequest>>& BatchScheduler::Schedule() {
  KLLM_LOG_DEBUG << "Try scheduler loop.";
  std::lock_guard<std::mutex> guard(batch_state_->queue_mutex);

  schedule_strategy_->Schedule();

  size_t batch_size = batch_state_->running_queue.size();
  REPORT_METRIC(batch_scheduler_batch_size, batch_size);
  REPORT_METRIC(batch_scheduler_waiting_size, batch_state_->waiting_queue.size());
  REPORT_METRIC(batch_scheduler_swapped_size, batch_state_->swapped_queue.size());

  if (batch_size > 0) {
    size_t token_num = 0;
    const auto current_time = ProfileTimer::GetCurrentTimeInMs();
    for (const auto& req : batch_state_->running_queue) {
      token_num += req->output_tokens.size();
      if (req->kv_cached_token_num == 0) {
        REPORT_METRIC(batch_manager_schedule_ms, current_time - req->timestamp_in_ms);
      }
      REPORT_METRIC(req_total_cost_in_queue_ms, current_time - req->timestamp_in_ms);
    }
    REPORT_METRIC(token_num_in_batch, token_num);

    // token_fill_ratio represents token number per block compared to BlockTokenNum
    // It is always less than 1. If kv caches are shared, it may be greater than 1.
    if (GetBlockManager()->GetDeviceUsedBlockNumber() > 0) {
      REPORT_METRIC(
          token_fill_ratio,
          token_num * 1.0 / (GetBlockManager()->GetDeviceUsedBlockNumber() * GetBlockManager()->GetBlockTokenNum()));
    }
  }
  REPORT_METRIC(block_num_free, GetBlockManager()->GetDeviceFreeBlockNumber());
  REPORT_METRIC(block_num_used, GetBlockManager()->GetDeviceUsedBlockNumber());

  KLLM_LOG_DEBUG << "batch scheduler result: " << batch_state_->running_queue.size();
  return batch_state_->running_queue;
}

}  // namespace ksana_llm
