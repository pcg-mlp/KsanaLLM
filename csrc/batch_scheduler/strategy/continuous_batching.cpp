/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "batch_scheduler/strategy/continuous_batching.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <utility>

#include "fmt/format.h"

#include "base_strategy.h"
#include "batch_scheduler/global_cache_put_tracker.h"
#include "batch_scheduler/planning_workload_estimator.h"
#include "batch_scheduler/request_finish_checker.h"
#include "batch_scheduler/state/scheduler_tick_tok.h"
#include "cache_manager/prefix_cache_manager.h"
#include "profiler/reporter.h"
#include "runtime/infer_request.h"
#include "transfer/transfer_engine.h"
#include "utils/logger.h"
#include "utils/memory_utils.h"
#include "utils/ret_code.h"
#include "utils/singleton.h"
#include "utils/status.h"
#include "utils/stop_checker.h"
#include "utils/tokenizer.h"

namespace ksana_llm {

extern size_t GetDecodeTokenNumThreshold();

ContinuousBatchingStrategy::ContinuousBatchingStrategy(const BatchSchedulerConfig &batch_scheduler_config,
                                                       const RuntimeConfig &runtime_config)
    : BaseScheduleStrategy(batch_scheduler_config, runtime_config) {
  const auto env = Singleton<Environment>::GetInstance();
  env->GetConnectorConfigs(connector_config_);
  if (connector_config_.group_role != GroupRole::NONE) {
    if (batch_scheduler_config_.split_fuse_token_num > 0) {
      KLLM_LOG_INFO << "Disable split fuse in prefill-decoding separating node";
      batch_scheduler_config_.split_fuse_token_num = 0;
    }
  }

  dp_max_step_token_num_ = batch_scheduler_config_.max_step_token_num;
  dp_max_batch_size_ = batch_scheduler_config_.max_batch_size;
  per_dp_max_batch_size_ = dp_max_batch_size_;

  // When DeepEP low_latency is enabled, limit each DP to max_batch_size / dp requests.
  // dp_max_batch_size_ stays as the global limit (checked against the shared counter across all DPs).
  // per_dp_max_batch_size_ is the per-DP local limit (checked against this DP's own request count).
  if (runtime_config_.moe_all_to_all_mode == MoeAllToAllMode::DEEPEP_LOW_LATENCY) {
    per_dp_max_batch_size_ =
        batch_scheduler_config_.max_batch_size / runtime_config_.parallel_basic_config.attn_data_parallel_size;
    KLLM_LOG_INFO << fmt::format("DeepEP low_latency: per_dp_max_batch_size_ = {} (max_batch_size={}, dp={})",
                                 per_dp_max_batch_size_, batch_scheduler_config_.max_batch_size,
                                 runtime_config_.parallel_basic_config.attn_data_parallel_size);
  }

  dp_max_logits_num_ = dp_max_batch_size_ * batch_scheduler_config.max_decode_tokens_per_req;
  if (connector_config_.group_role == GroupRole::DECODE) {
    dp_max_decode_batch_size_ = dp_max_batch_size_;
    // 增加预参数的大小
    dp_max_batch_size_ = batch_scheduler_config_.max_batch_size + (batch_scheduler_config_.max_pretransfer_batch_size);
    KLLM_LOG_INFO << "decode dp_max_batch_size_:" << dp_max_batch_size_
                  << ", dp_max_decode_batch_size_:" << dp_max_decode_batch_size_;
  }
}

size_t ContinuousBatchingStrategy::GetMaxRequiredTokenNum(const size_t token_num) const {
  return token_num + runtime_config_.mtp_step_num - std::min(runtime_config_.mtp_step_num, 1ul);
}

bool ContinuousBatchingStrategy::CheckRequestTimeout(const std::shared_ptr<InferRequest> req) {
  return batch_state_->schedule_time_in_ms >=
         req->timestamp_in_us / 1000 + batch_scheduler_config_.waiting_timeout_in_ms;
}

Status ContinuousBatchingStrategy::RecomputeMockRequest(std::shared_ptr<InferRequest> &req) {
  KLLM_LOG_SCHEDULER << "RecomputeMockRequest " << req;

  // Add request to the beginning of waiting queue.
  RuntimeConfig runtime_config;
  Singleton<Environment>::GetInstance()->GetRuntimeConfig(runtime_config);
  req->kv_cache_blocks.clear();

  // To avoid Mock requests being categorized as SingleTokenForward requests, we
  // calculate the Mock request total length as: Mock total length = MTP token
  // count + SingleToken length + 1 (additional token)
  const size_t mock_request_length = runtime_config.mtp_step_num + 1 + 1;
  // After Mock request completes one inference round, rollback the newly generated tokens at the end to restore
  // the initial state.
  if (req->output_tokens.size() > mock_request_length) {
    req->output_tokens.resize(mock_request_length);
  }

  req->ResetPrefillingTokens();
  if (req->GetPlanningSequenceLen() > mock_request_length) {
    req->forwarding_tokens.resize(mock_request_length);
  }

  // Reset finished flag to allow MockRequest to be scheduled again
  req->finished = false;
  req->finish_status = Status();
  req->Restart();  // Restart the MockRequest so it can be scheduled again.

  batch_state_->mock_queue.push_back(req);

  // Record recomputed req_id
  if (req->attn_dp_group_id >= batch_state_->schedule_output->recompute_req_ids.size()) {
    size_t needed_push_size = req->attn_dp_group_id - batch_state_->schedule_output->recompute_req_ids.size() + 1;
    for (size_t idx = 0; idx < needed_push_size; ++idx) {
      batch_state_->schedule_output->recompute_req_ids.push_back(std::vector<int64_t>{});
    }
  }
  batch_state_->schedule_output->recompute_req_ids[req->attn_dp_group_id].push_back(req->req_id);

  return Status();
}

bool ContinuousBatchingStrategy::CheckRequestFinish(const std::shared_ptr<InferRequest> req) {
#ifdef CLEAR_CACHE
  // For testing purposes only
  // When a specific input is received, free all cached blocks.
  // TODO(robertyuan): Remove this to right place
  if (req->input_tokens.size() == 2 && req->input_tokens[0] == 0 && req->input_tokens[1] == 0) {
    size_t free_block_num;
    auto prefix_cache_manager_ptr = dynamic_cast<PrefixCacheManager *>(cache_manager_.get());
    if (prefix_cache_manager_ptr != nullptr) {
      prefix_cache_manager_ptr->FreeCachedBlocks(1e8, free_block_num);
      KLLM_LOG_WARNING << "cache_manager free " << free_block_num << " blocks.";
    }
  }
#endif
  if (IsRequestFinished(req, batch_scheduler_config_)) {
    KLLM_LOG_SCHEDULER << "Request " << req->req_id << " had finished."
                       << " req output_tokens size: " << req->output_tokens.size()
                       << " input_tokens size: " << req->input_tokens.size();
    return true;
  }
  if (HasRemainingPrefillWorkload(req)) {
    return false;
  }
  // TODO(robertyuan): User defined checking. Move this to PostGenHook
  // When stop strings are checked and matched, stop early
  return stop_checker_->CheckIncrementalStopStrings(req);
}
// TODO(robertyuan): Move to PostScheduleHook
void ContinuousBatchingStrategy::DetermineDraftNum(std::shared_ptr<InferRequest> req) {
  // Determine the number of draft_tokens to generate in the current step based on the scheduling status.
  req->suggested_draft_num = 0;
  constexpr size_t kDraftBatchSizeThreshold = 16;
  const size_t running_bs = batch_state_->decoding_queue.size();
  if (running_bs >= kDraftBatchSizeThreshold) {
    return;
  }
  const size_t draft_num_per_req = (kDraftBatchSizeThreshold - running_bs) / running_bs;
  req->suggested_draft_num = draft_num_per_req;
}

void ContinuousBatchingStrategy::RecomputeRequest(std::shared_ptr<InferRequest> req) {
  auto it = batch_state_->async_recomputed_reqs.find(req->req_id);
  if (it != batch_state_->async_recomputed_reqs.end()) {
    // planned to be recomputed in async, skip
    return;
  }

  REPORT_COUNTER("recompute_request_num", 1);
  if (!req->HasInflightTask()) {
    // No task is running, recompute immediately
    SyncRecomputeRequest(req);
  } else {
    KLLM_LOG_SCHEDULER << "Put req_id=" << req->req_id << " to async recompute queue";
    batch_state_->async_recomputed_reqs[req->req_id] = req;
  }
}

void ContinuousBatchingStrategy::SyncRecomputeRequest(std::shared_ptr<InferRequest> req) {
  static size_t total_recomputed_token_num = 0;
  total_recomputed_token_num += req->output_tokens.size();
  KLLM_LOG_INFO << "SyncRecomputeRequest req id is: " << req->req_id
                << ", intput_tokens.size()=" << req->input_tokens.size()
                << ", output_tokens.size()=" << req->output_tokens.size()
                << ", total_recomputed_token_num=" << total_recomputed_token_num;
  REPORT_COUNTER("recomputed_request_num", 1);

  // Rollback structured generator state if exists
  if (req->structured_generator != nullptr) {
    size_t rollback_token_num = req->output_tokens.size();
    KLLM_LOG_DEBUG << "Rolling back structured generator for req_id=" << req->req_id << " by " << rollback_token_num
                   << " tokens";
    req->structured_generator->Rollback(rollback_token_num);
  }

  // Add request to the beginning of waiting queue.
  req->kv_cache_blocks.clear();
  req->accepted_tokens.clear();
  req->generated_tokens.clear();
  req->draft_tokens.clear();
  req->forwarding_tokens.clear();
  req->ResetPrefillingTokens();

  // Record recomputed req_id
  if (req->attn_dp_group_id >= batch_state_->schedule_output->recompute_req_ids.size()) {
    size_t needed_push_size = req->attn_dp_group_id - batch_state_->schedule_output->recompute_req_ids.size() + 1;
    for (size_t idx = 0; idx < needed_push_size; ++idx) {
      batch_state_->schedule_output->recompute_req_ids.push_back(std::vector<int64_t>{});
    }
  }
  batch_state_->schedule_output->recompute_req_ids[req->attn_dp_group_id].push_back(req->req_id);

  if (connector_config_.group_role != GroupRole::NONE) {
    KLLM_LOG_INFO << "Request " << req->req_id
                  << " is recomputed due to exceeding max_step_token_num or max_batch_size in decode group.";
    Status status(RET_PREDICTOR_DISCARD, "Disaggregation of prefill and decoding could not be recomputed.");
    ResetRequest(req, status, true);
    return;
  }

  static constexpr bool terminate = false;
  ResetRequest(req, Status(RET_SUCCESS, "RecomputeRequest"), terminate);

  batch_state_->waiting_queue.insert(batch_state_->waiting_queue.begin(), req);
}

void ContinuousBatchingStrategy::StopRequest(std::shared_ptr<InferRequest> req, Status ret_status,
                                             RequestState req_state) {
  KLLM_LOG_SCHEDULER << "StopRequest req id is: " << req->req_id;
  if (req->HasInflightTask()) {
    batch_state_->async_stoped_reqs[req->req_id] = {req, ret_status, req_state};
  } else {
    SyncStopRequest(req, ret_status, req_state);
  }
}

void ContinuousBatchingStrategy::SyncStopRequest(std::shared_ptr<InferRequest> req, Status ret_status,
                                                 RequestState req_state) {
  StampOnce(req->finish_time_us);
  ResetRequest(req, ret_status, true);
  // Record finish req_id
  if (req_state == RequestState::kRunning) {
    if (req->attn_dp_group_id >= batch_state_->schedule_output->finish_req_ids.size()) {
      size_t needed_push_size = req->attn_dp_group_id - batch_state_->schedule_output->finish_req_ids.size() + 1;
      for (size_t idx = 0; idx < needed_push_size; ++idx) {
        batch_state_->schedule_output->finish_req_ids.push_back(std::vector<int64_t>{});
      }
    }
    batch_state_->schedule_output->finish_req_ids[req->attn_dp_group_id].push_back(req->req_id);
  }
  KLLM_LOG_SCHEDULER << "Stopped req_id=" << req->req_id << ", finished=" << req->finished
                     << ", ret_status=" << ret_status.GetCode() << ", batch_state=" << *batch_state_;
}

bool ContinuousBatchingStrategy::ProcessAsyncStoppedRequest(std::shared_ptr<InferRequest> &req) {
  auto it = batch_state_->async_stoped_reqs.find(req->req_id);
  if (it == batch_state_->async_stoped_reqs.end()) {
    return false;
  }
  // build-ahead（max_depth=2）下 async stop 入队时可能仍有 inflight step，须等 drain 后再 SyncStop。
  if (req->HasInflightTask()) {
    KLLM_LOG_SCHEDULER << "Async stop deferred: req_id=" << req->req_id
                       << " still has inflight (count=" << req->GetInflightCount() << ")";
    return true;
  }
  const auto &stop_info = it->second;

  KLLM_LOG_SCHEDULER << "Processing async stop request " << req->req_id;
  SyncStopRequest(req, stop_info.ret_status, stop_info.req_state);
  batch_state_->async_stoped_reqs.erase(it);
  return true;
}

void ContinuousBatchingStrategy::RecoverAsyncRecomputedRequests() {
  for (auto &it : batch_state_->async_recomputed_reqs) {
    const auto &req = it.second;
    if (req->IsStopped()) {
      continue;
    }

    ScheduleTaskWorkload remaining_workload = req->GetRemainingWorkload();
    if (remaining_workload.prefill_token_num > 0) {
      batch_state_->waiting_queue.push_back(req);
    } else {
      batch_state_->decoding_queue.push_back(req);
    }
  }
  batch_state_->async_recomputed_reqs.clear();
}

bool ContinuousBatchingStrategy::ProcessAsyncRecomputeRequest(const std::shared_ptr<InferRequest> &req) {
  auto it = batch_state_->async_recomputed_reqs.find(req->req_id);
  if (it == batch_state_->async_recomputed_reqs.end()) {
    return false;
  }

  // build-ahead（max_depth=2）下 preempt recompute 入队时可能仍有 inflight step，须等 drain 后再 SyncRecompute。
  if (req->HasInflightTask()) {
    KLLM_LOG_SCHEDULER << "Async recompute deferred: req_id=" << req->req_id
                       << " still has inflight (count=" << req->GetInflightCount() << ")";
    return true;
  }

  if (!req->IsStopped()) {
    KLLM_LOG_SCHEDULER << "Processing async recompute request " << req->req_id;
    SyncRecomputeRequest(req);
  }

  batch_state_->async_recomputed_reqs.erase(it);
  return true;
}

// In asynchronous mode, inflighting requests may be in other queues.
void ContinuousBatchingStrategy::RemoveRequestFromBatchState(const std::shared_ptr<InferRequest> &req) {
  if (RemoveRequestFromQueue(batch_state_->schedule_output->running_reqs, req)) {
    // TODO(robertyuan): Found in running queue, adjust planning schedule output if needed.
  }
  if (RemoveRequestFromQueue(batch_state_->decoding_queue, req) ||
      RemoveRequestFromQueue(batch_state_->waiting_queue, req)) {
    return;
  }
  auto it = batch_state_->async_recomputed_reqs.find(req->req_id);
  if (it != batch_state_->async_recomputed_reqs.end()) {
    batch_state_->async_recomputed_reqs.erase(it);
    return;
  }

  KLLM_LOG_FATAL << "Request " << req->req_id << " not found in batch state. " << batch_state_->ToString(true);
  // TODO(robertyuan): request maybe moved to another batch state.
}

void ContinuousBatchingStrategy::ResetRequest(std::shared_ptr<InferRequest> req, Status ret_status, bool terminate) {
  KLLM_LOG_SCHEDULER << "ResetRequest " << req->ScheduleStateToStr() << ", ret_status:" << ret_status.ToString()
                     << ", terminate:" << terminate;

  req->finish_status = ret_status;
  req->finished = terminate;
  if (terminate) {
    req->Stop();
  }

  // 若有 pending put，延迟调用 DestroyFinishedRequest，等待所有设备 put 完成后再释放 block
  if (!global_cache_put_tracker_ || !global_cache_put_tracker_->DeferDestructionIfNeeded(req->req_id)) {
    cache_manager_->DestroyFinishedRequest(req->req_id);
  }

  if (terminate) {
    req->Notify();
  }
}

std::pair<size_t, size_t> ContinuousBatchingStrategy::CheckRunningQueueStepTokens(
    const std::vector<std::shared_ptr<InferRequest>> &checking_reqs,
    std::vector<std::shared_ptr<InferRequest>> &passed_reqs) {
  // step_token_num: Controls the maximum total tokens in a batch (max_step_token_num), related to buffer allocation and
  // GPU memory management
  // step_not_kv_cached_token_num: Total count of tokens without KV caching, directly affecting computational workload
  // requirements
  passed_reqs.reserve(checking_reqs.size());

  scheduler_ticktok_->Barrier();

  if (dp_group_id_ == 0) {
    scheduler_ticktok_->Reset();
    scheduler_shared_counter_->step_batch_size.Reset(0);
    scheduler_shared_counter_->step_token_num.Reset(0);
    scheduler_shared_counter_->step_logits_num.Reset(0);
  }

  scheduler_ticktok_->Barrier();

  size_t step_token_num = 0, step_not_kv_cached_token_num = 0;
  size_t total_sampling_token_num = 0, req_num = 0;

  // count how many tokens can be scheduled in this step
  // the req_token_num is the snapshot when the request is added to running queue.
  // it may be smaller than the actual token number when the request is running.
  size_t local_step_token_num = 0;
  for (auto it = checking_reqs.begin(); it != checking_reqs.end();) {
    const auto &req = *it;
    // TODO(robertyuan): replace with step_token_num. They should be same in all case, check later
    const size_t not_kv_cached_token_num = req->GetPlanningQueryLen();
    size_t req_token_num = runtime_config_.attn_backend_config.enable_blocked_multi_token_forwarding_kv
                               ? not_kv_cached_token_num
                               : req->GetPlanningSequenceLen();
    if (batch_scheduler_config_.ptp_step_num > 0) {
      // When using Parallel-Token-Predict, we used not_kv_cached_token_num to raise max_batch_size
      req_token_num = not_kv_cached_token_num;
    }
    KLLM_LOG_DEBUG << fmt::format("Request {} req_token_num: {}, not_kv_cached_token_num: {}", req->req_id,
                                  req_token_num, not_kv_cached_token_num);

    scheduler_ticktok_->Lock();

    // The total num include other dp groups.
    req_num = scheduler_shared_counter_->step_batch_size.Get();
    step_token_num = scheduler_shared_counter_->step_token_num.Get();
    total_sampling_token_num = scheduler_shared_counter_->step_logits_num.Get();

    if (step_token_num + req_token_num > dp_max_step_token_num_ ||
        total_sampling_token_num + req->sampling_token_num > dp_max_logits_num_ || req_num >= dp_max_batch_size_ ||
        passed_reqs.size() >= per_dp_max_batch_size_) {
      scheduler_ticktok_->Unlock();
      ++it;
      continue;
    }
    step_not_kv_cached_token_num += not_kv_cached_token_num;

    scheduler_shared_counter_->step_batch_size.Increase(1);
    scheduler_shared_counter_->step_token_num.Increase(req_token_num);
    scheduler_shared_counter_->step_logits_num.Increase(req->sampling_token_num);
    local_step_token_num += req_token_num;
    passed_reqs.push_back(req);
    scheduler_ticktok_->Unlock();

    ++it;
  }

  // Current dp group finished, remove from loop list.
  scheduler_ticktok_->Skip();

  return {local_step_token_num, step_not_kv_cached_token_num};
}

void ContinuousBatchingStrategy::UpdateRunningRequests(const std::vector<std::shared_ptr<InferRequest>> &running_reqs) {
  KLLM_LOG_SCHEDULER << "update running requests size:" << running_reqs.size();

  bool has_block_freed = false;
  for (auto req : running_reqs) {
    // All req here should be decode now.
    if (req->IsStopped()) {
      continue;
    }
    req->UpdateAfterInflightTaskFinished();
    req->ResetInflightTask();
    // clear flexible cache copy tasks after context stage is finished
    req->flexible_cached_copy_tasks.clear();

    ReportRequestProgressInfo(req);

    if (ProcessAsyncStoppedRequest(req)) {
      has_block_freed = true;
      continue;
    }

    req->req_ctx->emplace("status_code", std::to_string(static_cast<int>(req->finish_status.GetCode())));

    // Always update cache manager, even if request is finished.
    bool block_merged = false;
    const Status status = cache_manager_->UpdateRequestTokens(
        req->req_id, req->forwarding_tokens, req->kv_cached_token_num, req->kv_cache_blocks, block_merged,
        req->swa_kv_cache_blocks, req->swa_kv_cache_block_idx_offsets);

    if (!status.OK()) {
      KLLM_LOG_SCHEDULER << "UpdateRequestTokens " << req << " error, recompute it, info: " << status.GetMessage();
      RemoveRequestFromBatchState(req);
      RecomputeRequest(req);
      has_block_freed = true;
      continue;
    }

    // TODO(zakwang): PD support StructuredOutput
    // If prefill finished, enter transfer queue
    if (connector_config_.group_role == GroupRole::PREFILL && !req->is_mock_req) {
      req->NotifyStep();
      continue;
    }

    // Check if finished.
    // ProcessPrefillTransferQueue also checks if the request is finished.
    if (CheckRequestFinish(req)) {
      const auto end_time = ProfileTimer::GetCurrentTimeInUs();
      const size_t output_token_num = req->output_tokens.size() - req->input_tokens.size();
      const uint64_t duration = end_time - req->timestamp_in_us;
      if (req->finish_status.GetCode() == RET_SUCCESS && output_token_num > 0) {
        REPORT_METRIC("total_latency_us", duration);
        REPORT_METRIC("output_token_len", output_token_num);
        REPORT_METRIC("input_token_len", req->input_tokens.size());
        // Access-log latency metrics. Timestamps share the ProfileTimer monotonic clock with
        // timestamp_in_us (arrive); 0 means the hook never stamped it, so skip to avoid bogus deltas.
        const uint64_t arrive = req->timestamp_in_us;
        const uint64_t scheduled = req->scheduled_time_us.load();
        const uint64_t first_token = req->first_token_time_us.load();
        const uint64_t finish = req->finish_time_us.load();
        if (scheduled >= arrive && arrive > 0) {
          REPORT_METRIC("waiting_latency_us", scheduled - arrive);
        }
        if (first_token >= arrive && arrive > 0) {
          REPORT_METRIC("ttft_us", first_token - arrive);
        }
        if (output_token_num > 1 && first_token > 0 && finish >= first_token) {
          REPORT_METRIC("tpot_us", (finish - first_token) / (output_token_num - 1));
        }
      } else {
        REPORT_METRIC("forward_req_error_num", req->finish_status.GetCode());
      }

      StopRequest(req, Status(RET_SUCCESS), RequestState::kRunning);

      // Put mock request back to mock_queue.
      if (req->is_mock_req) {
        RecomputeMockRequest(req);
      }
      RemoveRequestFromBatchState(req);
      has_block_freed = true;
      continue;
    }

    // Check timeout
    if (CheckRequestTimeout(req)) {
      REPORT_COUNTER("forward_req_timeout_num", static_cast<size_t>(1));
      KLLM_LOG_ERROR << "req timeout in running:" << req;

      StopRequest(req, Status(RET_REQUEST_TIMEOUT, "timeout in running."), RequestState::kRunning);
      RemoveRequestFromBatchState(req);
      has_block_freed = true;
      continue;
    }

    // Check abort.
    if (req->aborted) {
      KLLM_LOG_WARNING << "req aborted in running: " << req->req_id;

      StopRequest(req, Status(RET_REQUEST_TERMINATED, "req aborted in running."), RequestState::kRunning);
      RemoveRequestFromBatchState(req);
      REPORT_COUNTER("forward_req_aborted_num", static_cast<size_t>(1));
      has_block_freed = true;
      continue;
    }

    // Not finished, notify streaming iterator.
    req->NotifyStep();
  }

  if (batch_state_->async_recomputed_reqs.size() > 0) {
    if (has_block_freed) {
      // Some blocks freed, do not recompute async requests.
      RecoverAsyncRecomputedRequests();
    } else {
      for (const auto &req : running_reqs) {
        ProcessAsyncRecomputeRequest(req);
      }
    }
  }
  KLLM_LOG_SCHEDULER << "After updating, batch_state_=" << *batch_state_;
}

void ContinuousBatchingStrategy::ProcessDecodingQueue() {
  PROFILE_EVENT_SCOPE(ProcessDecodingQueue, "ProcessDecodingQueue");
  KLLM_LOG_SCHEDULER << "ProcessDecodingQueue invoked: " << *batch_state_
                     << ", total_free_block_num:" << cache_manager_->GetUsableBlockNumber();

  // Estimate inflight task result and update planning workloads
  for (auto &req : batch_state_->decoding_queue) {
    EstimateRequestPlanningWorkload(req, batch_scheduler_config_.mtp_step_num + batch_scheduler_config_.ptp_step_num);
  }

  // Check decoding queue to determine whether it exceeds the max_step_token_num.
  std::vector<std::shared_ptr<InferRequest>> passed_reqs;
  CheckRunningQueueStepTokens(batch_state_->decoding_queue, passed_reqs);

  if (passed_reqs.size() == 0) {
    return;
  }

  size_t alloc_failed_req_num = 0;
  for (auto it = passed_reqs.begin(); it != passed_reqs.end();) {
    const auto &req = *it;
    KLLM_CHECK(req->GetPlanningWorkload().prefill_token_num == 0);  // Decoding requests only
    const size_t max_required_token_num = GetMaxRequiredTokenNum(req->GetPlanningSequenceLen());
    const size_t req_step_block_num = cache_manager_->GetRequestStepBlockNumber(req->req_id, max_required_token_num);

    Status status =
        cache_manager_->AllocateRequestBlocks(req->req_id, req_step_block_num, req->kv_cache_blocks,
                                              req->swa_kv_cache_blocks, req->swa_kv_cache_block_idx_offsets);
    if (status.OK()) {
      DetermineDraftNum(req);
      batch_state_->schedule_output->running_reqs.push_back(req);
      ++it;
      continue;
    }

    // No more blocks, skip waiting_req launch.
    batch_state_->step_sched_finish = true;

    // decoding request can not run due to block limitation, try to free some blocks
    size_t recomputing_waiting_task_num = 0;
    if (!batch_state_->waiting_queue.empty() && batch_scheduler_config_.split_fuse_token_num > 0) {
      std::vector<std::shared_ptr<InferRequest>> recomputing_reqs;
      for (auto &waiting_req : batch_state_->waiting_queue) {
        if (!waiting_req->kv_cache_blocks.empty()) {
          recomputing_reqs.push_back(waiting_req);
        }
      }

      recomputing_waiting_task_num = recomputing_reqs.size();
      for (const auto &req : recomputing_reqs) {
        RemoveRequestFromQueue(batch_state_->waiting_queue, req);
        RecomputeRequest(req);
      }
    }

    if (recomputing_waiting_task_num > 0) {
      // Try to allocate again
      continue;
    }

    // If waiting queue cannot release any blocks and no asyn recopmute/stop tasks, try to recompute some decoding
    // tasks;
    // TODO(robertyuan): optimize choice to balance wasted computing and decoding in future
    if ((recomputing_waiting_task_num == 0) && (batch_state_->async_stoped_reqs.size() == 0) &&
        (batch_state_->async_recomputed_reqs.size() == 0) && (batch_state_->decoding_queue.size() > 1)) {
      RecomputeRequest(req);
      RemoveRequestFromQueue(batch_state_->decoding_queue, req);
      ++it;
      continue;
    }

    KLLM_LOG_SCHEDULER << "Allocate blocks error, info: " << status.GetMessage();
    alloc_failed_req_num++;
    ++it;
  }
}

void ContinuousBatchingStrategy::ProcessWaitingQueue() {
  PROFILE_EVENT_SCOPE(ProcessWaitingQueue, "ProcessWaitingQueue");
  KLLM_LOG_SCHEDULER << "ProcessWaitingQueue invoked, waiting queue size:" << batch_state_->waiting_queue.size()
                     << ", total_free_block_num:" << cache_manager_->GetUsableBlockNumber();

  std::vector<std::shared_ptr<InferRequest>> passed_reqs;
  auto [step_token_num, step_not_kv_cached_token_num] =
      CheckRunningQueueStepTokens(batch_state_->schedule_output->running_reqs, passed_reqs);
  KLLM_CHECK(passed_reqs.size() == batch_state_->schedule_output->running_reqs.size());

  size_t step_batch_size = batch_state_->schedule_output->running_reqs.size() + batch_state_->transfer_queue.size();
  // Per-DP local batch count, tracks this DP's own request count (not shared across DPs).
  size_t per_dp_batch_size = step_batch_size;
  size_t step_logits_num = 0;
  for (const auto &req : batch_state_->schedule_output->running_reqs) {
    step_logits_num += req->GetPlanningSamplingTokenNum();
  }

  scheduler_ticktok_->Barrier();

  if (dp_group_id_ == 0) {
    scheduler_ticktok_->Reset();
    scheduler_shared_counter_->step_batch_size.Reset(0);
    scheduler_shared_counter_->step_token_num.Reset(0);
    scheduler_shared_counter_->step_logits_num.Reset(0);
  }

  scheduler_ticktok_->Barrier();

  scheduler_shared_counter_->step_batch_size.Increase(step_batch_size);
  scheduler_shared_counter_->step_token_num.Increase(step_token_num);
  scheduler_shared_counter_->step_logits_num.Increase(step_logits_num);

  // Make sure all dp groups are accumulated.
  scheduler_ticktok_->Barrier();

  if (batch_state_->waiting_queue.empty()) {
    scheduler_ticktok_->Skip();
    return;
  }

  if (batch_state_->step_sched_finish) {
    KLLM_LOG_SCHEDULER << "Skip processing waiting_queue." << *batch_state_;
    scheduler_ticktok_->Skip();
    return;
  }

  for (auto it = batch_state_->waiting_queue.begin(); it != batch_state_->waiting_queue.end();) {
    auto req = *it;
    req->cache_manager = cache_manager_;
    ++it;
  }

  for (auto it = batch_state_->waiting_queue.begin(); it != batch_state_->waiting_queue.end();) {
    auto req = *it;
    EstimateRequestPlanningWorkload(req, batch_scheduler_config_.mtp_step_num + batch_scheduler_config_.ptp_step_num);

    // Check timeout, no finished req in waiting queue.
    if (CheckRequestTimeout(req)) {
      KLLM_LOG_SCHEDULER << "req timeout in waiting:" << req->req_id
                         << ", schedule_time_in_ms=" << batch_state_->schedule_time_in_ms
                         << ", req.timestamp_in_us/1000=" << req->timestamp_in_us / 1000
                         << ", waiting_timeout_in_ms=" << batch_scheduler_config_.waiting_timeout_in_ms;

      StopRequest(req, Status(RET_REQUEST_TIMEOUT, "timeout in waiting."), RequestState::kWaiting);
      it = batch_state_->waiting_queue.erase(it);
      continue;
    }

    // Check abort.
    if (req->aborted) {
      KLLM_LOG_SCHEDULER << "req aborted in waiting:" << req;
      StopRequest(req, Status(RET_REQUEST_TERMINATED, "req aborted in waiting."), RequestState::kWaiting);
      it = batch_state_->waiting_queue.erase(it);
      continue;
    }

    // Ignore draft tokens when speculative decoding is disabled
    if (!req->draft_tokens.spec.empty() && !batch_scheduler_config_.enable_speculative_decoding) {
      req->draft_tokens.spec.clear();
    }

    bool is_first_prefill_step = true;
    bool is_last_prefill_step = true;
    size_t req_step_block_num = 0;  // block number to be allocated in this step
    size_t shared_token_num = 0;
    size_t seq_token_num = req->GetPlanningSequenceLen();  // Trying to process remaining prefilling tokens

    // Reject empty requests early to prevent size_t underflow in prefix cache logic.
    if (seq_token_num == 0) {
      KLLM_LOG_ERROR << "Empty request detected, req_id=" << req->req_id << ", seq_token_num=0, aborting request.";
      StopRequest(req, Status(RET_INVALID_ARGUMENT, "Empty input tokens."), RequestState::kWaiting);
      it = batch_state_->waiting_queue.erase(it);
      continue;
    }
    ScheduleTaskWorkload planning_workload = req->GetPlanningWorkload();
    if (planning_workload.prefill_start_offset == 0) {
      // NOTE (winminkong): For input logits requests,
      // perform prefix block matching only for the tokens preceding the returned logits position.
      size_t prepared_shared_token_num = seq_token_num;
      if (runtime_config_.enable_prefix_caching) {
        auto it = req->request_target.find("logits");
        if (it != req->request_target.end() && !it->second.slice_pos.empty()) {
          prepared_shared_token_num = it->second.slice_pos.front().first;
        }
      }
      KLLM_LOG_SCHEDULER << "req_id: " << req->req_id << ", seq token num = " << seq_token_num
                         << ", prepared for prefix cached token num = " << prepared_shared_token_num;
      // Find shared blocks in first step
      size_t shared_block_num = 0;
      cache_manager_->GetRequestPrefixBlockNumber(req->req_id, req->GetPrefillingTokens(), prepared_shared_token_num,
                                                  shared_block_num, req_step_block_num, shared_token_num);
      if (shared_token_num > 0) {
        REPORT_COUNTER("prefix_cache_hit_req_num", static_cast<size_t>(1));
        REPORT_COUNTER("prefix_cache_hit_token_num", shared_token_num);
        REPORT_COUNTER("prefix_cache_hit_block_num", shared_block_num);
        KLLM_LOG_SCHEDULER << "req_id=" << req->req_id << " find shared_block_num=" << shared_block_num
                           << ", shared_token_num=" << shared_token_num;
      }
      REPORT_METRIC("prefix_cache_hit_ratio",
                    static_cast<double>(shared_token_num) / req->GetPrefillingTokens().size());
      if (shared_token_num > 0) {
        // Record this node's prefix-cache hit on the Request member (works even when req_ctx
        // is null, e.g. pd_v2 Prefill). KsanaPythonOutput / StreamingIterator read it back.
        req->prefix_cache_hit_tokens = static_cast<int>(shared_token_num);
      }
    } else {
      // Last task is chunked prefill task, previous tokens are sharable
      is_first_prefill_step = false;
      shared_token_num = planning_workload.prefill_start_offset;
    }
    // If splitfuse is enabled, and token number to be processed is larger than split_fuse_token_num,
    // adjust planning workload
    size_t req_step_token_num = seq_token_num - shared_token_num;
    if ((batch_scheduler_config_.split_fuse_token_num > 0) &&
        (req_step_token_num > batch_scheduler_config_.split_fuse_token_num)) {
      // not the last prefill step, adjust planning workload
      KLLM_LOG_SCHEDULER << "Not the last prefill step, req_step_token_num:" << req_step_token_num
                         << ", split_fuse_token_num:" << batch_scheduler_config_.split_fuse_token_num;

      is_last_prefill_step = false;
      planning_workload.prefill_token_num -= req_step_token_num - batch_scheduler_config_.split_fuse_token_num;
      req_step_token_num = batch_scheduler_config_.split_fuse_token_num;

      req->SetPlanningWorkload(planning_workload);
      seq_token_num = req->GetPlanningSequenceLen();

      assert(seq_token_num == shared_token_num + req_step_token_num);
      assert(!planning_workload.IsEmpty());
    }

    // add extra token for MTP sub model if needed
    req_step_block_num =
        cache_manager_->GetRequestStepBlockNumber(req->req_id, seq_token_num + runtime_config_.mtp_step_num);
    KLLM_LOG_SCHEDULER << "req_step_token_num:" << req_step_token_num << ", req_step_block_num:" << req_step_block_num
                       << ", seq_token_num:" << seq_token_num;

    const size_t launch_block_threshold = std::ceil(step_batch_size * batch_scheduler_config_.launch_block_threshold);

    // In this context, all other db groups will be paused.
    scheduler_ticktok_->Lock();

    // The total num include other dp groups.
    step_batch_size = scheduler_shared_counter_->step_batch_size.Get();
    step_token_num = scheduler_shared_counter_->step_token_num.Get();
    step_logits_num = scheduler_shared_counter_->step_logits_num.Get();

    // Get usable block number every time, the blocks matched by req are not reusable here.
    const size_t total_free_block_num = cache_manager_->GetRequestUsableBlockNumber(req->req_id);
    size_t req_token_num = runtime_config_.attn_backend_config.enable_blocked_multi_token_forwarding_kv
                               ? req_step_token_num
                               : seq_token_num;
    // Reserve at least mtp_step_num tokens per request to ensure
    // prefill drafts fit within buffers sized for multi-token decoding, e.g., MTP/PTP/DSpark.
    req_token_num = std::max(req_token_num, runtime_config_.mtp_step_num);
    if (step_logits_num + req->sampling_token_num <= dp_max_logits_num_ && step_batch_size < dp_max_batch_size_ &&
        per_dp_batch_size < per_dp_max_batch_size_ && step_token_num + req_token_num <= dp_max_step_token_num_ &&
        req_step_block_num + launch_block_threshold <= total_free_block_num) {
      // Assume we could succ, so that the AllocateRequestBlocks is not blocked.
      scheduler_shared_counter_->step_batch_size.Increase(1);
      scheduler_shared_counter_->step_token_num.Increase(req_token_num);
      scheduler_shared_counter_->step_logits_num.Increase(req->sampling_token_num);
      per_dp_batch_size++;
      scheduler_ticktok_->Unlock();

      Status status =
          cache_manager_->AllocateRequestBlocks(req->req_id, req_step_block_num, req->kv_cache_blocks,
                                                req->swa_kv_cache_blocks, req->swa_kv_cache_block_idx_offsets);
      if (status.OK()) {
        step_not_kv_cached_token_num += seq_token_num - shared_token_num;

        // Prefix/flexible 命中硬上限 seq-1，shared_token_num 恒小于 seq_token_num，不再走 FullHit 回退。
        KLLM_LOG_SCHEDULER << "shared token not equal forwarding size, " << req->ScheduleStateToStr();
        assert(seq_token_num > shared_token_num);
        if (shared_token_num > 0) {
          size_t planning_prefill_token_num =
              planning_workload.prefill_token_num + planning_workload.prefill_start_offset;
          planning_workload.prefill_start_offset = shared_token_num;
          planning_workload.prefill_token_num = planning_prefill_token_num - shared_token_num;
          req->SetPlanningWorkload(planning_workload);

          ScheduleTaskWorkload remaining_workload = req->GetRemainingWorkload();
          size_t prefill_token_num = remaining_workload.prefill_token_num + remaining_workload.prefill_start_offset;
          remaining_workload.prefill_start_offset = shared_token_num;
          remaining_workload.prefill_token_num = prefill_token_num - shared_token_num;
          req->SetRemainingWorkload(remaining_workload);
        }
        if (connector_config_.group_role == GroupRole::DECODE) {
          batch_state_->transfer_queue.emplace_back(req);
          it = batch_state_->waiting_queue.erase(it);
          KLLM_LOG_SCHEDULER << "Decode put req to transfer queue, req id: " << req->req_id;
          continue;
        }

        batch_state_->schedule_output->running_reqs.push_back(req);
        StampOnce(req->scheduled_time_us);
        if (is_last_prefill_step) {
          if (connector_config_.group_role == GroupRole::PREFILL && !req->is_mock_req) {
            req->sampling_config.max_new_tokens = 1;
            batch_state_->transfer_queue.push_back(req);
          } else {
            batch_state_->decoding_queue.push_back(req);
          }
          it = batch_state_->waiting_queue.erase(it);
        } else {
          it++;
        }

        // The flexible cache handling could be placed prior to the split_fuse break. Try moving it after testing.
        cache_manager_->UpdateFlexibleCache(req->req_id, req->GetPrefillingTokens(), shared_token_num,
                                            req->flexible_cached_copy_tasks, req->flexible_cache_len);
        if (req->flexible_cached_copy_tasks.size() > 0) {
          REPORT_COUNTER("flexible_cache_hit_req_num", static_cast<size_t>(1));
          REPORT_COUNTER("flexible_cache_hit_token_num", req->flexible_cached_copy_tasks.size());
          REPORT_METRIC("flexible_cache_hit_ratio",
                        static_cast<double>(req->flexible_cache_len) / req->GetPrefillingTokens().size());
        }

        req->SetCacheHitStatus(shared_token_num, is_first_prefill_step);

        continue;
      } else {
        // decrease counter if failed, thread-safe, no timing-control needed.
        scheduler_shared_counter_->step_batch_size.Decrease(1);
        scheduler_shared_counter_->step_token_num.Decrease(req_token_num);
        scheduler_shared_counter_->step_logits_num.Decrease(req->sampling_token_num);

        KLLM_LOG_ERROR << "Allocate blocks error, waiting req can not be launched, " << req->ScheduleStateToStr()
                       << ", msg=" << status.GetMessage();
      }
    } else {
      scheduler_ticktok_->Unlock();
    }

    KLLM_LOG_SCHEDULER << "total_free_block_num:" << cache_manager_->GetUsableBlockNumber();
    break;
  }

  // Current dp group finished, remove from loop list.
  scheduler_ticktok_->Skip();
}

void ContinuousBatchingStrategy::ProcessTransferQueue() {
  PROFILE_EVENT_SCOPE(ProcessTransferQueue, "ProcessTransferQueue");
  if (connector_config_.group_role == GroupRole::DECODE) {
    ProcessDecodeTransferQueue();
  }
  if (connector_config_.group_role == GroupRole::PREFILL) {
    ProcessPrefillTransferQueue();
  }
}

void ContinuousBatchingStrategy::Schedule(std::vector<std::shared_ptr<InferRequest>> &waiting_reqs) {
  batch_state_->ResetInfoBeforeSchedule();
  scheduler_ticktok_->SetThreadIndex(dp_group_id_);
  if (dp_group_id_ == 0) {
    scheduler_shared_counter_->step_batch_size.Reset(0);
    scheduler_shared_counter_->step_token_num.Reset(0);
    scheduler_shared_counter_->step_logits_num.Reset(0);
  }

  if (connector_config_.group_role != GroupRole::NONE) {
    // 对waiting_reqs排序，req_id小的在前
    std::sort(waiting_reqs.begin(), waiting_reqs.end(),
              [](const auto &a, const auto &b) { return a->req_id < b->req_id; });
  }
  batch_state_->MergeWaitingReqs(waiting_reqs);
  auto start_us = ProfileTimer::GetCurrentTimeInUs();

  // if last schedule output is not launched
  if (batch_state_->schedule_output->running_reqs.size() > 0) {
    return;
  }

  ProcessTransferQueue();
  auto transfer_queue_end_us = ProfileTimer::GetCurrentTimeInUs();
  REPORT_METRIC("batch_scheduler_transfer_queue_time_us", transfer_queue_end_us - start_us);
  ProcessDecodingQueue();
  auto running_queue_end_us = ProfileTimer::GetCurrentTimeInUs();
  REPORT_METRIC("batch_scheduler_running_queue_time_us", running_queue_end_us - transfer_queue_end_us);
  ProcessWaitingQueue();
  auto waiting_queue_end_us = ProfileTimer::GetCurrentTimeInUs();
  REPORT_METRIC("batch_scheduler_waiting_queue_time_us", waiting_queue_end_us - running_queue_end_us);
  AddTransferMeta(batch_state_->transfer_queue);
  auto add_transfer_meta_end_us = ProfileTimer::GetCurrentTimeInUs();
  REPORT_METRIC("batch_scheduler_add_transfer_meta_time_us", add_transfer_meta_end_us - waiting_queue_end_us);

  // Must barrier before reorder.
  scheduler_ticktok_->Barrier();

  // Change next visit order of dp groups, for load balance.
  if (dp_group_id_ == 0) {
    scheduler_ticktok_->Reorder();
  }

  batch_state_->schedule_output->SetPlanningTask();

  auto end_us = ProfileTimer::GetCurrentTimeInUs();
  REPORT_METRIC("batch_scheduler_time_us", end_us - start_us);
}

void ContinuousBatchingStrategy::ReportRequestProgressInfo(const std::shared_ptr<InferRequest> req) {
  if (req->kv_cached_token_num > req->computed_token_num && !req->is_mock_req) {
    if (connector_config_.group_role == GroupRole::DECODE && req->computed_token_num == 0) {
      req->computed_token_num = req->input_tokens.size();
    }
    auto computed_token_num = req->kv_cached_token_num - req->computed_token_num;
    if (req->kv_cached_token_num < req->input_tokens.size()) {
      REPORT_COUNTER("computed_input_token_num", computed_token_num);
    }
    if (req->kv_cached_token_num == req->input_tokens.size()) {
      REPORT_COUNTER("computed_input_token_num", computed_token_num);
      REPORT_COUNTER("computed_output_token_num", 1);
      REPORT_COUNTER("computed_token_num", 1);
    }
    if (req->kv_cached_token_num > req->input_tokens.size()) {
      REPORT_COUNTER("computed_output_token_num", computed_token_num);
    }
    REPORT_COUNTER("computed_token_num", computed_token_num);
    req->computed_token_num = req->kv_cached_token_num;
  }
}

}  // namespace ksana_llm
