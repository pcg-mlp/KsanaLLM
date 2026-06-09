/* Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * 事件驱动调度器：参考 ContinuousBatchingStrategy 的 ProcessDecodingQueue / ProcessWaitingQueue
 * 实现 ContinuousBatchingBackend，仅处理 decoding_reqs 与 prefilling_reqs。
 * ==============================================================================*/

#include "batch_scheduler/event_driven_scheduler/continuous_batching_backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>

#include "batch_scheduler/full_cache_hit_handler.h"
#include "batch_scheduler/planning_workload_estimator.h"
#include "configure/schedule_config_parser.h"
#include "profiler/reporter.h"
#include "runtime/schedule_output.h"
#include "utils/ret_code.h"

namespace ksana_llm {

namespace {

uint64_t BuildEventKey(ScheduleBackendEventType type, size_t dp_idx, int64_t req_id) {
  return (static_cast<uint64_t>(type) << 56) ^ (static_cast<uint64_t>(dp_idx) << 32) ^ static_cast<uint64_t>(req_id);
}

void EmitEventOnce(ScheduleBackendEventType type, size_t dp_idx, const std::shared_ptr<InferRequest>& req,
                   const Status& status, std::vector<ScheduleBackendEvent>& events,
                   std::unordered_set<uint64_t>& emitted_event_keys) {
  KLLM_CHECK(req != nullptr);
  KLLM_LOG_SCHEDULER << "EmitEventOnce: type=" << ScheduleBackendEventType2Str(type) << " dp_idx=" << dp_idx
                     << " req_id=" << req->req_id;
  const uint64_t event_key = BuildEventKey(type, dp_idx, req->req_id);
  if (!emitted_event_keys.insert(event_key).second) {
    return;
  }
  ScheduleBackendEvent event;
  event.type = type;
  event.dp_idx = dp_idx;
  event.req_ptr = req;
  event.status = status;
  events.push_back(std::move(event));
}

size_t GetMaxRequiredTokenNum(size_t token_num, size_t mtp_step_num) {
  return token_num + mtp_step_num - std::min(mtp_step_num, static_cast<size_t>(1));
}

bool IsDpBatchFull(const DpBatch& dp_batch, const BatchSchedulerConfig& config) {
  KLLM_CHECK(config.max_batch_size > 0);
  return dp_batch.running_reqs.size() >= static_cast<size_t>(config.max_batch_size);
}

bool CheckRequestFinish(const std::shared_ptr<InferRequest>& req, const BatchSchedulerConfig& config) {
  return req->IsEosGenerated() ||
         (req->sampling_config.max_new_tokens > 0 &&
          req->output_tokens.size() >= req->input_tokens.size() + req->sampling_config.max_new_tokens) ||
         // Keep the same semantics as ContinuousBatchingStrategy:
         // max_token_len limits the total sequence length (prompt + generated tokens), not generated-only length.
         req->output_tokens.size() >= config.max_token_len;
}

}  // namespace

ContinuousBatchingBackend::ContinuousBatchingBackend(const BatchSchedulerConfig& batch_scheduler_config)
    : ScheduleBackendInterface(batch_scheduler_config) {
  KLLM_CHECK(batch_scheduler_config.max_pp_batch_num > 0);
}

void ContinuousBatchingBackend::Schedule(const ScheduleContext& context,
                                         std::vector<std::shared_ptr<PlanningBatch>>& planning_batches,
                                         std::vector<ScheduleBackendEvent>& events) {
  events.clear();
  if (stopped_.load()) {
    return;
  }

  KLLM_CHECK(!planning_batches.empty());

  // TODO(robertyuan): support multi-batch
  PlanningBatch& planning_batch = *planning_batches[0];

  std::unordered_set<uint64_t> emitted_event_keys;
  ProcessDecodingQueue(context, planning_batch, events, emitted_event_keys);
  ProcessPrefillingQueue(context, planning_batch, events, emitted_event_keys);
}

void ContinuousBatchingBackend::UpdateWithGenerationResult(size_t multi_batch_id,
                                                           const GenerationOutputGroup& generation_output,
                                                           const ScheduleContext& context,
                                                           std::vector<ScheduleBackendEvent>& events) {
  events.clear();
  if (stopped_.load()) {
    return;
  }

  std::unordered_set<uint64_t> emitted_event_keys;
  for (size_t dp_idx = 0; dp_idx < generation_output.reqs.size(); ++dp_idx) {
    std::shared_ptr<CacheManagerInterface> cache_manager = context.GetDpCacheManager(dp_idx);
    for (const auto& req : generation_output.reqs[dp_idx]) {
      // req should never be nullptr here; otherwise the state machine is broken.
      KLLM_CHECK(req != nullptr);

      // If request is already stopped, do nothing.
      if (req->IsStopped()) {
        continue;
      }

      KLLM_LOG_SCHEDULER << "Processing generation result for req_id=" << req->req_id << " dp_idx=" << dp_idx
                         << ", generated_tokens=" << Vector2Str(req->generated_tokens);

      // For non-stopped requests, inflight task must exist here; otherwise the state machine is broken.
      // We must call ResetInflightTask() BEFORE handling the abort branch — otherwise HasInflightTask()
      // stays true and ProcessDelayedBackendEvents (event_driven_schedule_processor.cpp:325) skips the
      // queued kStopRequest event forever, leaving aborted reqs running with KV blocks held.
      KLLM_CHECK(req->HasInflightTask());
      req->UpdateAfterInflightTaskFinished();
      req->ResetInflightTask();
      req->flexible_cached_copy_tasks.clear();

      // Request may become stopped after UpdateAfterInflightTaskFinished(); emit stop event immediately.
      if (req->IsStopped()) {
        continue;
      }

      // If client aborted (e.g. disconnected mid-stream), emit a stop event so HandleStopRequest releases
      // KV blocks and calls req->Notify(), which signals abort_waiter and wakes the dying_request thread
      // parked in ~StreamingIterator() (streaming_iterator.cpp:46).
      if (req->aborted) {
        KLLM_LOG_WARNING << "client aborted in UpdateWithGenerationResult, emit stop: req_id=" << req->req_id
                         << " dp_idx=" << dp_idx;
        req->finish_status = Status(RET_REQUEST_TERMINATED, "req aborted in running.");
        EmitEventOnce(ScheduleBackendEventType::kStopRequest, dp_idx, req, req->finish_status, events,
                      emitted_event_keys);
        REPORT_COUNTER("forward_req_aborted_num", static_cast<size_t>(1));
        continue;
      }

      bool block_merged = false;
      const Status update_status = cache_manager->UpdateRequestTokens(
          req->req_id, req->forwarding_tokens, req->kv_cached_token_num, req->kv_cache_blocks, block_merged,
          req->swa_kv_cache_blocks, req->swa_kv_cache_block_idx_offsets);
      if (!update_status.OK()) {
        EmitEventOnce(ScheduleBackendEventType::kRecomputeRequest, dp_idx, req, update_status, events,
                      emitted_event_keys);
        continue;
      }

      if (CheckRequestFinish(req, batch_scheduler_config_)) {
        req->finish_status = Status(RET_SUCCESS);
        req->finished = true;
        EmitEventOnce(ScheduleBackendEventType::kStopRequest, dp_idx, req, Status(RET_SUCCESS), events,
                      emitted_event_keys);
        continue;
      }

      req->NotifyStep();
    }
  }
}

void ContinuousBatchingBackend::Stop() { stopped_.store(true); }

void ContinuousBatchingBackend::ProcessDecodingQueue(const ScheduleContext& context, PlanningBatch& planning_batch,
                                                     std::vector<ScheduleBackendEvent>& events,
                                                     std::unordered_set<uint64_t>& emitted_event_keys) {
  // NOTE:
  // - For decoding stage, allocation failure is handled in a dp-level manner.
  // - If *all* allocatable requests in a dp fail to allocate blocks and no running requests in the dp, it indicates OOM
  //   pressure. The scheduler should trigger a recompute event to release memory.
  const std::vector<WorkloadQuota> dp_scheduling_order =
      workload_balancer_->GetDpSchedulingOrderAndQuota(planning_batch, context, InferStage::kDecode);
  for (const WorkloadQuota& quota : dp_scheduling_order) {
    const size_t dp_idx = quota.dp_idx;
    KLLM_CHECK(dp_idx < planning_batch.dp_batches.size());
    const DpScheduleState& dp_state = context.GetDpScheduleState(dp_idx);
    DpScheduleOpsInPlan& in_plan_ops = context.GetMutableInPlanScheduleOps(dp_idx);
    auto& dp_batch = planning_batch.dp_batches[dp_idx];
    auto& cache_manager = context.GetDpCacheManager(dp_idx);

    KLLM_LOG_SCHEDULER << "ProcessDecodingQueue: dp_idx=" << dp_idx
                       << ", decoding_reqs=" << dp_state.decoding_reqs.size()
                       << ", running_reqs=" << dp_batch.running_reqs.size();

    bool has_candidate = false;
    bool has_success = false;
    std::shared_ptr<InferRequest> decoding_recompute_candidate = nullptr;
    Status last_failed_status;

    for (const auto& req : dp_state.decoding_reqs) {
      KLLM_CHECK(req != nullptr && !req->IsStopped());

      // Catch reqs aborted while sitting in the decoding queue — must emit
      // kStopRequest so HandleStopRequest releases blocks and notifies
      // abort_waiter (otherwise the dying_request thread parked in
      // ~StreamingIterator() blocks forever).
      if (req->aborted) {
        KLLM_LOG_WARNING << "client aborted in decoding queue, emit stop: req_id=" << req->req_id
                         << " dp_idx=" << dp_idx;
        EmitEventOnce(ScheduleBackendEventType::kStopRequest, dp_idx, req,
                      Status(RET_REQUEST_TERMINATED, "client aborted"), events, emitted_event_keys);
        continue;
      }

      if (IsDpBatchFull(dp_batch, batch_scheduler_config_)) {
        KLLM_LOG_SCHEDULER << "Skip adding decoding req because batch is full. dp_idx=" << dp_idx
                           << ", running_reqs=" << dp_batch.running_reqs.size()
                           << ", max_batch_size=" << batch_scheduler_config_.max_batch_size;
        break;
      }

      if (dp_batch.IsInBatch(req->req_id)) {
        continue;
      }
      if (req->HasPlanningTask()) {
        continue;
      }

      EstimateRequestPlanningWorkload(req, batch_scheduler_config_.mtp_step_num + batch_scheduler_config_.ptp_step_num);

      has_candidate = true;
      if (decoding_recompute_candidate == nullptr) {
        decoding_recompute_candidate = req;
      }

      const size_t max_required_token_num =
          GetMaxRequiredTokenNum(req->GetPlanningSequenceLen(), batch_scheduler_config_.mtp_step_num);
      const bool request_acceptable =
          workload_balancer_->IsRequestAcceptable(req, planning_batch, quota, max_required_token_num);
      if (!request_acceptable) {
        continue;
      }

      const size_t req_step_block_num = cache_manager->GetRequestStepBlockNumber(req->req_id, max_required_token_num);
      Status status =
          cache_manager->AllocateRequestBlocks(req->req_id, req_step_block_num, req->kv_cache_blocks,
                                               req->swa_kv_cache_blocks, req->swa_kv_cache_block_idx_offsets);
      if (status.OK()) {
        has_success = true;
        planning_batch.AddRunningReq(dp_idx, req);
        KLLM_LOG_SCHEDULER << "Allocated " << req_step_block_num << " blocks for req_id=" << req->req_id
                           << ", add to batch, " << req->ScheduleStateToStr();
      } else {
        last_failed_status = status;
      }
    }

    // If decoding queue has no launchable request and current dp batch is empty, and no async recompute/stop/swapout
    // event has been emitted, trigger a recompute/swapout event to release memory.
    // 1) Prefer recompute a prefilling request that already owns kv blocks.
    // 2) Otherwise recompute one decoding request.
    if (has_candidate && !has_success && (dp_batch.running_reqs.size() == 0) && in_plan_ops.stop_req_ids.empty() &&
        in_plan_ops.recompute_req_ids.empty()) {
      std::shared_ptr<InferRequest> recompute_candidate = nullptr;
      for (const auto& prefill_req : dp_state.prefilling_reqs) {
        if (prefill_req == nullptr || prefill_req->IsStopped()) {
          continue;
        }
        if (!prefill_req->kv_cache_blocks.empty()) {
          recompute_candidate = prefill_req;
          batch_scheduler_config_.split_fuse_token_num = 0;
          KLLM_LOG_INFO << "Split fuse disabled due to allocation failure. Recompute prefilling request req_id="
                        << prefill_req->req_id << ", dp_idx=" << dp_idx;
          break;
        }
      }
      if (recompute_candidate == nullptr && (dp_state.decoding_reqs.size() > 1)) {
        recompute_candidate = decoding_recompute_candidate;
      }
      if (recompute_candidate != nullptr) {
        EmitEventOnce(ScheduleBackendEventType::kRecomputeRequest, dp_idx, recompute_candidate, last_failed_status,
                      events, emitted_event_keys);
        in_plan_ops.recompute_req_ids.insert(recompute_candidate->req_id);
      }
    }
  }
}

void ContinuousBatchingBackend::ProcessPrefillingQueue(const ScheduleContext& context, PlanningBatch& planning_batch,
                                                       std::vector<ScheduleBackendEvent>& events,
                                                       std::unordered_set<uint64_t>& emitted_event_keys) {
  const std::vector<WorkloadQuota> dp_scheduling_order =
      workload_balancer_->GetDpSchedulingOrderAndQuota(planning_batch, context, InferStage::kContext);
  for (const WorkloadQuota& quota : dp_scheduling_order) {
    const size_t dp_idx = quota.dp_idx;
    KLLM_CHECK(dp_idx < planning_batch.dp_batches.size());
    const DpScheduleState& dp_state = context.GetDpScheduleState(dp_idx);
    DpScheduleOpsInPlan& in_plan_ops = context.GetMutableInPlanScheduleOps(dp_idx);

    auto& dp_batch = planning_batch.dp_batches[dp_idx];
    auto& cache_manager = context.GetDpCacheManager(dp_idx);

    KLLM_LOG_SCHEDULER << "ProcessPrefillingQueue: dp_idx=" << dp_idx
                       << ", prefilling_reqs=" << dp_state.prefilling_reqs.size()
                       << ", running_reqs=" << dp_batch.running_reqs.size();

    bool has_candidate = false;
    bool has_success = false;
    Status last_failed_status;
    for (const auto& req : dp_state.prefilling_reqs) {
      if (IsDpBatchFull(dp_batch, batch_scheduler_config_)) {
        KLLM_LOG_SCHEDULER << "Skip adding prefilling req because batch is full. dp_idx=" << dp_idx
                           << ", running_reqs=" << dp_batch.running_reqs.size()
                           << ", max_batch_size=" << batch_scheduler_config_.max_batch_size;
        break;
      }

      if (req == nullptr || req->IsStopped()) {
        continue;
      }
      if (req->aborted) {
        KLLM_LOG_WARNING << "client aborted in prefilling queue, emit stop: req_id=" << req->req_id
                         << " dp_idx=" << dp_idx;
        EmitEventOnce(ScheduleBackendEventType::kStopRequest, dp_idx, req,
                      Status(RET_REQUEST_TERMINATED, "client aborted"), events, emitted_event_keys);
        continue;
      }
      if (dp_batch.IsInBatch(req->req_id)) {
        continue;
      }
      if (req->HasPlanningTask()) {
        continue;
      }

      EstimateRequestPlanningWorkload(req, batch_scheduler_config_.mtp_step_num + batch_scheduler_config_.ptp_step_num);
      has_candidate = true;

      bool is_first_prefill_step = true;
      bool is_last_prefill_step = true;
      size_t shared_block_num = 0;
      size_t shared_token_num = 0;
      size_t req_step_block_num = 0;
      size_t seq_token_num = req->GetPlanningSequenceLen();
      ScheduleTaskWorkload planning_workload = req->GetPlanningWorkload();
      if (planning_workload.prefill_start_offset == 0) {
        Status status =
            cache_manager->GetRequestPrefixBlockNumber(req->req_id, req->GetPrefillingTokens(), seq_token_num,
                                                       shared_block_num, req_step_block_num, shared_token_num);
        if (!status.OK()) {
          continue;
        }
      } else {
        is_first_prefill_step = false;
        shared_token_num = planning_workload.prefill_start_offset;
      }

      size_t req_step_token_num = seq_token_num - shared_token_num;
      if (batch_scheduler_config_.split_fuse_token_num > 0 &&
          req_step_token_num > batch_scheduler_config_.split_fuse_token_num) {
        // not the last prefill step, adjust planning workload
        KLLM_LOG_SCHEDULER << "req_id=" << req->req_id
                           << ". Not the last prefill step, req_step_token_num:" << req_step_token_num
                           << ", split_fuse_token_num:" << batch_scheduler_config_.split_fuse_token_num
                           << ", adjust current planning workload " << planning_workload.ToString();
        is_last_prefill_step = false;
        planning_workload.prefill_token_num -= (req_step_token_num - batch_scheduler_config_.split_fuse_token_num);
        req_step_token_num = planning_workload.prefill_token_num;
        req->SetPlanningWorkload(planning_workload);
        seq_token_num = req->GetPlanningSequenceLen();
      }

      const bool request_acceptable =
          workload_balancer_->IsRequestAcceptable(req, planning_batch, quota, req_step_token_num);
      if (!request_acceptable) {
        continue;
      }

      // add extra token for MTP sub model if needed
      req_step_block_num =
          cache_manager->GetRequestStepBlockNumber(req->req_id, seq_token_num + batch_scheduler_config_.mtp_step_num);

      const size_t launch_block_threshold =
          static_cast<size_t>(std::ceil(dp_batch.running_reqs.size() * batch_scheduler_config_.launch_block_threshold));
      const size_t total_free_block_num = cache_manager->GetRequestUsableBlockNumber(req->req_id);
      if (req_step_block_num + launch_block_threshold > total_free_block_num) {
        continue;
      }

      Status status =
          cache_manager->AllocateRequestBlocks(req->req_id, req_step_block_num, req->kv_cache_blocks,
                                               req->swa_kv_cache_blocks, req->swa_kv_cache_block_idx_offsets);
      if (!status.OK()) {
        KLLM_LOG_SCHEDULER << "Failed to allocate blocks for req_id=" << req->req_id
                           << ", required_block_num=" << req_step_block_num
                           << ", total_free_block_num=" << total_free_block_num;
        last_failed_status = status;
        continue;
      }
      has_success = true;

      if (shared_token_num >= seq_token_num) {
        HandleFullPrefixCacheHit(req, planning_workload, seq_token_num);
      } else if (shared_token_num > 0) {
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

      req->SetCacheHitStatus(shared_token_num, planning_workload.prefill_start_offset == 0);
      planning_batch.AddRunningReq(dp_idx, req);
      KLLM_LOG_SCHEDULER << "req_id=" << req->req_id << ", shared_token_num=" << shared_token_num << ", "
                         << req->ScheduleStateToStr() << ", is_last_prefill_step=" << is_last_prefill_step;
      if (is_last_prefill_step) {
        // After last prefill step is planned, move request to decoding queue for next step planning.
        EmitEventOnce(ScheduleBackendEventType::kMoveToDecode, dp_idx, req, Status(RET_SUCCESS), events,
                      emitted_event_keys);
      }

      // The flexible cache handling could be placed prior to the split_fuse break. Try moving it after testing.
      cache_manager->UpdateFlexibleCache(req->req_id, req->GetPrefillingTokens(), shared_token_num,
                                         req->flexible_cached_copy_tasks, req->flexible_cache_len);

      req->SetCacheHitStatus(shared_token_num, is_first_prefill_step);
    }

    // if this dp has launchable prefill requests but all allocations fail and no running request exists,
    // trigger recompute chunk-prefilled requests to release memory.
    if (has_candidate && !has_success && (dp_batch.running_reqs.size() == 0) && dp_state.decoding_reqs.empty() &&
        in_plan_ops.stop_req_ids.empty() && in_plan_ops.recompute_req_ids.empty()) {
      std::shared_ptr<InferRequest> recompute_candidate = nullptr;
      for (const auto& prefill_req : dp_state.prefilling_reqs) {
        if (prefill_req == nullptr || prefill_req->IsStopped()) {
          continue;
        }
        if (!prefill_req->kv_cache_blocks.empty()) {
          recompute_candidate = prefill_req;
          batch_scheduler_config_.split_fuse_token_num = 0;
          KLLM_LOG_INFO << "Split fuse disabled due to allocation failure. Recompute prefilling request req_id="
                        << prefill_req->req_id << ", dp_idx=" << dp_idx;
          break;
        }
      }

      if (recompute_candidate != nullptr) {
        // Recompute only. Swapout is not supported for chunk-prefilled requests.
        EmitEventOnce(ScheduleBackendEventType::kRecomputeRequest, dp_idx, recompute_candidate, last_failed_status,
                      events, emitted_event_keys);
        in_plan_ops.recompute_req_ids.insert(recompute_candidate->req_id);
      }
    }
  }
}

}  // namespace ksana_llm
