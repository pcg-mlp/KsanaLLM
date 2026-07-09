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
#include "batch_scheduler/request_finish_checker.h"
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

  size_t multi_batch_id = context.GetCurrentMultiBatchId();
  KLLM_CHECK(multi_batch_id < planning_batches.size());
  PlanningBatch& planning_batch = *planning_batches[multi_batch_id];

  std::unordered_set<uint64_t> emitted_event_keys;
  // pd_v2: scan preprocessing_reqs for client abort (defer stop when awaiting prefill).
  ScanAbortedPreprocessingReqs(context, planning_batch, events, emitted_event_keys);
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
      // stays true and ProcessDelayedBackendEvents skips the queued kStopRequest event
      // forever, leaving aborted reqs running with KV blocks held.
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
      // parked in ~StreamingIterator().
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
        KLLM_LOG_SCHEDULER << "UpdateRequestTokens failed for req_id=" << req->req_id << ", dp_idx=" << dp_idx
                           << ", status=" << update_status.GetMessage()
                           << ", forwarding_tokens.size=" << req->forwarding_tokens.size()
                           << ", kv_cached_token_num=" << req->kv_cached_token_num
                           << ", kv_cache_blocks.size=" << req->kv_cache_blocks.size()
                           << " => emitting kRecomputeRequest";
        EmitEventOnce(ScheduleBackendEventType::kRecomputeRequest, dp_idx, req, update_status, events,
                      emitted_event_keys);
        continue;
      }

      if (IsRequestFinished(req, batch_scheduler_config_)) {
        req->finish_status = Status(RET_SUCCESS);
        req->finished = true;
        StampOnce(req->finish_time_us);
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

  // 同一 dp_idx 可能有多轮 quota，需要跨 quota 聚合 has_schedulable_req / has_scheduled_req / candidate。
  // round-1 small_quota 拒绝后不能立刻触发 recompute，round-2 large_quota 才是预期 fallback。
  struct DecodeDpAggregate {
    // 有可调度候选：请求已经通过前置过滤，准备尝试调度。
    bool has_schedulable_req = false;
    // 有成功调度请求：至少一个请求已经成功分配 block，并加入 running batch。
    bool has_scheduled_req = false;
    std::shared_ptr<InferRequest> decoding_recompute_candidate;
    Status last_failed_status;
  };
  std::unordered_map<size_t, DecodeDpAggregate> agg_by_dp;
  // dp_first_seen_order: 保序去重的 dp_idx 索引，与 agg_by_dp 配套使用。
  // 同一 dp_idx 在 dp_scheduling_order 中会出现多次（round-1 small_quota +
  // round-2 large_quota），这里仅在首次遇到某 dp_idx 时记录一次，记录的是它
  // 被调度到的先后顺序。后处理阶段（recompute 触发）按此顺序遍历，而非直接
  // 遍历 agg_by_dp——unordered_map 遍历顺序不确定，借此保证遍历/触发顺序稳定可复现。
  std::vector<size_t> dp_first_seen_order;
  dp_first_seen_order.reserve(dp_scheduling_order.size());

  for (const WorkloadQuota& quota : dp_scheduling_order) {
    const size_t dp_idx = quota.dp_idx;
    KLLM_CHECK(dp_idx < planning_batch.dp_batches.size());
    const DpScheduleState& dp_state = context.GetDpScheduleState(context.GetCurrentMultiBatchId(), dp_idx);
    auto& dp_batch = planning_batch.dp_batches[dp_idx];
    auto& cache_manager = context.GetDpCacheManager(dp_idx);

    KLLM_LOG_SCHEDULER << "ProcessDecodingQueue: dp_idx=" << dp_idx
                       << ", decoding_reqs=" << dp_state.decoding_reqs.size()
                       << ", running_reqs=" << dp_batch.running_reqs.size()
                       << ", step_token_limit=" << quota.step_token_limit;

    auto agg_it = agg_by_dp.find(dp_idx);
    if (agg_it == agg_by_dp.end()) {
      agg_it = agg_by_dp.emplace(dp_idx, DecodeDpAggregate{}).first;
      dp_first_seen_order.push_back(dp_idx);
    }
    DecodeDpAggregate& agg = agg_it->second;

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

      agg.has_schedulable_req = true;
      // pd_v2: 允许 kv_from_remote 也就是正在进行decode的请求成为抢占候选。Decode 侧请求都会带 kv_from_remote=true，
      if (agg.decoding_recompute_candidate == nullptr) {
        agg.decoding_recompute_candidate = req;
      }

      const size_t max_required_token_num =
          GetMaxRequiredTokenNum(req->GetPlanningSequenceLen(), batch_scheduler_config_.mtp_step_num);
      const size_t step_query_token_num = req->GetPlanningQueryLen();
      // decode 一步只算 per-step query (sampling + draft ≈ 3 token), 全局与 per-dp 检查同单位,
      // 故两个参数都传 step_query_token_num。
      const bool request_acceptable = workload_balancer_->IsRequestAcceptable(
          req, planning_batch, quota, step_query_token_num, step_query_token_num);
      if (!request_acceptable) {
        KLLM_LOG_SCHEDULER << "Skip decoding req_id=" << req->req_id
                           << ": IsRequestAcceptable=false, running_req_num=" << planning_batch.GetRunningReqNum()
                           << ", max_batch_size=" << batch_scheduler_config_.max_batch_size
                           << ", max_required_token_num=" << max_required_token_num
                           << ", sampling_token_num=" << req->sampling_token_num;
        continue;
      }

      const size_t req_step_block_num = cache_manager->GetRequestStepBlockNumber(req->req_id, max_required_token_num);
      Status status =
          cache_manager->AllocateRequestBlocks(req->req_id, req_step_block_num, req->kv_cache_blocks,
                                               req->swa_kv_cache_blocks, req->swa_kv_cache_block_idx_offsets);
      if (status.OK()) {
        agg.has_scheduled_req = true;
        planning_batch.AddRunningReq(dp_idx, req);
        // T4: first entry into the running batch = scheduled (decode compute start on the pd_v2
        // Decode node). Revives waiting_ms = scheduled - arrive in the access log.
        StampOnce(req->scheduled_time_us);
        KLLM_LOG_SCHEDULER << "Allocated " << req_step_block_num << " blocks for req_id=" << req->req_id
                           << ", add to batch, " << req->ScheduleStateToStr();
      } else {
        agg.last_failed_status = status;
      }
    }
  }

  // 同一 dp 的所有 quota 都处理完后再触发 recompute/swapout，避免跳过 round-2 fallback。
  // 优先 recompute 已持有 KV block 的 prefilling 请求，否则选择一个 decoding 请求。
  for (size_t dp_idx : dp_first_seen_order) {
    const DecodeDpAggregate& agg = agg_by_dp.at(dp_idx);
    if (!agg.has_schedulable_req || agg.has_scheduled_req) {
      continue;
    }
    const DpScheduleState& dp_state = context.GetDpScheduleState(context.GetCurrentMultiBatchId(), dp_idx);
    DpScheduleOpsInPlan& in_plan_ops = context.GetMutableInPlanScheduleOps(context.GetCurrentMultiBatchId(), dp_idx);
    auto& dp_batch = planning_batch.dp_batches[dp_idx];
    if (dp_batch.running_reqs.size() != 0 || !in_plan_ops.stop_req_ids.empty() ||
        !in_plan_ops.recompute_req_ids.empty()) {
      continue;
    }
    std::shared_ptr<InferRequest> recompute_candidate = nullptr;
    for (const auto& prefill_req : dp_state.prefilling_reqs) {
      if (prefill_req == nullptr || prefill_req->IsStopped()) {
        continue;
      }
      // pd_v2 state-machine invariant check: a req with pd_v2_awaiting_prefill
      // = true has kv_cache_blocks wired into peer Prefill's RDMA target list.
      // It MUST sit in preprocessing_reqs (the hook chain returned
      // kWaitAsyncResult, which routes there) and be invisible to backend
      // victim scans. If we see one here, the state machine drifted somewhere
      // — log loudly so we can locate the bug. We deliberately do NOT skip
      // it: silent skip would mask the bug; letting recompute proceed will
      // surface as visible KV corruption in tests, which is the correct
      // failure mode for a state-machine violation.
      if (prefill_req->pd_v2.awaiting_prefill) {
        KLLM_LOG_ERROR << "State machine violation: pd_v2 state-C req_id=" << prefill_req->req_id
                       << " (pd_v2_awaiting_prefill=true) found in prefilling_reqs, "
                       << "expected preprocessing_reqs. RDMA may be in flight.";
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
      recompute_candidate = agg.decoding_recompute_candidate;
    }
    if (recompute_candidate == nullptr) {
      continue;
    }
    // The kv_from_remote branch is handled downstream in HandleRecomputeRequest.
    EmitEventOnce(ScheduleBackendEventType::kRecomputeRequest, dp_idx, recompute_candidate, agg.last_failed_status,
                  events, emitted_event_keys);
    in_plan_ops.recompute_req_ids.insert(recompute_candidate->req_id);
  }
}

void ContinuousBatchingBackend::ProcessPrefillingQueue(const ScheduleContext& context, PlanningBatch& planning_batch,
                                                       std::vector<ScheduleBackendEvent>& events,
                                                       std::unordered_set<uint64_t>& emitted_event_keys) {
  const std::vector<WorkloadQuota> dp_scheduling_order =
      workload_balancer_->GetDpSchedulingOrderAndQuota(planning_batch, context, InferStage::kContext);

  // 同一 dp_idx 可能有多轮 quota，需要跨 quota 聚合 has_schedulable_req / has_scheduled_req，
  // 避免 round-1 small_quota 拒绝后直接 recompute，跳过 round-2 large_quota fallback。
  struct PrefillDpAggregate {
    // 有可调度候选：请求已经通过前置过滤，准备尝试调度。
    bool has_schedulable_req = false;
    // 有成功调度请求：至少一个请求已经成功分配 block，并加入 running batch。
    bool has_scheduled_req = false;
    Status last_failed_status;
  };
  std::unordered_map<size_t, PrefillDpAggregate> agg_by_dp;
  // dp_first_seen_order: 保序去重的 dp_idx 索引，与 agg_by_dp 配套使用。
  // 同一 dp_idx 在 dp_scheduling_order 中会出现多次（round-1 small_quota +
  // round-2 large_quota），这里仅在首次遇到某 dp_idx 时记录一次，记录的是它
  // 被调度到的先后顺序。后处理阶段（recompute 触发）按此顺序遍历，而非直接
  // 遍历 agg_by_dp——unordered_map 遍历顺序不确定，借此保证遍历/触发顺序稳定可复现。
  std::vector<size_t> dp_first_seen_order;
  dp_first_seen_order.reserve(dp_scheduling_order.size());

  for (const WorkloadQuota& quota : dp_scheduling_order) {
    const size_t dp_idx = quota.dp_idx;
    KLLM_CHECK(dp_idx < planning_batch.dp_batches.size());
    const DpScheduleState& dp_state = context.GetDpScheduleState(context.GetCurrentMultiBatchId(), dp_idx);

    auto& dp_batch = planning_batch.dp_batches[dp_idx];
    auto& cache_manager = context.GetDpCacheManager(dp_idx);

    KLLM_LOG_SCHEDULER << "ProcessPrefillingQueue: dp_idx=" << dp_idx
                       << ", prefilling_reqs=" << dp_state.prefilling_reqs.size()
                       << ", running_reqs=" << dp_batch.running_reqs.size()
                       << ", step_token_limit=" << quota.step_token_limit;

    auto agg_it = agg_by_dp.find(dp_idx);
    if (agg_it == agg_by_dp.end()) {
      agg_it = agg_by_dp.emplace(dp_idx, PrefillDpAggregate{}).first;
      dp_first_seen_order.push_back(dp_idx);
    }
    PrefillDpAggregate& agg = agg_it->second;

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
      agg.has_schedulable_req = true;

      bool is_first_prefill_step = true;
      bool is_last_prefill_step = true;
      size_t shared_block_num = 0;
      size_t shared_token_num = 0;
      size_t req_step_block_num = 0;
      size_t seq_token_num = req->GetPlanningSequenceLen();
      ScheduleTaskWorkload planning_workload = req->GetPlanningWorkload();
      // 跳过条件：hook 已经分配过 block (kv_cache_blocks 非空)。这条路径覆盖
      // pd_v2 / V1 + global cache async 等场景：GlobalCacheQueryHook 已完成
      // GetRequestPrefixBlockNumber + AllocateRequestBlocks，state 已推到 kRunning，
      // 此处再调 GetRequestPrefixBlockNumber 会撞状态守卫 (要求 kWaiting / is_remote_loading)。
      // 直接复用 hook 写入的 workload。
      const bool prepared_by_hook = !req->kv_cache_blocks.empty();
      if (prepared_by_hook) {
        is_first_prefill_step = (planning_workload.prefill_start_offset == 0);
        shared_token_num = planning_workload.prefill_start_offset;
      } else if (planning_workload.prefill_start_offset == 0) {
        Status status =
            cache_manager->GetRequestPrefixBlockNumber(req->req_id, req->GetPrefillingTokens(), seq_token_num,
                                                       shared_block_num, req_step_block_num, shared_token_num);
        if (!status.OK()) {
          KLLM_LOG_SCHEDULER << "Skip prefilling req_id=" << req->req_id
                             << ": GetRequestPrefixBlockNumber failed: " << status.GetMessage();
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

      // 全局 max_step 检查用整条 seq (prefix + new): MLA attention kernel 为
      // total_tokens = total_q_tokens + total_prefix_tokens 分配 prefix_kv_buffer, 只用
      // req_step_token_num (新 token) 会在 prefix 命中大时超 max_step_tokens → CUDA illegal
      // memory access (MlaFlashPrefixKVReverseCacheCopy)。
      // per-dp quota 检查用 req_step_token_num (本步实际新算的 token): 它与 dp_batch
      // .running_step_token_num 累加的 per-step query 同单位, 也不会让 prefix 命中的请求被
      // per-dp 份额误拒 (见 IsRequestAcceptable 的 dp 空豁免)。
      const size_t req_total_token_num = seq_token_num;
      const bool request_acceptable =
          workload_balancer_->IsRequestAcceptable(req, planning_batch, quota, req_total_token_num, req_step_token_num);
      if (!request_acceptable) {
        KLLM_LOG_SCHEDULER << "Skip prefilling req_id=" << req->req_id
                           << ": IsRequestAcceptable=false, running_req_num=" << planning_batch.GetRunningReqNum()
                           << ", req_step_token_num=" << req_step_token_num;
        continue;
      }

      // mtp_step_num 单独透传给 cache_manager，统一处理 next-sampler 与 MTP lookahead 的 block 预留。
      // 这样 hook fast-path 与 backend fallback 共享同一套 block 计数逻辑，避免 pd_v2 dispatcher 两侧不一致。
      req_step_block_num =
          cache_manager->GetRequestStepBlockNumber(req->req_id, seq_token_num, batch_scheduler_config_.mtp_step_num);

      const size_t launch_block_threshold =
          static_cast<size_t>(std::ceil(dp_batch.running_reqs.size() * batch_scheduler_config_.launch_block_threshold));
      const size_t total_free_block_num = cache_manager->GetRequestUsableBlockNumber(req->req_id);
      if (req_step_block_num + launch_block_threshold > total_free_block_num) {
        KLLM_LOG_SCHEDULER << "Skip prefilling req_id=" << req->req_id
                           << ": block threshold check failed, req_step_block_num=" << req_step_block_num
                           << ", launch_block_threshold=" << launch_block_threshold
                           << ", total_free_block_num=" << total_free_block_num;
        continue;
      }

      Status status =
          cache_manager->AllocateRequestBlocks(req->req_id, req_step_block_num, req->kv_cache_blocks,
                                               req->swa_kv_cache_blocks, req->swa_kv_cache_block_idx_offsets);
      if (!status.OK()) {
        KLLM_LOG_SCHEDULER << "Failed to allocate blocks for req_id=" << req->req_id
                           << ", required_block_num=" << req_step_block_num
                           << ", total_free_block_num=" << total_free_block_num;
        agg.last_failed_status = status;
        continue;
      }
      agg.has_scheduled_req = true;

      if (shared_token_num >= seq_token_num) {
        // pd_v2 Prefill 节点全命中绝不能 flip 到 kDecode: 带 decode_targets 的请求必须保持
        // kContext 走完整 prefill 流程 (回退 1 token 算 logits) 并把 KV RDMA 给 decode 对端;
        // 一旦 flip 成 kDecode, 请求会在本地连续 decode、KvDispatcher 因 step!=0 跳过 KV 分发,
        // barrier 永不上报 → 对端 decode 永久挂起 (见 full_cache_hit_handler.h 注释)。
        // 与 GlobalCacheQueryHook / GlobalCacheFetcher 的 is_pd_v2_prefill 分支对齐。
        HandleFullPrefixCacheHit(req, planning_workload, seq_token_num,
                                 /*flip_to_decode_stage=*/req->pd_v2.decode_targets.empty());
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

      if (shared_token_num > 0) {
        // Single source of truth: the Request member (works even when req_ctx is null, e.g.
        // pd_v2 Prefill). KsanaPythonOutput / StreamingIterator read it back.
        req->prefix_cache_hit_tokens = static_cast<int>(shared_token_num);
      }

      req->SetCacheHitStatus(shared_token_num, planning_workload.prefill_start_offset == 0);
      planning_batch.AddRunningReq(dp_idx, req);
      // T4 (aggregate/prefill path): first entry into the running batch = scheduled.
      StampOnce(req->scheduled_time_us);
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
  }

  // 同一 dp 的所有 quota 都处理完后再触发 recompute，避免跳过 round-2 large_quota fallback。
  // 若该 dp 有可调度 prefill 请求但分配全部失败，且没有运行中的请求，则 recompute 一个
  // chunk-prefilled 请求释放内存。
  for (size_t dp_idx : dp_first_seen_order) {
    const PrefillDpAggregate& agg = agg_by_dp.at(dp_idx);
    if (!agg.has_schedulable_req || agg.has_scheduled_req) {
      continue;
    }
    const DpScheduleState& dp_state = context.GetDpScheduleState(context.GetCurrentMultiBatchId(), dp_idx);
    DpScheduleOpsInPlan& in_plan_ops = context.GetMutableInPlanScheduleOps(context.GetCurrentMultiBatchId(), dp_idx);
    auto& dp_batch = planning_batch.dp_batches[dp_idx];
    if (dp_batch.running_reqs.size() != 0 || !dp_state.decoding_reqs.empty() || !in_plan_ops.stop_req_ids.empty() ||
        !in_plan_ops.recompute_req_ids.empty()) {
      continue;
    }
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
    if (recompute_candidate == nullptr) {
      continue;
    }
    // Recompute only. Swapout is not supported for chunk-prefilled requests.
    EmitEventOnce(ScheduleBackendEventType::kRecomputeRequest, dp_idx, recompute_candidate, agg.last_failed_status,
                  events, emitted_event_keys);
    in_plan_ops.recompute_req_ids.insert(recompute_candidate->req_id);
  }
}

void ContinuousBatchingBackend::ScanAbortedPreprocessingReqs(const ScheduleContext& context,
                                                             const PlanningBatch& planning_batch,
                                                             std::vector<ScheduleBackendEvent>& events,
                                                             std::unordered_set<uint64_t>& emitted_event_keys) {
  for (size_t dp_idx = 0; dp_idx < planning_batch.dp_batches.size(); ++dp_idx) {
    const DpScheduleState& dp_state = context.GetDpScheduleState(context.GetCurrentMultiBatchId(), dp_idx);
    DpScheduleOpsInPlan& in_plan_ops = context.GetMutableInPlanScheduleOps(context.GetCurrentMultiBatchId(), dp_idx);
    for (const auto& req_kv : dp_state.preprocessing_reqs) {
      const std::shared_ptr<InferRequest>& req = req_kv.second;
      if (req == nullptr || req->IsStopped() || !req->aborted) {
        continue;
      }
      // pd_v2 WAITING_KVCACHE: RDMA may be in flight into this req's blocks.
      // Defer Stop until PdV2DecodeHook sees OnPrefillComplete.
      if (req->pd_v2.awaiting_prefill) {
        if (!req->pd_v2.pending_stop) {
          req->pd_v2.pending_stop = true;
          req->finish_status = Status(RET_REQUEST_TERMINATED, "client aborted while awaiting pd_v2 prefill");
          KLLM_LOG_WARNING << "client aborted while awaiting prefill, defer stop: req_id=" << req->req_id
                           << " dp_idx=" << dp_idx;
        }
        continue;
      }
      if (in_plan_ops.stop_req_ids.count(req->req_id) > 0) {
        continue;
      }
      KLLM_LOG_WARNING << "client aborted in preprocessing queue, emit stop: req_id=" << req->req_id
                       << " dp_idx=" << dp_idx;
      EmitEventOnce(ScheduleBackendEventType::kStopRequest, dp_idx, req,
                    Status(RET_REQUEST_TERMINATED, "client aborted"), events, emitted_event_keys);
      in_plan_ops.stop_req_ids.insert(req->req_id);
    }
  }
}

}  // namespace ksana_llm
