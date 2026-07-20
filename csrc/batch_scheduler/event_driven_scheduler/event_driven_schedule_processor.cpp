/* Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * 事件驱动调度器：单主线程 + 事件队列实现。
 * ==============================================================================*/

#include "batch_scheduler/event_driven_scheduler/event_driven_schedule_processor.h"

#include <algorithm>
#include <unordered_set>

#include "batch_scheduler/event_driven_scheduler/continuous_batching_backend.h"
#include "configure/environment.h"
#include "profiler/profile_event.h"
#include "profiler/timer.h"
#include "runtime/schedule_output.h"
#include "utils/logger.h"

namespace ksana_llm {

namespace {

bool HasReqInQueue(const std::vector<std::shared_ptr<InferRequest>>& queue, int64_t req_id) {
  for (const auto& req : queue) {
    if (req != nullptr && req->req_id == req_id) {
      return true;
    }
  }
  return false;
}

void RemoveReqByIdFromQueue(std::vector<std::shared_ptr<InferRequest>>& queue, int64_t req_id) {
  queue.erase(std::remove_if(queue.begin(), queue.end(),
                             [req_id](const std::shared_ptr<InferRequest>& req) {
                               return req != nullptr && req->req_id == req_id;
                             }),
              queue.end());
}

}  // namespace

void EventDrivenScheduleProcessor::HandleStopRequest(const std::shared_ptr<InferRequest>& req, Status ret_status,
                                                     size_t dp_idx) {
  if (req == nullptr) {
    KLLM_LOG_FATAL << "req is nullptr";
    return;
  }
  // Repeat Stop events: request already stopped and cleaned up, nothing to do.
  if (req->IsStopped()) {
    KLLM_LOG_SCHEDULER << "HandleStopRequest: req_id=" << req->req_id << " already stopped, skip";
    return;
  }
  KLLM_LOG_SCHEDULER << "HandleStopRequest: req_id=" << req->req_id << " dp_idx=" << dp_idx;

  // Look up which PP batch this request belongs to.
  size_t multi_batch_id = LookupReqBatchIdx(req->req_id, "HandleStopRequest");
  DpScheduleState& dp_state = schedule_state_.schedule_states[multi_batch_id][dp_idx];

  // Stop request is now actually applied, remove it from in-plan ops.
  // Also clear recompute_req_ids in case this request had a pending recompute that was delayed:
  // the delayed HandleRecomputeRequest will see IsStopped()=true and return early without
  // clearing recompute_req_ids, which would permanently block future OOM-triggered recomputes.
  dp_state.in_plan_schedule_ops.stop_req_ids.erase(req->req_id);
  dp_state.in_plan_schedule_ops.recompute_req_ids.erase(req->req_id);

  req->Stop();

  // Release blocks and notify clients.
  req->finished = true;
  req->finish_status = ret_status;
  StampOnce(req->finish_time_us);
  if (!global_cache_put_tracker_ || !global_cache_put_tracker_->DeferDestructionIfNeeded(req->req_id, dp_idx)) {
    dp_cache_managers_[dp_idx]->DestroyFinishedRequest(req->req_id);
  }
  req->kv_cache_blocks.clear();
  req->Notify();

  // Record stopped reqs for workers
  dp_state.schedule_ops_for_nodes.stopped_req_ids.push_back(req->req_id);

  // pd_v2 Decode: 该 req 彻底离场, 从 GlobalCacheQueryHook 的 in-flight 集合移除, 腾出 admission 名额。
  if (schedule_hook_manager_) {
    schedule_hook_manager_->OnEvent(HookEvent(HookEventType::kRequestExited, req->req_id));
  }

  // Remove from global req index and batch index when stop is actually applied.
  schedule_state_.req_id_to_req_ptr.erase(req->req_id);
  schedule_state_.req_to_batch_idx.erase(req->req_id);
}

void EventDrivenScheduleProcessor::HandleHookFailAndEnd(const std::shared_ptr<InferRequest>& req, size_t dp_idx,
                                                        const std::string& reason) {
  if (req == nullptr) {
    KLLM_LOG_ERROR << "HandleHookFailAndEnd: req is nullptr";
    return;
  }
  // Already stopped — nothing to clean up.
  if (req->IsStopped()) {
    KLLM_LOG_SCHEDULER << "HandleHookFailAndEnd: req_id=" << req->req_id << " already stopped, skip";
    return;
  }

  // Look up which PP batch this request belongs to.
  size_t multi_batch_id = LookupReqBatchIdx(req->req_id, "HandleHookFailAndEnd");
  DpScheduleState& dp_state = schedule_state_.schedule_states[multi_batch_id][dp_idx];
  // Capture req_id up front: the final req_id_to_req_ptr.erase() below drops
  // the map's shared_ptr (often the last owner), which would dangle `req`
  // (a reference into that map slot). Use the local for all bookkeeping and
  // the trailing log so nothing dereferences `req` after the erase.
  const int64_t req_id = req->req_id;
  dp_state.preprocessing_reqs.erase(req_id);
  RemoveReqByIdFromQueue(dp_state.prefilling_reqs, req_id);
  RemoveReqByIdFromQueue(dp_state.decoding_reqs, req_id);

  req->Stop();
  // 请求可能已被 GlobalCacheFetcher 拒绝并 Notify，需避免重复通知
  bool already_finished = req->finished;
  req->finished = true;
  // Preserve a more specific terminal status if a deferred-stop set-point
  // already recorded one (RET_REQUEST_TIMEOUT on waiting-timeout,
  // RET_REQUEST_TERMINATED on client abort); only fall back to the generic
  // hook-failure reason for genuine hook failures.
  if (req->finish_status.OK()) {
    req->finish_status = Status(RET_RUNTIME_FAILED, reason);
  }
  StampOnce(req->finish_time_us);
  if (!global_cache_put_tracker_ || !global_cache_put_tracker_->DeferDestructionIfNeeded(req_id, dp_idx)) {
    dp_cache_managers_[dp_idx]->DestroyFinishedRequest(req_id);
  }
  req->kv_cache_blocks.clear();
  if (!already_finished) {
    req->Notify();
  }
  // pd_v2 Decode: 该 req 彻底离场, 从 GlobalCacheQueryHook 的 in-flight 集合移除。
  if (schedule_hook_manager_) {
    schedule_hook_manager_->OnEvent(HookEvent(HookEventType::kRequestExited, req_id));
  }
  schedule_state_.req_id_to_req_ptr.erase(req_id);
  schedule_state_.req_to_batch_idx.erase(req_id);
  KLLM_LOG_WARNING << "Hook fail-and-end req_id=" << req_id << ", reason=" << reason;
}

void EventDrivenScheduleProcessor::HandleRecomputeRequest(const std::shared_ptr<InferRequest>& req, Status ret_status,
                                                          size_t dp_idx) {
  if (req == nullptr) {
    KLLM_LOG_ERROR << "req is nullptr";
    return;
  }

  if (req->IsStopped()) {
    KLLM_LOG_SCHEDULER << "req_id=" << req->req_id << " is stopped, drop recompute event";
    return;
  }

  // Look up which PP batch this request belongs to.
  size_t multi_batch_id = LookupReqBatchIdx(req->req_id, "HandleRecomputeRequest");
  DpScheduleState& dp_state = schedule_state_.schedule_states[multi_batch_id][dp_idx];

  // Recompute request is now actually applied, remove it from in-plan ops.
  dp_state.in_plan_schedule_ops.recompute_req_ids.erase(req->req_id);

  // pd_v2 Decode-side: a kv_from_remote request cannot be re-prefilled locally
  // (it would overwrite peer-written KV with a divergent attention path). With
  // swap removed, the only safe preempt path is to release the local KV and
  // re-fetch from a peer Prefill via a full restart through AddInferRequest.
  //
  // pd_v2 Decode 还必须覆盖 kLocal (kv_from_remote=false) 请求：PD decode 下 KV block
  // 分配的唯一所有者是 GlobalCacheQueryHook。若走下面的非 PD 分支把请求直接 push 回
  // prefilling_reqs，它会绕过 hook、空 kv_cache_blocks 滞留在 prefilling_reqs（backend
  // 不是 PD 的分配者，且会被 per-dp step 配额挡住），导致永久 kWaiting / 引擎空转。
  // 因此 PD decode 下所有 recompute 都必须经 HandlePdV2DecodeRestart 重投 AddRequest，
  // 让 hook 链重新分配 block。非 PD 仍走下面的 backend 直接回灌（backend 即分配者，正确）。
  if (req->pd_v2.kv_from_remote || pd_v2_decode_mode_) {
    HandlePdV2DecodeRestart(req, dp_idx);
    return;
  }

  req->ResetInflightTask();
  req->ResetPlanningTask();
  req->Restart();
  req->ResetPrefillingTokens();
  if (!global_cache_put_tracker_ || !global_cache_put_tracker_->DeferDestructionIfNeeded(req->req_id, dp_idx)) {
    dp_cache_managers_[dp_idx]->DestroyFinishedRequest(req->req_id);
  }
  req->kv_cache_blocks.clear();
  req->accepted_tokens.clear();
  req->generated_tokens.clear();
  req->draft_tokens.clear();
  req->forwarding_tokens.clear();

  // pd_v2 Decode reqs 都是 kv_from_remote, 走上面的 HandlePdV2DecodeRestart, 不会到这里;
  // 此处仅为防御性保持 in-flight 集合一致 (非 pd_v2 路径 defer_admission_on_block_full_=false,
  // kRequestExited 事件为 no-op)。
  if (schedule_hook_manager_) {
    schedule_hook_manager_->OnEvent(HookEvent(HookEventType::kRequestExited, req->req_id));
  }

  if (!HasReqInQueue(dp_state.prefilling_reqs, req->req_id)) {
    dp_state.prefilling_reqs.push_back(req);
  }

  KLLM_LOG_SCHEDULER << "After recompute req_id=" << req->req_id << ". dp_idx=" << dp_idx
                     << ", decoding_reqs=" << dp_state.decoding_reqs.size()
                     << ", prefilling_reqs=" << dp_state.prefilling_reqs.size();

  // Record recomputed reqs for workers
  dp_state.schedule_ops_for_nodes.recomputed_req_ids.push_back(req->req_id);
}

size_t EventDrivenScheduleProcessor::LookupReqBatchIdx(int64_t req_id, const char* caller) const {
  auto it = schedule_state_.req_to_batch_idx.find(req_id);
  if (it == schedule_state_.req_to_batch_idx.end()) {
    KLLM_THROW(FormatStr("%s: req_id=%ld not found in req_to_batch_idx_, state machine bug", caller, req_id));
  }
  return it->second;
}

void EventDrivenScheduleProcessor::HandlePdV2DecodeRestart(const std::shared_ptr<InferRequest>& req, size_t dp_idx) {
  size_t multi_batch_id = LookupReqBatchIdx(req->req_id, "HandlePdV2DecodeRestart");
  DpScheduleState& dp_state = schedule_state_.schedule_states[multi_batch_id][dp_idx];

  // (a) Release the local KV blocks held by this Decode-side request and tell
  // workers to free their copy (recomputed_req_ids). The peer-written KV is
  // discarded entirely — we will re-fetch it from a peer Prefill after restart.
  if (!global_cache_put_tracker_ || !global_cache_put_tracker_->DeferDestructionIfNeeded(req->req_id, dp_idx)) {
    dp_cache_managers_[dp_idx]->DestroyFinishedRequest(req->req_id);
  }
  req->kv_cache_blocks.clear();
  req->swa_kv_cache_blocks.clear();
  req->swa_kv_cache_block_idx_offsets.clear();
  dp_state.schedule_ops_for_nodes.recomputed_req_ids.push_back(req->req_id);

  // pd_v2 Decode 抢占重启: 本地 KV 已释放, 先把该 req 从 in-flight 集合移除; 下面重新走
  // OnEventAddRequest → GlobalCacheQueryHook 准入时会重新计入 (insert 幂等), net 一致。
  if (schedule_hook_manager_) {
    schedule_hook_manager_->OnEvent(HookEvent(HookEventType::kRequestExited, req->req_id));
  }

  // (b) Reset scheduling / PD state to the pristine "pre-hook" state, but KEEP
  // output_tokens (already-generated tokens). OnEventAddRequest will run
  // ResetPrefillingTokens so prefilling_tokens_ = output_tokens (the
  // accumulated sequence) — the whole hook chain (GlobalCacheQueryHook +
  // PdV2DecodeHook) then operates on that accumulated sequence.
  req->ResetInflightTask();
  req->ResetPlanningTask();
  req->Restart();
  req->pd_v2.kv_from_remote = false;
  req->pd_v2.awaiting_prefill = false;
  req->pd_v2.pending_stop = false;
  req->infer_stage = InferStage::kContext;
  req->kv_cached_token_num = 0;
  req->prefix_cache_len = 0;
  req->need_recreate_generator = true;
  req->pd_v2.restart_resubmit = true;
  {
    std::lock_guard<std::mutex> ol(req->output_mutex);
    req->draft_tokens.clear();
  }

  KLLM_LOG_SCHEDULER << "pd_v2 recompute restart: req_id=" << req->req_id << " dp_idx=" << dp_idx
                     << ", output_tokens=" << req->output_tokens.size() << ", re-injecting as AddRequest event";

  // (c) Re-execute from the very beginning of the schedule pipeline. We are
  // already on the EDS thread holding infer_request_mutex_, so we must NOT
  // call the public AddInferRequest here — it re-locks the same non-recursive
  // mutex and would deadlock. Instead re-register the global index directly
  // (already under the lock) and post an EventAddRequest; the loop's inline
  // drain (or the next iteration) runs OnEventAddRequest, which re-runs the
  // full hook chain (GlobalCacheQueryHook + PdV2DecodeHook) on the accumulated
  // sequence — exactly the AddInferRequest entry path the caller asked for.
  schedule_state_.req_id_to_req_ptr[req->req_id] = req;
  auto info = std::make_shared<EventAddRequestInfo>();
  info->reqs.push_back(req);
  event_queue_.Put(SchedEvent(std::move(info)));
}

bool EventDrivenScheduleProcessor::ApplyBackendEventsToState(const std::vector<ScheduleBackendEvent>& backend_events) {
  if (backend_events.empty()) {
    return false;
  }
  bool has_resource_released = false;
  KLLM_LOG_SCHEDULER << "ApplyBackendEventsToState: " << backend_events.size() << " events";
  auto remove_from_state_and_batches = [this](size_t dp_idx, int64_t req_id) {
    size_t multi_batch_id = LookupReqBatchIdx(req_id, "remove_from_state_and_batches");
    DpScheduleState& dp_state = schedule_state_.schedule_states[multi_batch_id][dp_idx];
    dp_state.preprocessing_reqs.erase(req_id);
    RemoveReqByIdFromQueue(dp_state.prefilling_reqs, req_id);
    RemoveReqByIdFromQueue(dp_state.decoding_reqs, req_id);

    // Remove from all planning batches.
    for (const auto& pb : planning_batches_) {
      if (pb == nullptr) {
        continue;
      }
      if (dp_idx < pb->dp_batches.size()) {
        pb->RemoveRunningReq(dp_idx, req_id);
      }
    }
    // NOTE: req_to_batch_idx_ is NOT cleared here. It is cleared by HandleStopRequest/HandleRecomputeRequest
    // after the delayed event is actually applied.
  };

  KLLM_LOG_DEBUG << "ApplyBackendEventsToState: " << backend_events.size() << " events";
  for (const auto& ev : backend_events) {
    const size_t dp_idx = ev.dp_idx;
    std::shared_ptr<InferRequest> req = ev.req_ptr;
    if (req == nullptr) {
      KLLM_LOG_ERROR << "Invalid backend event: null req_ptr, dp_idx=" << dp_idx;
      continue;
    }
    const int64_t req_id = req->req_id;

    if (ev.type == ScheduleBackendEventType::kMoveToDecode) {
      // Move request from prefilling queue to decoding queue in its assigned batch.
      size_t multi_batch_id = LookupReqBatchIdx(req_id, "kMoveToDecode");
      DpScheduleState& dp_state = schedule_state_.schedule_states[multi_batch_id][dp_idx];
      RemoveReqByIdFromQueue(dp_state.prefilling_reqs, req_id);
      if (!HasReqInQueue(dp_state.decoding_reqs, req_id)) {
        dp_state.decoding_reqs.push_back(req);
      }
      continue;
    }

    // Stop/Recompute 的请求级操作
    // 立即从状态与 batch 中摘除，避免后续调度继续使用。
    KLLM_LOG_DEBUG << "ApplyBackendEvents: remove_from_state req_id=" << req_id
                   << " type=" << ScheduleBackendEventType2Str(ev.type) << " has_inflight=" << req->HasInflightTask()
                   << " dp_idx=" << dp_idx;
    remove_from_state_and_batches(dp_idx, req_id);

    // Stop/Recompute 的请求级操作延迟到无 in-flight task 时执行。
    if (req->HasInflightTask()) {
      KLLM_LOG_SCHEDULER << "Delaying " << (ev.type == ScheduleBackendEventType::kStopRequest ? "Stop" : "Recompute")
                         << " event for req_id=" << req_id << " dp_idx=" << dp_idx
                         << " (HasInflightTask=true, inflight_count=" << req->GetInflightCount() << ")";
      ScheduleBackendEvent delayed_ev;
      delayed_ev.type = ev.type;
      delayed_ev.dp_idx = dp_idx;
      delayed_ev.req_ptr = req;
      delayed_ev.status = ev.status;
      delayed_backend_events_.push_back(std::move(delayed_ev));
    } else if (ev.type == ScheduleBackendEventType::kStopRequest) {
      HandleStopRequest(req, ev.status, dp_idx);
      has_resource_released = true;
    } else if (ev.type == ScheduleBackendEventType::kRecomputeRequest) {
      HandleRecomputeRequest(req, ev.status, dp_idx);
      has_resource_released = true;
    }
  }
  // pd_v2: KV blocks may have just been freed (Stop / Recompute / Swapout).
  // Wake hooks that hold deferred-admission queues so they retry alloc on
  // the EDS thread (via async_callback_ → event_queue_ → next loop turn →
  // UpdateAsyncResult). Currently only GlobalCacheQueryHook (pd_v2 Decode role)
  // implements this; others use the default no-op. The hook itself
  // short-circuits when its queue is empty, so the call is cheap on V1-only paths.
  if (has_resource_released && schedule_hook_manager_) {
    schedule_hook_manager_->OnEvent(HookEvent(HookEventType::kBlocksFreed));
  }
  return has_resource_released;
}

bool EventDrivenScheduleProcessor::ProcessDelayedBackendEvents() {
  if (delayed_backend_events_.empty()) {
    return false;
  }
  bool has_resource_released = false;
  KLLM_LOG_SCHEDULER << "ProcessDelayedBackendEvents: " << delayed_backend_events_.size() << " delayed events";
  for (size_t i = 0; i < delayed_backend_events_.size();) {
    ScheduleBackendEvent& ev = delayed_backend_events_[i];
    if (ev.req_ptr == nullptr) {
      KLLM_LOG_ERROR << "Invalid backend event: null req_ptr, dp_idx=" << ev.dp_idx;
      delayed_backend_events_.erase(delayed_backend_events_.begin() + static_cast<ptrdiff_t>(i));
      continue;
    }

    std::shared_ptr<InferRequest> req = ev.req_ptr;
    if (req->HasInflightTask()) {
      ++i;
      continue;
    }

    const size_t dp_idx = ev.dp_idx;
    if (ev.type == ScheduleBackendEventType::kStopRequest) {
      HandleStopRequest(req, ev.status, dp_idx);
      has_resource_released = true;
    } else if (ev.type == ScheduleBackendEventType::kRecomputeRequest) {
      HandleRecomputeRequest(req, ev.status, dp_idx);
      has_resource_released = true;
    }

    delayed_backend_events_.erase(delayed_backend_events_.begin() + static_cast<ptrdiff_t>(i));
  }
  if (has_resource_released && schedule_hook_manager_) {
    schedule_hook_manager_->OnEvent(HookEvent(HookEventType::kBlocksFreed));
  }
  return has_resource_released;
}

EventDrivenScheduleProcessor::EventDrivenScheduleProcessor(size_t dp_num, size_t max_pp_batch_num)
    : max_pp_batch_num_(max_pp_batch_num), dp_num_(dp_num) {}

EventDrivenScheduleProcessor::~EventDrivenScheduleProcessor() { Stop(); }

void EventDrivenScheduleProcessor::Initialize(std::shared_ptr<BatchSchedulerInterface> batch_scheduler) {
  batch_scheduler_ = batch_scheduler;  // 满足基类 ScheduleProcessorInterface 的接口约定

  schedule_state_.schedule_states.clear();
  schedule_state_.schedule_states.resize(max_pp_batch_num_);
  for (size_t bid = 0; bid < max_pp_batch_num_; ++bid) {
    schedule_state_.schedule_states[bid].resize(dp_num_);
  }

  // Initialize reusable data structures once and reuse them afterwards.
  planning_batches_.assign(max_pp_batch_num_, nullptr);
  for (size_t batch_idx = 0; batch_idx < max_pp_batch_num_; ++batch_idx) {
    planning_batches_[batch_idx] = std::make_shared<PlanningBatch>();
    planning_batches_[batch_idx]->batch_idx = static_cast<int>(batch_idx);
    planning_batches_[batch_idx]->dp_batches.assign(dp_num_, DpBatch{});
  }
  schedule_context_.SetScheduleState(&schedule_state_);
  schedule_context_.SetDpCacheManagers(&dp_cache_managers_);

  schedule_hook_manager_->SetAsyncTaskFinishedCallback(
      [this](const EventAsyncTaskFinishedInfo& info) { EnqueueAsyncTaskFinished(info); });
  KLLM_LOG_INFO << "EventDrivenScheduleProcessor initialized, max_pp_batch_num=" << max_pp_batch_num_;
}

Status EventDrivenScheduleProcessor::AddInferRequest(std::vector<std::shared_ptr<InferRequest>>& infer_request_group) {
  // Maintain global req_id -> req_ptr index as early as possible.
  // This is required because async hook next_steps may only contain req_id.
  {
    std::lock_guard<std::mutex> guard(infer_request_mutex_);
    for (const auto& req : infer_request_group) {
      if (req == nullptr) {
        continue;
      }
      schedule_state_.req_id_to_req_ptr[req->req_id] = req;
    }
  }
  KLLM_LOG_SCHEDULER << infer_request_group.size() << " requests, req_ids=" << RequestVector2Str(infer_request_group);

  auto info = std::make_shared<EventAddRequestInfo>();
  info->reqs = infer_request_group;
  SchedEvent ev(std::move(info));
  event_queue_.Put(std::move(ev));
  return Status();
}

std::shared_ptr<ScheduleResult> EventDrivenScheduleProcessor::GetNextScheduleResult() {
  SchedEvent ev(std::make_shared<EventGetScheduleResultInfo>());
  event_queue_.Put(std::move(ev));
  auto ret = result_queue_.Get();
  return ret;
}

void EventDrivenScheduleProcessor::UpdateWithGenerationResult(const GenerationOutputGroup& generation_output) {
  auto info = std::make_shared<EventUpdateAfterGenerationInfo>();
  info->gen_results = generation_output;
  SchedEvent ev(std::move(info));
  event_queue_.Put(std::move(ev));
}

void EventDrivenScheduleProcessor::Lock() { infer_request_mutex_.lock(); }

void EventDrivenScheduleProcessor::Unlock() { infer_request_mutex_.unlock(); }

void EventDrivenScheduleProcessor::EnqueueAsyncTaskFinished(const EventAsyncTaskFinishedInfo& event_info) {
  if (event_info.task_results.empty()) {
    return;
  }
  auto info = std::make_shared<EventAsyncTaskFinishedInfo>();
  info->task_results = event_info.task_results;
  SchedEvent ev(std::move(info));
  event_queue_.Put(std::move(ev));
}

void EventDrivenScheduleProcessor::UpdateAfterAsyncTaskFinished(const EventAsyncTaskFinishedInfo& event_info) {
  if (event_info.task_results.empty()) {
    return;
  }
  KLLM_LOG_DEBUG << "UpdateAfterAsyncTaskFinished: task_results.size=" << event_info.task_results.size();
  ReqIdToNextStep next_steps = schedule_hook_manager_->UpdateAsyncResult(event_info.task_results, schedule_context_);
  KLLM_LOG_DEBUG << "UpdateAfterAsyncTaskFinished: next_steps.size=" << next_steps.size();

  // next_steps only provides req_id -> NextStep. We must locate req_ptr first, then route by attn_dp_group_id.
  for (const auto& ns_kv : next_steps) {
    const int64_t req_id = ns_kv.first;
    const NextStep& step = ns_kv.second;
    auto req_it = schedule_state_.req_id_to_req_ptr.find(req_id);
    if (req_it == schedule_state_.req_id_to_req_ptr.end() || req_it->second == nullptr) {
      KLLM_LOG_FATAL << "Async next_steps contains unknown req_id=" << req_id;
      continue;
    }
    const std::shared_ptr<InferRequest>& req_ptr = req_it->second;

    const size_t dp_idx = static_cast<size_t>(req_ptr->attn_dp_group_id);

    // Look up which PP batch this request belongs to.
    size_t multi_batch_id = LookupReqBatchIdx(req_id, "UpdateAfterAsyncTaskFinished");
    DpScheduleState& dp_state = schedule_state_.schedule_states[multi_batch_id][dp_idx];

    KLLM_LOG_DEBUG << "UpdateAfterAsyncTaskFinished: req_id=" << req_id << " action=" << static_cast<int>(step.action)
                   << " infer_stage=" << static_cast<int>(req_ptr->infer_stage) << " dp_idx=" << dp_idx
                   << " preprocessing=" << dp_state.preprocessing_reqs.size()
                   << " prefilling=" << dp_state.prefilling_reqs.size()
                   << " decoding=" << dp_state.decoding_reqs.size();
    switch (step.action) {
      case NextStepAction::kContinue: {
        // Move from preprocessing -> prefilling (or decoding for pd_v2).
        auto pre_it = dp_state.preprocessing_reqs.find(req_id);
        if (pre_it != dp_state.preprocessing_reqs.end()) {
          // Step-token backpressure is enforced at batch-formation time by the
          // backend's WorkloadBalancer (IsRequestAcceptable), so no admission
          // gate is needed here — route straight into the active queues.
          if (req_ptr->infer_stage == InferStage::kDecode) {
            dp_state.decoding_reqs.push_back(pre_it->second);
            dp_state.preprocessing_reqs.erase(pre_it);
            KLLM_LOG_DEBUG << "UpdateAfterAsyncTaskFinished: req_id=" << req_id << " moved preprocessing -> decoding";
          } else {
            dp_state.prefilling_reqs.push_back(pre_it->second);
            dp_state.preprocessing_reqs.erase(pre_it);
            KLLM_LOG_DEBUG << "UpdateAfterAsyncTaskFinished: req_id=" << req_id
                           << " moved preprocessing -> prefilling (infer_stage="
                           << static_cast<int>(req_ptr->infer_stage) << ")";
          }
        } else {
          KLLM_LOG_ERROR << "UpdateAfterAsyncTaskFinished: req_id=" << req_id
                         << " kContinue but NOT in preprocessing_reqs! ORPHANED."
                         << " dp_idx=" << dp_idx << " preprocessing=" << dp_state.preprocessing_reqs.size()
                         << " prefilling=" << dp_state.prefilling_reqs.size()
                         << " decoding=" << dp_state.decoding_reqs.size()
                         << " req_id_to_req_ptr=" << schedule_state_.req_id_to_req_ptr.size();
        }
        break;
      }
      case NextStepAction::kFailAndEnd: {
        KLLM_LOG_INFO << "UpdateAfterAsyncTaskFinished: req_id=" << req_id << " kFailAndEnd";
        HandleHookFailAndEnd(req_ptr, dp_idx, "hook async processing failed");
        break;
      }
      case NextStepAction::kWaitAsyncResult:
      default: {
        KLLM_LOG_DEBUG << "UpdateAfterAsyncTaskFinished: req_id=" << req_id << " kWaitAsyncResult, still waiting";
        break;
      }
    }
  }
}

void EventDrivenScheduleProcessor::Start() {
  KLLM_CHECK_WITH_INFO(backend_ != nullptr, "backend_ must be set before Start()");
  KLLM_CHECK_WITH_INFO(schedule_hook_manager_ != nullptr, "schedule_hook_manager_ must be set before Start()");
  KLLM_CHECK_WITH_INFO(!dp_cache_managers_.empty(), "cache managers must be set before Start()");

  terminated_ = false;
  sched_thread_ = std::thread(&EventDrivenScheduleProcessor::AsyncSchedulerLoop, this);
  KLLM_LOG_INFO << "EventDrivenScheduleProcessor started, dp_num=" << dp_num_
                << ", max_pp_batch_num=" << max_pp_batch_num_;
}

void EventDrivenScheduleProcessor::Stop() {
  terminated_ = true;
  if (backend_) {
    backend_->Stop();
  }
  event_queue_.Put(SchedEvent{});  // 唤醒可能阻塞的 Get
  if (sched_thread_.joinable()) {
    sched_thread_.join();
  }
  result_queue_.Put(nullptr);
  KLLM_LOG_INFO << "EventDrivenScheduleProcessor stopped";
}

void EventDrivenScheduleProcessor::AsyncSchedulerLoop() {
  KLLM_LOG_SCHEDULER << "started";
  // Publish this thread's id so GetLocalLoad() can detect re-entry from the
  // scheduler thread (which holds infer_request_mutex_ during DispatchEvent)
  // and avoid self-deadlocking on the non-recursive mutex.
  async_loop_tid_.store(std::this_thread::get_id(), std::memory_order_relaxed);
  // When a consumer (ModelDriver) is parked on result_queue_.Get() but the last
  // TryBuildAndLaunchScheduleResult could not launch (IsLaunchable()==false), the
  // launch is only re-attempted on the next incoming event (see the pending-retry
  // block below). If the gating condition clears WITHOUT enqueuing an event, the
  // loop would block on event_queue_.Get() forever while decoding_reqs stays
  // non-empty -> GPU idles, clients hang. Poll while a pull is pending so the
  // scheduler always makes forward progress.
  constexpr size_t kPendingSchedulePollMs = 50;
  while (!terminated_) {
    SchedEvent event;
    {
      PROFILE_EVENT_SCOPE(sched_loop_wait, "Engine.AsyncSchedulerLoop.Wait");
      event = pending_get_schedule_result_ ? event_queue_.Get(kPendingSchedulePollMs) : event_queue_.Get();
    }
    if (terminated_) {
      break;
    }
    if (event.IsEmpty()) {
      // Empty event: either Stop() wakeup or a pending-schedule poll timeout.
      // On timeout, re-run scheduling and re-attempt the gated launch so the
      // engine recovers as soon as IsLaunchable() turns true.
      if (!terminated_ && pending_get_schedule_result_) {
        std::lock_guard<std::mutex> guard(infer_request_mutex_);
        RunScheduleForBatches();
        if (TryBuildAndLaunchScheduleResult(schedule_context_)) {
          RunScheduleForBatches();
        }
      }
      continue;  // 空事件仅用于 Stop() 时唤醒 Get() 或 pending 轮询超时
    }
    PROFILE_EVENT_SCOPE(sched_loop_iter, "Engine.AsyncSchedulerLoop.iter");
    {
      std::lock_guard<std::mutex> guard(infer_request_mutex_);
      DispatchEvent(event);

      while (true) {
        SchedEvent next = event_queue_.NonBlockingGet();
        if (next.IsEmpty()) {
          break;
        }
        DispatchEvent(next);
      }
      if (terminated_) {
        break;
      }

      RunScheduleForBatches();
    }
    if (pending_get_schedule_result_) {
      PROFILE_EVENT_SCOPE(sched_pending_retry, "Engine.AsyncSchedulerLoop.PendingRetry");
      std::lock_guard<std::mutex> guard(infer_request_mutex_);
      if (TryBuildAndLaunchScheduleResult(schedule_context_)) {
        // Prepare next schedule result in advance after current schedule result is launched
        RunScheduleForBatches();
      }
    }
  }
  KLLM_LOG_SCHEDULER << "thread terminated";
}

void EventDrivenScheduleProcessor::DispatchEvent(SchedEvent& event) {
  PROFILE_EVENT_SCOPE(dispatch_event, "Engine.DispatchEvent");
  KLLM_LOG_SCHEDULER << "EventType: " << SchedEventType2Str(event.GetType());
  switch (event.GetType()) {
    case SchedEventType::EventAddRequest: {
      auto info = std::dynamic_pointer_cast<EventAddRequestInfo>(event.event_info);
      if (info == nullptr) {
        KLLM_LOG_FATAL << "Event type/payload mismatch for EventAddRequest";
        return;
      }
      OnEventAddRequest(info);
      break;
    }
    case SchedEventType::EventUpdateAfterGeneration: {
      auto info = std::dynamic_pointer_cast<EventUpdateAfterGenerationInfo>(event.event_info);
      if (info == nullptr) {
        KLLM_LOG_FATAL << "Event type/payload mismatch for EventUpdateAfterGeneration";
        return;
      }
      OnEventUpdateAfterGeneration(info);
      break;
    }
    case SchedEventType::EventAsyncTaskFinished: {
      auto info = std::dynamic_pointer_cast<EventAsyncTaskFinishedInfo>(event.event_info);
      if (info == nullptr) {
        KLLM_LOG_FATAL << "Event type/payload mismatch for EventAsyncTaskFinished";
        return;
      }
      OnEventAsyncTaskFinished(info);
      break;
    }
    case SchedEventType::EventGetScheduleResult: {
      OnEventGetScheduleResult();
      break;
    }
  }
}

void EventDrivenScheduleProcessor::OnEventGetScheduleResult() {
  PROFILE_EVENT_SCOPE(on_get_schedule_result, "Engine.OnEventGetScheduleResult");
  if (TryBuildAndLaunchScheduleResult(schedule_context_)) {
    pending_get_schedule_result_ = false;
  } else {
    pending_get_schedule_result_ = true;
  }
}

void EventDrivenScheduleProcessor::OnEventAddRequest(const std::shared_ptr<EventAddRequestInfo>& info) {
  if (!info || info->reqs.empty()) {
    return;
  }
  KLLM_LOG_SCHEDULER << " req num:" << info->reqs.size();

  for (auto& req : info->reqs) {
    // pd_v2 recompute restart re-injects an in-flight req to re-fetch KV from a
    // peer Prefill. We must keep its accumulated output_tokens so the restart
    // re-prefills [prompt + already-generated] and no progress is lost; only
    // fresh requests reset output_tokens back to the prompt.
    if (!req->pd_v2.restart_resubmit) {
      req->output_tokens = req->input_tokens;
    }
    req->ResetPrefillingTokens();
    req->pd_v2.restart_resubmit = false;
  }

  std::vector<std::shared_ptr<InferRequest>> requests = info->reqs;
  if (workload_balancer_ != nullptr) {
    workload_balancer_->AddInferRequests(requests, schedule_context_);
  }

  // Distribute requests across PP batches via PpBatchDistributor.
  // Requests are placed into schedule_states_[multi_batch_id][dp_idx].prefilling_reqs
  // and req_to_batch_idx is populated.
  // IMPORTANT: once assigned, a request MUST NOT be moved to a different batch.
  // req_to_batch_idx is immutable after this point; all subsequent event handlers
  // (HandleStopRequest, HandleRecomputeRequest, kMoveToDecode, etc.) operate on
  // the same batch as determined here.
  if (pp_batch_distributor_ != nullptr) {
    pp_batch_distributor_->DistributeRequests(requests, max_pp_batch_num_, dp_num_, schedule_state_.schedule_states,
                                              schedule_state_.req_to_batch_idx);
  } else {
    // Without distributor (max_pp_batch_num == 1), all requests go to batch 0.
    // Populate req_to_batch_idx so event handlers (kMoveToDecode, remove_from_state_and_batches, etc.)
    // can rely on it unconditionally.
    for (const auto& req : requests) {
      if (req != nullptr) {
        schedule_state_.req_to_batch_idx[req->req_id] = 0;
      }
    }
  }

  ReqIdToNextStep next_steps = schedule_hook_manager_->Process(requests, schedule_context_);
  for (auto& req : info->reqs) {
    const size_t dp_idx = static_cast<size_t>(req->attn_dp_group_id);
    // Determine which PP batch this request was assigned to.
    // req_to_batch_idx is always populated: distributor sets it, or we default to 0 above.
    size_t multi_batch_id = 0;
    auto batch_it = schedule_state_.req_to_batch_idx.find(req->req_id);
    if (batch_it != schedule_state_.req_to_batch_idx.end()) {
      multi_batch_id = batch_it->second;
    }
    DpScheduleState& dp_state = schedule_state_.schedule_states[multi_batch_id][dp_idx];
    auto ns = next_steps.find(req->req_id);
    // pd_v2: KV already staged from peer Prefill and hook set infer_stage =
    // kDecode. EDS skips prefill and routes straight to decoding_reqs.
    const bool pd_v2_skip_prefill = req->infer_stage == InferStage::kDecode &&
                                    (ns == next_steps.end() || ns->second.action == NextStepAction::kContinue);
    if (pd_v2_skip_prefill) {
      // pd_v2 decode-only: KV already staged from peer Prefill, so this req
      // skips prefill and routes straight into decoding_reqs.
      KLLM_LOG_SCHEDULER << "req_id: " << req->req_id
                         << " pd_v2 infer_stage=kDecode, routed straight to decoding_reqs. dp_idx=" << dp_idx;
      dp_state.decoding_reqs.push_back(req);
      continue;
    }
    if (ns == next_steps.end()) {
      KLLM_LOG_WARNING << "req_id: " << req->req_id << " has no next_step, default continue";
      // Request was already placed in prefilling_reqs by distributor.
      // If no distributor, manually add to prefilling.
      if (pp_batch_distributor_ == nullptr) {
        dp_state.prefilling_reqs.push_back(req);
      }
      continue;
    }
    if (ns->second.action == NextStepAction::kContinue) {
      KLLM_LOG_SCHEDULER << "req_id: " << req->req_id << " continue, added to prefilling_reqs. dp_idx=" << dp_idx
                         << ", " << req->ScheduleStateToStr()
                         << ", stop_tokens=" << Vector2Str(req->sampling_config.stop_token_ids);
      // If distributor already placed it, nothing to do. Otherwise add manually.
      if (pp_batch_distributor_ == nullptr) {
        dp_state.prefilling_reqs.push_back(req);
      }
    } else if (ns->second.action == NextStepAction::kWaitAsyncResult) {
      // Req parks in preprocessing_reqs until its async hook task completes,
      // then transitions to prefilling/decoding via UpdateAfterAsyncTaskFinished.
      KLLM_LOG_SCHEDULER << "req_id: " << req->req_id << " wait async result, moved to preprocessing_reqs";
      // If distributor placed request in prefilling, move it to preprocessing.
      if (pp_batch_distributor_ != nullptr) {
        RemoveReqByIdFromQueue(dp_state.prefilling_reqs, req->req_id);
      }
      dp_state.preprocessing_reqs[req->req_id] = req;
    } else if (ns->second.action == NextStepAction::kFailAndEnd) {
      // If distributor placed request in prefilling, HandleHookFailAndEnd will clean it up.
      HandleHookFailAndEnd(req, dp_idx, "hook processing failed on add request");
    }
  }
}

void EventDrivenScheduleProcessor::OnEventUpdateAfterGeneration(
    const std::shared_ptr<EventUpdateAfterGenerationInfo>& info) {
  PROFILE_EVENT_SCOPE(on_update_after_generation, "Engine.OnEventUpdateAfterGeneration");
  size_t multi_batch_id = info->gen_results.multi_batch_id;
  std::vector<ScheduleBackendEvent> backend_events;
  backend_->UpdateWithGenerationResult(multi_batch_id, info->gen_results, schedule_context_, backend_events);

  ApplyBackendEventsToState(backend_events);

  // Some delayed events can be processed after inflight tasks are finished.
  ProcessDelayedBackendEvents();

  // Release one pipeline slot when a generation result comes back.
  if (pipeline_depth_ > 0) {
    --pipeline_depth_;
  }
  // Fold this step's wall time + token count into the per-step EWMA.
  RecordStepLatency();
}

void EventDrivenScheduleProcessor::OnEventAsyncTaskFinished(const std::shared_ptr<EventAsyncTaskFinishedInfo>& info) {
  if (!info || info->task_results.empty()) {
    return;
  }
  UpdateAfterAsyncTaskFinished(*info);
}

void EventDrivenScheduleProcessor::ReportEventDrivenState(const char* phase) const {
  KLLM_CHECK(backend_ != nullptr);
  const BatchSchedulerConfig& config = backend_->GetBatchSchedulerConfig();

  size_t total_preprocessing_reqs = 0;
  size_t total_prefilling_reqs = 0;
  size_t total_decoding_reqs = 0;
  size_t total_planning_running_reqs = 0;
  size_t total_planning_step_token_num = 0;
  size_t total_planning_step_logits_num = 0;

  for (size_t multi_batch_id = 0; multi_batch_id < max_pp_batch_num_; ++multi_batch_id) {
    const PlanningBatch& planning_batch = *planning_batches_[multi_batch_id];
    const bool launchable = planning_batch.IsLaunchable();
    for (size_t dp_idx = 0; dp_idx < dp_num_; ++dp_idx) {
      const DpScheduleState& dp_state = schedule_state_.schedule_states[multi_batch_id][dp_idx];
      // dp_batches is initialized with exactly dp_num_ entries in Initialize(), so dp_idx < dp_num_
      // guarantees dp_idx is within bounds.
      const DpBatch& dp_batch = planning_batch.dp_batches[dp_idx];
      KLLM_LOG_SCHEDULER << "EventDrivenState[" << phase << "]: multi_batch_id=" << multi_batch_id
                         << ", dp_idx=" << dp_idx
                         << ", preprocessing_reqs=" << dp_state.preprocessing_reqs.size()
                         << ", prefilling_reqs=" << dp_state.prefilling_reqs.size()
                         << ", decoding_reqs=" << dp_state.decoding_reqs.size()
                         << ", planning_running_reqs=" << dp_batch.running_reqs.size()
                         << ", planning_step_token_num=" << dp_batch.running_step_token_num
                         << ", planning_step_logits_num=" << dp_batch.running_step_logits_num
                         << ", max_batch_size=" << config.max_batch_size
                         << ", max_step_token_num=" << config.max_step_token_num
                         << ", max_waiting_queue_len=" << config.max_waiting_queue_len
                         << ", launchable=" << launchable;

      total_preprocessing_reqs += dp_state.preprocessing_reqs.size();
      total_prefilling_reqs += dp_state.prefilling_reqs.size();
      total_decoding_reqs += dp_state.decoding_reqs.size();
      total_planning_running_reqs += dp_batch.running_reqs.size();
      total_planning_step_token_num += dp_batch.running_step_token_num;
      total_planning_step_logits_num += dp_batch.running_step_logits_num;
    }
  }

  KLLM_LOG_SCHEDULER << "EventDrivenStateSummary[" << phase << "]: preprocessing_reqs="
                     << total_preprocessing_reqs << ", prefilling_reqs=" << total_prefilling_reqs
                     << ", decoding_reqs=" << total_decoding_reqs
                     << ", planning_running_reqs=" << total_planning_running_reqs
                     << ", planning_step_token_num=" << total_planning_step_token_num
                     << ", planning_step_logits_num=" << total_planning_step_logits_num
                     << ", pipeline_depth=" << pipeline_depth_;
}

void EventDrivenScheduleProcessor::RunScheduleForBatches() {
  PROFILE_EVENT_SCOPE(run_schedule_for_batches, "Engine.RunScheduleForBatches");
  KLLM_LOG_SCHEDULER << "enter";
  ReportEventDrivenState("before_schedule");
  bool has_resource_released = false;
  do {
    has_resource_released = false;
    const int64_t schedule_start_time_in_ms = ProfileTimer::GetCurrentTimeInMs();
    schedule_context_.SetScheduleStartTimeInMs(schedule_start_time_in_ms);

    // PreSchedule timeout/abort scan runs once, covering ALL batches.
    // Done outside the per-batch loop to avoid N×N redundant scans.
    std::vector<ScheduleBackendEvent> pre_schedule_events;
    {
      PROFILE_EVENT_SCOPE(pre_schedule, "Engine.RunScheduleForBatches.PreSchedule");
      backend_->PreSchedule(schedule_context_, planning_batches_, pre_schedule_events);
    }
    has_resource_released |= ApplyBackendEventsToState(pre_schedule_events);

    // Schedule each PP batch independently.
    for (size_t multi_batch_id = 0; multi_batch_id < max_pp_batch_num_; ++multi_batch_id) {
      schedule_context_.SetCurrentMultiBatchId(multi_batch_id);

      std::vector<ScheduleBackendEvent> backend_events;
      {
        PROFILE_EVENT_SCOPE(schedule_backend, "Engine.RunScheduleForBatches.Schedule");
        backend_->Schedule(schedule_context_, planning_batches_, backend_events);
      }

      has_resource_released |= ApplyBackendEventsToState(backend_events);
    }
  } while (has_resource_released);
  ReportEventDrivenState("after_schedule");
}

bool EventDrivenScheduleProcessor::TryBuildAndLaunchScheduleResult(ScheduleContext& ctx) {
  PROFILE_EVENT_SCOPE(try_build_and_launch, "Engine.TryBuildAndLaunchScheduleResult");
  static constexpr size_t kMaxPipelineDepth = 2;

  bool any_launched = false;
  bool any_pending = false;  // IsLaunchable but blocked by pipeline depth.

  for (size_t multi_batch_id = 0; multi_batch_id < max_pp_batch_num_; ++multi_batch_id) {
    PlanningBatch* planning_batch = planning_batches_[multi_batch_id].get();

    if (!planning_batch->IsLaunchable()) {
      // Diagnostic (throttled to ~1/s): record WHY the launch is gated so a wedge
      // can be attributed to an empty batch (transient resource shortage) vs a
      // request stuck with an unfinished inflight task (a lost completion event).
      static int64_t last_gate_log_ms = 0;
      const int64_t now_ms = ProfileTimer::GetCurrentTimeInMs();
      if (now_ms - last_gate_log_ms >= 1000) {
        last_gate_log_ms = now_ms;
        size_t inflight_cnt = 0;
        std::string inflight_ids;
        for (const auto& dp_batch : planning_batch->dp_batches) {
          for (const auto& r : dp_batch.running_reqs) {
            if (r.second != nullptr && !r.second->IsStopped() && r.second->HasInflightTask()) {
              ++inflight_cnt;
              if (inflight_ids.size() < 256) {
                inflight_ids += std::to_string(r.first) + ",";
              }
            }
          }
        }
        KLLM_LOG_DEBUG << "TryBuildAndLaunchScheduleResult gated (not launchable): multi_batch_id=" << multi_batch_id
                       << " running_req_num=" << planning_batch->GetRunningReqNum() << " inflight_reqs=" << inflight_cnt
                       << " inflight_ids=[" << inflight_ids << "]";
      }
      continue;
    }

    if (pipeline_depth_ >= kMaxPipelineDepth) {
      any_pending = true;
      continue;
    }

    std::shared_ptr<ScheduleResult> result = std::make_shared<ScheduleResult>();
    result->schedule_output = std::make_shared<ScheduleOutput>();

    // Batch insert all dp_batches[*].running_reqs into schedule_output->running_reqs.
    result->schedule_output->finish_req_ids.resize(dp_num_);
    result->schedule_output->recompute_req_ids.resize(dp_num_);
    for (size_t dp_idx = 0; dp_idx < dp_num_; ++dp_idx) {
      for (auto it : planning_batch->dp_batches[dp_idx].running_reqs) {
        result->schedule_output->running_reqs.push_back(it.second);
      }
      auto& dp_state = schedule_state_.schedule_states[multi_batch_id][dp_idx];
      // Info for nodes — per-batch schedule ops.
      result->schedule_output->finish_req_ids[dp_idx] = dp_state.schedule_ops_for_nodes.stopped_req_ids;
      result->schedule_output->recompute_req_ids[dp_idx] = dp_state.schedule_ops_for_nodes.recomputed_req_ids;
      KLLM_LOG_SCHEDULER << "multi_batch_id=" << multi_batch_id << " dp_idx=" << dp_idx
                         << ", dp_state=" << dp_state.schedule_ops_for_nodes.ToString();
      dp_state.schedule_ops_for_nodes.Reset();
    }

    result->schedule_output->schedule_id = schedule_id_++;
    result->schedule_output->enable_profile_metrics = planning_batch->enable_profile_metrics;

    result->generation_output_group.multi_batch_id = multi_batch_id;
    result->generation_output_group.reqs.resize(dp_num_);
    for (size_t dp_idx = 0; dp_idx < dp_num_; ++dp_idx) {
      for (const auto& r : planning_batch->dp_batches[dp_idx].running_reqs) {
        result->generation_output_group.reqs[dp_idx].push_back(r.second);
        r.second->LaunchPlanningTask();
      }
    }

    // 为本轮 batch 中的 prefill 请求注册 pending put 计数
    if (global_cache_put_tracker_ && !result->schedule_output->running_reqs.empty()) {
      global_cache_put_tracker_->OnBatchDispatched(result->schedule_output->running_reqs);
    }

    ProcessScheduleDataInternal(multi_batch_id, *result);
    KLLM_LOG_SCHEDULER << "Built ScheduleResult for multi_batch_id=" << multi_batch_id
                       << ", schedule_id=" << result->schedule_output->schedule_id
                       << ", num_reqs=" << result->schedule_output->running_reqs.size()
                       << ", step_token_num=" << planning_batch->running_step_token_num
                       << ", step_logits_num=" << planning_batch->running_step_logits_num
                       << ", pipeline_depth=" << pipeline_depth_;

    // pd_v2: record this step's launch timestamp + token count; folded into the
    // per-step EWMA when its generation result returns (OnEventUpdateAfterGeneration → RecordStepLatency).
    last_step_launch_ms_ = ProfileTimer::GetCurrentTimeInMs();
    last_step_token_num_ = static_cast<uint32_t>(planning_batch->running_step_token_num);

    result_queue_.Put(result);
    if (workload_balancer_ != nullptr) {
      workload_balancer_->RecordInflightBatch(*planning_batch, ctx);
    }

    ++pipeline_depth_;
    planning_batch->Reset();
    any_launched = true;
  }

  // If any batch is launchable but blocked by pipeline depth, keep the flag so
  // AsyncSchedulerLoop will retry after a generation result comes back.
  // If at least one batch was launched, clear the flag.
  // If nothing was launchable at all, leave the flag unchanged — the caller
  // (OnEventGetScheduleResult) will set it to true for the next retry.
  if (any_pending) {
    pending_get_schedule_result_ = true;
  } else if (any_launched) {
    pending_get_schedule_result_ = false;
  }

  return any_launched;
}

Status EventDrivenScheduleProcessor::ProcessScheduleDataInternal(size_t multi_batch_id, ScheduleResult& result) {
  // TODO(robertyuan): deep copy running_reqs to avoid InferRequest changesin async scheduling
  result.running_reqs.clear();
  for (auto& req : result.schedule_output->running_reqs) {
    req->accepted_tokens.clear();
    req->generated_tokens.clear();
    result.running_reqs[req->req_id] = req;
  }
  return Status();
}

void EventDrivenScheduleProcessor::RecordStepLatency() {
  if (last_step_launch_ms_ == 0) {
    return;  // no step launched yet
  }
  const int64_t now_ms = ProfileTimer::GetCurrentTimeInMs();
  const int64_t elapsed_ms = now_ms - last_step_launch_ms_;
  const uint32_t latency_us = static_cast<uint32_t>(std::max<int64_t>(0, elapsed_ms) * 1000);

  // EWMA with alpha = 0.2; first sample (old == 0) seeds directly.
  constexpr double kAlpha = 0.2;
  const uint32_t old_tok = step_token_ewma_.load(std::memory_order_relaxed);
  const uint32_t old_lat = step_latency_us_ewma_.load(std::memory_order_relaxed);
  const uint32_t new_tok = old_tok == 0
                               ? last_step_token_num_
                               : static_cast<uint32_t>(kAlpha * last_step_token_num_ + (1.0 - kAlpha) * old_tok);
  const uint32_t new_lat =
      old_lat == 0 ? latency_us : static_cast<uint32_t>(kAlpha * latency_us + (1.0 - kAlpha) * old_lat);
  step_token_ewma_.store(new_tok, std::memory_order_relaxed);
  step_latency_us_ewma_.store(new_lat, std::memory_order_relaxed);
  step_latency_us_inst_.store(latency_us, std::memory_order_relaxed);
}

ScheduleLoadStat EventDrivenScheduleProcessor::GetLocalLoadUnlocked() {
  ScheduleLoadStat stat;
  // schedule_states is a 2D matrix [multi_batch_id][dp_idx] after the
  // pipeline-parallel multi-batch rework; aggregate load across both dims.
  for (const auto& batch_states : schedule_state_.schedule_states) {
    for (const auto& dp_state : batch_states) {
      stat.waiting_reqs += static_cast<uint32_t>(dp_state.prefilling_reqs.size());
      stat.running_reqs += static_cast<uint32_t>(dp_state.decoding_reqs.size());
      for (const auto& req : dp_state.prefilling_reqs) {
        if (req != nullptr) {
          stat.pending_prefill_tokens += static_cast<uint64_t>(req->GetPlanningSequenceLen());
        }
      }
    }
  }
  stat.step_token_num = step_token_ewma_.load(std::memory_order_relaxed);
  stat.step_latency_us = step_latency_us_ewma_.load(std::memory_order_relaxed);
  stat.step_latency_us_inst = step_latency_us_inst_.load(std::memory_order_relaxed);
  return stat;
}

ScheduleLoadStat EventDrivenScheduleProcessor::GetLocalLoad() {
  // Reentrancy guard: snapshot_fn (engine PeerHealth provider) calls this from
  // the prefill-completion path, which fires synchronously via req->Notify()
  // inside HandleStopRequest — i.e. on the AsyncSchedulerLoop thread while it
  // already holds infer_request_mutex_. Re-locking the non-recursive mutex
  // there self-deadlocks the engine (observed: prefill wedged under load, GPU
  // 0%). Off-thread callers (heartbeat BroadcastSnapshot, IPC OnBarrierDone)
  // do not hold the lock and take it normally.
  if (std::this_thread::get_id() == async_loop_tid_.load(std::memory_order_relaxed)) {
    return GetLocalLoadUnlocked();
  }
  std::lock_guard<std::mutex> lk(infer_request_mutex_);
  return GetLocalLoadUnlocked();
}

void EventDrivenScheduleProcessor::SetScheduleBackend(std::shared_ptr<ScheduleBackendInterface> backend) {
  KLLM_LOG_SCHEDULER << "Setting ScheduleBackend " << backend.get();
  backend_ = std::move(backend);
  if (backend_ != nullptr && workload_balancer_ != nullptr) {
    backend_->SetWorkloadBalancer(workload_balancer_);
  }
}

void EventDrivenScheduleProcessor::SetScheduleHookManager(std::shared_ptr<ScheduleHookManagerInterface> hook_manager) {
  KLLM_LOG_SCHEDULER << "ScheduleHookManager: " << hook_manager.get();
  schedule_hook_manager_ = std::move(hook_manager);
  KLLM_CHECK(schedule_hook_manager_ != nullptr);
  schedule_hook_manager_->SetAsyncTaskFinishedCallback(
      [this](const EventAsyncTaskFinishedInfo& info) { EnqueueAsyncTaskFinished(info); });
}

void EventDrivenScheduleProcessor::SetWorkloadBalancer(std::shared_ptr<WorkloadBalancerInterface> workload_balancer) {
  KLLM_LOG_SCHEDULER << "WorkloadBalancer: " << workload_balancer.get();
  workload_balancer_ = std::move(workload_balancer);
  if (backend_ != nullptr) {
    backend_->SetWorkloadBalancer(workload_balancer_);
  }
}

void EventDrivenScheduleProcessor::SetGlobalCachePutTracker(std::shared_ptr<GlobalCachePutTracker> tracker) {
  global_cache_put_tracker_ = std::move(tracker);
  KLLM_LOG_SCHEDULER << "GlobalCachePutTracker set for EventDrivenScheduleProcessor";
}

void EventDrivenScheduleProcessor::SetPpBatchDistributor(std::shared_ptr<PpBatchDistributor> distributor) {
  pp_batch_distributor_ = std::move(distributor);
  KLLM_LOG_SCHEDULER << "PpBatchDistributor set for EventDrivenScheduleProcessor";
}

void EventDrivenScheduleProcessor::SetCacheManagers(
    std::vector<std::shared_ptr<CacheManagerInterface>> dp_cache_managers) {
  KLLM_LOG_SCHEDULER << "CacheManager number: " << dp_cache_managers.size();
  KLLM_CHECK(dp_cache_managers.size() == dp_num_);
  dp_cache_managers_ = std::move(dp_cache_managers);
}

}  // namespace ksana_llm
