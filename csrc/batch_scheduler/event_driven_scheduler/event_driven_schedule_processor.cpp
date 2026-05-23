/* Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * 事件驱动调度器：单主线程 + 事件队列实现。
 * ==============================================================================*/

#include "batch_scheduler/event_driven_scheduler/event_driven_schedule_processor.h"

#include <algorithm>

#include "batch_scheduler/event_driven_scheduler/continuous_batching_backend.h"
#include "configure/environment.h"
#include "profiler/profile_event.h"
#include "profiler/timer.h"
#include "runtime/schedule_output.h"
#include "utils/logger.h"

namespace ksana_llm {

namespace {

EventDrivenScheduleProcessor* g_swap_callback_target = nullptr;

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
  KLLM_LOG_SCHEDULER << "HandleStopRequest: req_id=" << req->req_id << " dp_idx=" << dp_idx;
  DpScheduleState& dp_state = schedule_state_.schedule_states[dp_idx];

  // Stop request is now actually applied, remove it from in-plan ops.
  dp_state.in_plan_schedule_ops.stop_req_ids.erase(req->req_id);

  req->Stop();

  // Release blocks and notify clients.
  req->finished = true;
  req->finish_status = ret_status;
  if (!global_cache_put_tracker_ || !global_cache_put_tracker_->DeferDestructionIfNeeded(req->req_id, dp_idx)) {
    dp_cache_managers_[dp_idx]->DestroyFinishedRequest(req->req_id);
  }
  req->kv_cache_blocks.clear();
  req->Notify();

  // Record stopped reqs for workers
  dp_state.schedule_ops_for_nodes.stopped_req_ids.push_back(req->req_id);

  // Remove from global req index when stop is actually applied.
  schedule_state_.req_id_to_req_ptr.erase(req->req_id);
}

void EventDrivenScheduleProcessor::HandleHookFailAndEnd(const std::shared_ptr<InferRequest>& req, size_t dp_idx,
                                                        const std::string& reason) {
  if (req == nullptr) {
    KLLM_LOG_ERROR << "HandleHookFailAndEnd: req is nullptr";
    return;
  }
  DpScheduleState& dp_state = schedule_state_.schedule_states[dp_idx];
  dp_state.preprocessing_reqs.erase(req->req_id);
  RemoveReqByIdFromQueue(dp_state.prefilling_reqs, req->req_id);
  RemoveReqByIdFromQueue(dp_state.decoding_reqs, req->req_id);
  RemoveReqByIdFromQueue(dp_state.swapped_reqs, req->req_id);

  req->Stop();
  // 请求可能已被 GlobalCacheFetcher 拒绝并 Notify，需避免重复通知
  bool already_finished = req->finished;
  req->finished = true;
  req->finish_status = Status(RET_RUNTIME_FAILED, reason);
  if (!global_cache_put_tracker_ || !global_cache_put_tracker_->DeferDestructionIfNeeded(req->req_id, dp_idx)) {
    dp_cache_managers_[dp_idx]->DestroyFinishedRequest(req->req_id);
  }
  req->kv_cache_blocks.clear();
  if (!already_finished) {
    req->Notify();
  }
  schedule_state_.req_id_to_req_ptr.erase(req->req_id);
  KLLM_LOG_WARNING << "Hook fail-and-end req_id=" << req->req_id << ", reason=" << reason;
}

void EventDrivenScheduleProcessor::HandleRecomputeRequest(const std::shared_ptr<InferRequest>& req, Status ret_status,
                                                          size_t dp_idx) {
  if (req == nullptr) {
    KLLM_LOG_ERROR << "req is nullptr";
    return;
  }

  DpScheduleState& dp_state = schedule_state_.schedule_states[dp_idx];

  // Recompute request is now actually applied, remove it from in-plan ops.
  dp_state.in_plan_schedule_ops.recompute_req_ids.erase(req->req_id);
  if (req->IsStopped()) {
    KLLM_LOG_SCHEDULER << "req_id=" << req->req_id << " is stopped, drop recompute event";
    return;
  }
  req->ResetInflightTask();
  req->Restart();
  req->ResetPrefillingTokens();
  if (!global_cache_put_tracker_ || !global_cache_put_tracker_->DeferDestructionIfNeeded(req->req_id, dp_idx)) {
    dp_cache_managers_[dp_idx]->DestroyFinishedRequest(req->req_id);
  }
  req->kv_cache_blocks.clear();

  if (!HasReqInQueue(dp_state.prefilling_reqs, req->req_id)) {
    dp_state.prefilling_reqs.push_back(req);
  }

  KLLM_LOG_SCHEDULER << "After recompute req_id=" << req->req_id << ". dp_idx=" << dp_idx
                     << ", decoding_reqs=" << dp_state.decoding_reqs.size()
                     << ", prefilling_reqs=" << dp_state.prefilling_reqs.size();

  // Record recomputed reqs for workers
  dp_state.schedule_ops_for_nodes.recomputed_req_ids.push_back(req->req_id);
}

bool EventDrivenScheduleProcessor::HandleSwapoutRequest(const std::shared_ptr<InferRequest>& req, Status ret_status,
                                                        size_t dp_idx) {
  if (req == nullptr) {
    KLLM_LOG_ERROR << "req is nullptr";
    return false;
  }
  KLLM_CHECK_WITH_INFO(dp_idx < dp_cache_managers_.size(), "Invalid dp_idx for swapout request");

  DpScheduleState& dp_state = schedule_state_.schedule_states[dp_idx];

  if (req->IsStopped()) {
    KLLM_LOG_SCHEDULER << "req_id=" << req->req_id << " is stopped, drop swapout event";
    dp_state.in_plan_schedule_ops.swapout_req_ids.erase(req->req_id);
    return false;
  }

  auto& cache_manager = dp_cache_managers_[dp_idx];
  KLLM_CHECK(cache_manager != nullptr);

  size_t free_block_num = 0;
  size_t swapped_block_num = 0;
  std::vector<int> swapout_memory_blocks;
  std::vector<int> swapout_host_block_ids;

  Status swap_status = cache_manager->SwapoutRequestAsync(req->req_id, swapped_block_num, free_block_num,
                                                          swapout_memory_blocks, swapout_host_block_ids);
  if (!swap_status.OK()) {
    // when swapout cannot be started, fallback to recompute immediately.
    KLLM_LOG_ERROR << "HandleSwapoutRequest error, fallback to recompute req_id=" << req->req_id
                   << ", msg=" << swap_status.GetMessage();
    dp_state.in_plan_schedule_ops.swapout_req_ids.erase(req->req_id);
    dp_state.in_plan_schedule_ops.recompute_req_ids.insert(req->req_id);
    HandleRecomputeRequest(req, ret_status, dp_idx);
    return true;
  }

  SwapReqBlockInfo& swapout_info = dp_state.schedule_ops_for_nodes.swapout_req_block_infos[req->req_id];
  swapout_info.dev_block_ids = std::move(swapout_memory_blocks);
  swapout_info.host_block_ids = std::move(swapout_host_block_ids);
  KLLM_LOG_SCHEDULER << "Start swapout for req_id=" << req->req_id;

  // Structured generator state is not preserved by swap and should be recreated after swapin.
  if (req->structured_generator_config.HasConstraint()) {
    req->need_recreate_generator = true;
  }
  return false;
}

bool EventDrivenScheduleProcessor::HandleSwapinRequest(const std::shared_ptr<InferRequest>& req, Status ret_status,
                                                       size_t dp_idx) {
  if (req == nullptr) {
    KLLM_LOG_ERROR << "req is nullptr";
    return false;
  }
  KLLM_CHECK_WITH_INFO(dp_idx < dp_cache_managers_.size(), "Invalid dp_idx for swapin request");

  DpScheduleState& dp_state = schedule_state_.schedule_states[dp_idx];

  if (req->IsStopped()) {
    KLLM_LOG_SCHEDULER << "req_id=" << req->req_id << " is stopped, drop swapin event";
    dp_state.in_plan_schedule_ops.swapin_req_ids.erase(req->req_id);
    return false;
  }

  auto& cache_manager = dp_cache_managers_[dp_idx];
  KLLM_CHECK(cache_manager != nullptr);

  size_t swapin_block_num = 0;
  Status needed_status = cache_manager->GetRequestNeededBlockNumForOneNextToken(req->req_id, swapin_block_num);
  if (!needed_status.OK()) {
    KLLM_LOG_ERROR << "GetRequestNeededBlockNumForOneNextToken failed for req_id=" << req->req_id
                   << ", msg=" << needed_status.GetMessage();
    dp_state.in_plan_schedule_ops.swapin_req_ids.erase(req->req_id);
    dp_state.swapped_reqs.insert(dp_state.swapped_reqs.begin(),
                                 req);  // keep request in swapped queue if failed to start swapin task.
    return false;
  }

  std::vector<int> swapin_memory_blocks;
  std::vector<int> swapin_host_block_ids;
  Status swap_status = cache_manager->SwapinRequestAsync(req->req_id, swapin_block_num, req->kv_cache_blocks,
                                                         swapin_memory_blocks, swapin_host_block_ids);
  if (!swap_status.OK()) {
    KLLM_LOG_ERROR << "HandleSwapinRequest failed for req_id=" << req->req_id << ", msg=" << swap_status.GetMessage();
    dp_state.in_plan_schedule_ops.swapin_req_ids.erase(req->req_id);
    dp_state.swapped_reqs.insert(dp_state.swapped_reqs.begin(),
                                 req);  // keep request in swapped queue if failed to start swapin task.
    return false;
  }

  SwapReqBlockInfo& swapin_info = dp_state.schedule_ops_for_nodes.swapin_req_block_infos[req->req_id];
  swapin_info.dev_block_ids = std::move(swapin_memory_blocks);
  swapin_info.host_block_ids = std::move(swapin_host_block_ids);

  KLLM_LOG_SCHEDULER << "Start swapin for req_id=" << req->req_id;

  return false;
}

bool EventDrivenScheduleProcessor::ApplyBackendEventsToState(const std::vector<ScheduleBackendEvent>& backend_events) {
  if (backend_events.empty()) {
    return false;
  }
  bool has_resource_released = false;
  KLLM_LOG_SCHEDULER << "ApplyBackendEventsToState: " << backend_events.size() << " events";
  auto remove_from_state_and_batches = [this](size_t dp_idx, int64_t req_id) {
    DpScheduleState& dp_state = schedule_state_.schedule_states[dp_idx];
    dp_state.preprocessing_reqs.erase(req_id);
    RemoveReqByIdFromQueue(dp_state.prefilling_reqs, req_id);
    RemoveReqByIdFromQueue(dp_state.decoding_reqs, req_id);
    RemoveReqByIdFromQueue(dp_state.swapped_reqs, req_id);

    // Remove from all planning batches.
    for (const auto& pb : planning_batches_) {
      if (pb == nullptr) {
        continue;
      }
      if (dp_idx < pb->dp_batches.size()) {
        pb->RemoveRunningReq(dp_idx, req_id);
      }
    }
  };

  for (const auto& ev : backend_events) {
    const size_t dp_idx = ev.dp_idx;
    DpScheduleState& dp_state = schedule_state_.schedule_states[dp_idx];
    std::shared_ptr<InferRequest> req = ev.req_ptr;
    if (req == nullptr) {
      KLLM_LOG_ERROR << "Invalid backend event: null req_ptr, dp_idx=" << dp_idx;
      continue;
    }
    const int64_t req_id = req->req_id;

    if (ev.type == ScheduleBackendEventType::kMoveToDecode) {
      // Move request from prefilling queue to decoding queue.
      RemoveReqByIdFromQueue(dp_state.prefilling_reqs, req_id);
      if (!HasReqInQueue(dp_state.decoding_reqs, req_id)) {
        dp_state.decoding_reqs.push_back(req);
      }
      continue;
    }

    if (ev.type == ScheduleBackendEventType::kSwapinRequest) {
      // Start swapin immediately and keep request out of swapped queue while pending.
      RemoveReqByIdFromQueue(dp_state.swapped_reqs, req_id);
      has_resource_released |= HandleSwapinRequest(req, ev.status, dp_idx);
      continue;
    }

    // Stop/Recompute/Swapout 的请求级操作
    // 需要确认是否要延迟到无 in-flight task 时执行。

    // 立即从状态与 batch 中摘除，避免后续调度继续使用。
    remove_from_state_and_batches(dp_idx, req_id);

    // Stop/Recompute/Swapout 的请求级操作延迟到无 in-flight task 时执行。
    if (req->HasInflightTask()) {
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
    } else if (ev.type == ScheduleBackendEventType::kSwapoutRequest) {
      has_resource_released |= HandleSwapoutRequest(req, ev.status, dp_idx);
    }
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
    } else if (ev.type == ScheduleBackendEventType::kSwapoutRequest) {
      has_resource_released |= HandleSwapoutRequest(req, ev.status, dp_idx);
    } else if (ev.type == ScheduleBackendEventType::kSwapinRequest) {
      has_resource_released |= HandleSwapinRequest(req, ev.status, dp_idx);
    }

    delayed_backend_events_.erase(delayed_backend_events_.begin() + static_cast<ptrdiff_t>(i));
  }
  return has_resource_released;
}

EventDrivenScheduleProcessor::EventDrivenScheduleProcessor(size_t dp_num, size_t max_pp_batch_num)
    : dp_num_(dp_num), max_pp_batch_num_(max_pp_batch_num) {}

EventDrivenScheduleProcessor::~EventDrivenScheduleProcessor() { Stop(); }

void EventDrivenScheduleProcessor::Initialize(std::shared_ptr<BatchSchedulerInterface> batch_scheduler) {
  batch_scheduler_ = batch_scheduler;  // 满足基类 ScheduleProcessorInterface 的接口约定

  schedule_state_.schedule_states.clear();
  schedule_state_.schedule_states.resize(dp_num_);

  // Initialize reusable data structures once and reuse them afterwards.
  planning_batches_.assign(max_pp_batch_num_, nullptr);
  for (size_t batch_idx = 0; batch_idx < max_pp_batch_num_; ++batch_idx) {
    planning_batches_[batch_idx] = std::make_shared<PlanningBatch>();
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

void EventDrivenScheduleProcessor::EnqueueSwapTaskFinished(const EventSwapTaskFinishedInfo& event_info) {
  auto info = std::make_shared<EventSwapTaskFinishedInfo>();
  *info = event_info;
  SchedEvent ev(std::move(info));
  event_queue_.Put(std::move(ev));
}

void EventDrivenScheduleProcessor::UpdateAfterAsyncTaskFinished(const EventAsyncTaskFinishedInfo& event_info) {
  if (event_info.task_results.empty()) {
    return;
  }
  KLLM_LOG_SCHEDULER << event_info.task_results.size() << " task results";
  ReqIdToNextStep next_steps = schedule_hook_manager_->UpdateAsyncResult(event_info.task_results, schedule_context_);

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
    DpScheduleState& dp_state = schedule_state_.schedule_states[dp_idx];

    switch (step.action) {
      case NextStepAction::kContinue: {
        // Move from preprocessing -> prefilling if present.
        auto pre_it = dp_state.preprocessing_reqs.find(req_id);
        if (pre_it != dp_state.preprocessing_reqs.end()) {
          dp_state.prefilling_reqs.push_back(pre_it->second);
          dp_state.preprocessing_reqs.erase(pre_it);
          KLLM_LOG_SCHEDULER << "Async task finished for req_id=" << req_id
                             << ", moved from preprocessing to prefilling";
        } else {
          KLLM_LOG_SCHEDULER << "req_id=" << req_id << " not found in preprocessing_reqs";
        }
        break;
      }
      case NextStepAction::kFailAndEnd: {
        HandleHookFailAndEnd(req_ptr, dp_idx, "hook async processing failed");
        break;
      }
      case NextStepAction::kWaitAsyncResult:
      default: {
        KLLM_LOG_SCHEDULER << "Async task finished for req_id=" << req_id
                           << ", no state change, still waiting async result";
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
  if (g_swap_callback_target == this) {
    g_swap_callback_target = nullptr;
  }
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
  while (!terminated_) {
    SchedEvent event;
    {
      PROFILE_EVENT_SCOPE(sched_loop_wait, "Engine.AsyncSchedulerLoop.Wait");
      event = event_queue_.Get();
    }
    if (terminated_) {
      break;
    }
    if (event.IsEmpty()) {
      continue;  // 空事件仅用于 Stop() 时唤醒 Get()
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
    case SchedEventType::EventSwapTaskFinished: {
      auto info = std::dynamic_pointer_cast<EventSwapTaskFinishedInfo>(event.event_info);
      if (info == nullptr) {
        KLLM_LOG_FATAL << "Event type/payload mismatch for EventSwapTaskFinished";
        return;
      }
      OnEventSwapTaskFinished(info);
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
    req->output_tokens = req->input_tokens;
    req->ResetPrefillingTokens();
  }

  std::vector<std::shared_ptr<InferRequest>> requests = info->reqs;
  if (workload_balancer_ != nullptr) {
    workload_balancer_->AddInferRequests(requests, schedule_context_);
  }
  ReqIdToNextStep next_steps = schedule_hook_manager_->Process(requests, schedule_context_);
  for (auto& req : info->reqs) {
    const size_t dp_idx = static_cast<size_t>(req->attn_dp_group_id);
    DpScheduleState& dp_state = schedule_state_.schedule_states[dp_idx];
    auto ns = next_steps.find(req->req_id);
    if (ns == next_steps.end()) {
      KLLM_LOG_WARNING << "req_id: " << req->req_id << " has no next_step, default continue";
      dp_state.prefilling_reqs.push_back(req);
      continue;
    }
    if (ns->second.action == NextStepAction::kContinue) {
      KLLM_LOG_SCHEDULER << "req_id: " << req->req_id << " continue, added to prefilling_reqs. dp_idx=" << dp_idx
                         << ", " << req->ScheduleStateToStr()
                         << ", stop_tokens=" << Vector2Str(req->sampling_config.stop_token_ids);
      dp_state.prefilling_reqs.push_back(req);
    } else if (ns->second.action == NextStepAction::kWaitAsyncResult) {
      KLLM_LOG_SCHEDULER << "req_id: " << req->req_id << " wait async result, added to preprocessing_reqs";
      dp_state.preprocessing_reqs[req->req_id] = req;
    } else if (ns->second.action == NextStepAction::kFailAndEnd) {
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
}

void EventDrivenScheduleProcessor::OnEventAsyncTaskFinished(const std::shared_ptr<EventAsyncTaskFinishedInfo>& info) {
  if (!info || info->task_results.empty()) {
    return;
  }
  UpdateAfterAsyncTaskFinished(*info);
}

void EventDrivenScheduleProcessor::OnEventSwapTaskFinished(const std::shared_ptr<EventSwapTaskFinishedInfo>& info) {
  if (!info) {
    return;
  }
  KLLM_LOG_SCHEDULER << "Swap task finished event, req_id=" << info->req_id
                     << ", op=" << (info->is_swapout ? "swapout" : "swapin") << ", block_num=" << info->block_num;
  if (!info->status.OK()) {
    return;
  }

  auto req_it = schedule_state_.req_id_to_req_ptr.find(info->req_id);
  if (req_it == schedule_state_.req_id_to_req_ptr.end() || req_it->second == nullptr) {
    KLLM_LOG_WARNING << "Swap task finished for unknown req_id=" << info->req_id;
    return;
  }
  const size_t dp_idx = static_cast<size_t>(req_it->second->attn_dp_group_id);
  if (dp_idx >= schedule_state_.schedule_states.size()) {
    KLLM_LOG_WARNING << "Invalid dp_idx in swap finished event, req_id=" << info->req_id << ", dp_idx=" << dp_idx;
    return;
  }
  DpScheduleState& dp_state = schedule_state_.schedule_states[dp_idx];
  if (info->is_swapout) {
    // Swapout
    std::unordered_set<int64_t> waiting_req_ids;
    waiting_req_ids.insert(info->req_id);
    std::vector<int64_t> finished_req_ids;
    size_t left_req_num = 0;
    Status wait_status =
        dp_cache_managers_[dp_idx]->WaitSwapoutRequests(waiting_req_ids, finished_req_ids, left_req_num, true);
    if (!wait_status.OK()) {
      KLLM_LOG_ERROR << "WaitSwapoutRequests failed: " << wait_status.GetMessage();
      return;
    }
    KLLM_LOG_SCHEDULER << "Swapout finished for req_id=" << info->req_id
                       << ", finished_req_ids=" << Vector2Str(finished_req_ids) << ", left_req_num=" << left_req_num;
    KLLM_CHECK(info->req_id == finished_req_ids[0]);
    Status merge_status = dp_cache_managers_[dp_idx]->MergeSwapoutRequest(info->req_id);
    if (!merge_status.OK()) {
      KLLM_LOG_ERROR << "MergeSwapoutRequest failed for req_id=" << info->req_id
                     << ", msg=" << merge_status.GetMessage();
      return;
    }
    dp_state.in_plan_schedule_ops.swapout_req_ids.erase(info->req_id);
    if (!HasReqInQueue(dp_state.swapped_reqs, info->req_id)) {
      dp_state.swapped_reqs.push_back(req_it->second);
    }
    dp_state.schedule_ops_for_nodes.merged_swapout_req_ids.push_back(info->req_id);
  } else {
    // Swapin
    std::unordered_set<int64_t> waiting_req_ids;
    waiting_req_ids.insert(info->req_id);
    std::vector<int64_t> finished_req_ids;
    size_t left_req_num = 0;
    Status wait_status =
        dp_cache_managers_[dp_idx]->WaitSwapinRequests(waiting_req_ids, finished_req_ids, left_req_num, true);
    if (!wait_status.OK()) {
      KLLM_LOG_ERROR << "WaitSwapinRequests failed: " << wait_status.GetMessage();
      return;
    }
    KLLM_LOG_SCHEDULER << "Swapin finished for req_id=" << info->req_id
                       << ", finished_req_ids=" << Vector2Str(finished_req_ids) << ", left_req_num=" << left_req_num;
    KLLM_CHECK(info->req_id == finished_req_ids[0]);

    Status merge_status = dp_cache_managers_[dp_idx]->MergeSwapinRequest(
        info->req_id, req_it->second->kv_cache_blocks, req_it->second->swa_kv_cache_blocks,
        req_it->second->swa_kv_cache_block_idx_offsets);
    if (!merge_status.OK()) {
      KLLM_LOG_ERROR << "MergeSwapinRequest failed for req_id=" << info->req_id
                     << ", msg=" << merge_status.GetMessage();
      return;
    }

    dp_state.in_plan_schedule_ops.swapin_req_ids.erase(info->req_id);
    RemoveReqByIdFromQueue(dp_state.swapped_reqs, info->req_id);
    if (!HasReqInQueue(dp_state.decoding_reqs, info->req_id)) {
      dp_state.decoding_reqs.push_back(req_it->second);
    }
    dp_state.schedule_ops_for_nodes.merged_swapin_req_ids.push_back(info->req_id);
  }
}

void EventDrivenScheduleProcessor::RunScheduleForBatches() {
  PROFILE_EVENT_SCOPE(run_schedule_for_batches, "Engine.RunScheduleForBatches");
  KLLM_LOG_SCHEDULER << "enter";
  bool has_resource_released = false;
  do {
    has_resource_released = false;
    const int64_t schedule_start_time_in_ms = ProfileTimer::GetCurrentTimeInMs();
    schedule_context_.SetScheduleStartTimeInMs(schedule_start_time_in_ms);
    for (size_t multi_batch_id = 0; multi_batch_id < max_pp_batch_num_; ++multi_batch_id) {
      std::vector<ScheduleBackendEvent> pre_schedule_events;
      {
        PROFILE_EVENT_SCOPE(pre_schedule, "Engine.RunScheduleForBatches.PreSchedule");
        backend_->PreSchedule(schedule_context_, planning_batches_, pre_schedule_events);
      }
      bool resource_released = ApplyBackendEventsToState(pre_schedule_events);
      has_resource_released |= resource_released;

      std::vector<ScheduleBackendEvent> backend_events;
      {
        PROFILE_EVENT_SCOPE(schedule_backend, "Engine.RunScheduleForBatches.Schedule");
        backend_->Schedule(schedule_context_, planning_batches_, backend_events);
      }

      resource_released = ApplyBackendEventsToState(backend_events);
      has_resource_released |= resource_released;
    }
  } while (has_resource_released);
}

bool EventDrivenScheduleProcessor::TryBuildAndLaunchScheduleResult(ScheduleContext& ctx) {
  PROFILE_EVENT_SCOPE(try_build_and_launch, "Engine.TryBuildAndLaunchScheduleResult");
  PlanningBatch* planning_batch = planning_batches_[0].get();

  if (!planning_batch->IsLaunchable()) {
    return false;
  }

  size_t multi_batch_id = planning_batch->batch_idx;

  std::shared_ptr<ScheduleResult> result = std::make_shared<ScheduleResult>();
  result->schedule_output = std::make_shared<ScheduleOutput>();

  // Batch insert all dp_batches[*].running_reqs into schedule_output->running_reqs.
  // Note: dp_batches[dp_idx].running_reqs is a map keyed by req_id.
  result->schedule_output->finish_req_ids.resize(dp_num_);
  result->schedule_output->recompute_req_ids.resize(dp_num_);
  result->schedule_output->merged_swapin_req_ids.resize(dp_num_);
  result->schedule_output->merged_swapout_req_ids.resize(dp_num_);
  result->schedule_output->swapin_req_block_ids.resize(dp_num_);
  result->schedule_output->swapout_req_block_ids.resize(dp_num_);
  for (size_t dp_idx = 0; dp_idx < dp_num_; ++dp_idx) {
    for (auto it : planning_batch->dp_batches[dp_idx].running_reqs) {
      result->schedule_output->running_reqs.push_back(it.second);
    }
    auto& dp_state = schedule_state_.schedule_states[dp_idx];
    // Info for nodes
    result->schedule_output->finish_req_ids[dp_idx] = dp_state.schedule_ops_for_nodes.stopped_req_ids;
    result->schedule_output->recompute_req_ids[dp_idx] = dp_state.schedule_ops_for_nodes.recomputed_req_ids;
    result->schedule_output->merged_swapin_req_ids[dp_idx] = dp_state.schedule_ops_for_nodes.merged_swapin_req_ids;
    result->schedule_output->merged_swapout_req_ids[dp_idx] = dp_state.schedule_ops_for_nodes.merged_swapout_req_ids;
    result->schedule_output->swapin_req_block_ids[dp_idx] = dp_state.schedule_ops_for_nodes.swapin_req_block_infos;
    result->schedule_output->swapout_req_block_ids[dp_idx] = dp_state.schedule_ops_for_nodes.swapout_req_block_infos;
    KLLM_LOG_SCHEDULER << "dp_idx=" << dp_idx << ", dp_state=" << dp_state.schedule_ops_for_nodes.ToString();
    dp_state.schedule_ops_for_nodes.Reset();  // Clear after copying to output
  }

  result->schedule_output->schedule_id = schedule_id_++;

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
                     << ", num_reqs=" << result->schedule_output->running_reqs.size();

  // Clear the flag before putting result into queue to avoid racing condition
  pending_get_schedule_result_ = false;

  result_queue_.Put(result);
  if (workload_balancer_ != nullptr) {
    workload_balancer_->RecordInflightBatch(*planning_batch, ctx);
  }

  // Clear for next batch
  planning_batch->Reset();
  return true;
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

void EventDrivenScheduleProcessor::UpdateAfterSwapTaskFinished(int64_t req_id, bool is_swapout, size_t block_num,
                                                               const Status& status) {
  EventDrivenScheduleProcessor* processor = g_swap_callback_target;
  if (processor == nullptr) {
    KLLM_LOG_WARNING << "Swap task callback dropped: processor is null, req_id=" << req_id;
    return;
  }
  EventSwapTaskFinishedInfo event_info;
  event_info.req_id = req_id;
  event_info.is_swapout = is_swapout;
  event_info.block_num = block_num;
  event_info.status = status;
  processor->EnqueueSwapTaskFinished(event_info);
}

void EventDrivenScheduleProcessor::SetGlobalCachePutTracker(std::shared_ptr<GlobalCachePutTracker> tracker) {
  global_cache_put_tracker_ = std::move(tracker);
  KLLM_LOG_SCHEDULER << "GlobalCachePutTracker set for EventDrivenScheduleProcessor";
}

void EventDrivenScheduleProcessor::SetCacheManagers(
    std::vector<std::shared_ptr<CacheManagerInterface>> dp_cache_managers) {
  KLLM_LOG_SCHEDULER << "CacheManager number: " << dp_cache_managers.size();
  KLLM_CHECK(dp_cache_managers.size() == dp_num_);
  dp_cache_managers_ = std::move(dp_cache_managers);
  g_swap_callback_target = this;
  for (const auto& cache_manager : dp_cache_managers_) {
    KLLM_CHECK(cache_manager != nullptr);
    cache_manager->SetSwapCallbackFunc(&EventDrivenScheduleProcessor::UpdateAfterSwapTaskFinished);
  }
}

}  // namespace ksana_llm
