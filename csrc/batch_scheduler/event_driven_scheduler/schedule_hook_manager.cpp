/* Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * 事件驱动调度器：Hook 责任链管理器实现。
 * ==============================================================================*/

#include "batch_scheduler/event_driven_scheduler/schedule_hook_manager.h"

#include <algorithm>

#include "utils/logger.h"

namespace ksana_llm {

void ScheduleHookManager::AddHook(ScheduleHookPtr hook) {
  if (hook != nullptr) {
    hook->SetAsyncTaskFinishedCallback(async_task_finished_callback_);
    hooks_.push_back(std::move(hook));
  }
}

void ScheduleHookManager::SetAsyncTaskFinishedCallback(AsyncTaskFinishedCallback callback) {
  async_task_finished_callback_ = std::move(callback);
  for (auto& hook : hooks_) {
    if (hook != nullptr) {
      hook->SetAsyncTaskFinishedCallback(async_task_finished_callback_);
    }
  }
}

ReqIdToNextStep ScheduleHookManager::Process(const std::vector<InferRequestPtr>& requests, ScheduleContext& context) {
  return ProcessInternal(requests, context, std::unordered_map<int64_t, size_t>{});
}

ReqIdToNextStep ScheduleHookManager::ProcessInternal(
    const std::vector<InferRequestPtr>& requests, ScheduleContext& context,
    const std::unordered_map<int64_t, size_t>& req_id_to_start_hook_idx) {
  ReqIdToNextStep result;
  if (requests.empty()) {
    return result;
  }

  const size_t hook_count = hooks_.size();
  for (const auto& kv : req_id_to_start_hook_idx) {
    if (kv.second > hook_count) {
      KLLM_LOG_ERROR << "Invalid start_hook_idx for req_id=" << kv.first << ": " << kv.second
                     << ", hook_count=" << hook_count;
      result[kv.first] = NextStep{NextStepAction::kFailAndEnd, 0};
    }
  }

  auto get_start = [&req_id_to_start_hook_idx](int64_t req_id) -> size_t {
    auto it = req_id_to_start_hook_idx.find(req_id);
    return it != req_id_to_start_hook_idx.end() ? it->second : 0;
  };

  // 已经没有可执行 Hook 的请求，直接视为链路继续。
  for (const auto& req : requests) {
    if (result.find(req->req_id) != result.end()) {
      continue;
    }
    if (get_start(req->req_id) == hook_count) {
      result[req->req_id] = NextStep{NextStepAction::kContinue, 0};
    }
  }

  std::vector<InferRequestPtr> active_requests;
  for (size_t cur_hook_idx = 0; cur_hook_idx < hook_count; ++cur_hook_idx) {
    // 在调用当前 Hook 前，将 start_hook_idx == cur_hook_idx 的请求加入活跃列表。
    for (const auto& req : requests) {
      if (result.find(req->req_id) != result.end()) {
        continue;
      }
      if (get_start(req->req_id) == cur_hook_idx) {
        active_requests.push_back(req);
      }
    }
    if (active_requests.empty()) {
      continue;
    }

    ReqIdToNextStep next_steps = hooks_[cur_hook_idx]->Process(active_requests, context);
    std::vector<InferRequestPtr> next_active_requests;
    next_active_requests.reserve(active_requests.size());
    for (auto& req : active_requests) {
      auto it = next_steps.find(req->req_id);
      if (it != next_steps.end()) {
        switch (it->second.action) {
          case NextStepAction::kContinue:
            next_active_requests.push_back(req);
            break;
          case NextStepAction::kWaitAsyncResult:
          case NextStepAction::kFailAndEnd:
            result[req->req_id] = it->second;
            break;
          default:
            KLLM_LOG_WARNING << "Unknown NextStepAction for req_id=" << req->req_id << ", fallback to fail-and-end";
            result[req->req_id] = NextStep{NextStepAction::kFailAndEnd, 0};
            break;
        }
      } else {
        // 该 Hook 没有返回该请求的下一步指引，默认继续。
        next_active_requests.push_back(req);
      }
    }
    active_requests = std::move(next_active_requests);
  }

  // 穿过全部 Hook 仍在活跃列表中的请求，最终为 kContinue。
  for (auto& req : active_requests) {
    if (result.find(req->req_id) == result.end()) {
      result[req->req_id] = NextStep{NextStepAction::kContinue, 0};
    }
  }
  return result;
}

ReqIdToNextStep ScheduleHookManager::UpdateAsyncResult(const std::vector<AsyncTaskResult>& finished_tasks,
                                                       ScheduleContext& context) {
  ReqIdToNextStep result;
  if (finished_tasks.empty()) {
    return result;
  }

  auto find_preprocessing_req = [&context](int64_t req_id) -> InferRequestPtr {
    InferRequestPtr req = context.GetReqById(req_id);
    if (req == nullptr) {
      return nullptr;
    }
    const size_t dp_idx = static_cast<size_t>(req->attn_dp_group_id);
    size_t multi_batch_id = context.GetReqBatchIdx(req_id);
    const DpScheduleState& dp_state = context.GetDpScheduleState(multi_batch_id, dp_idx);
    auto req_it = dp_state.preprocessing_reqs.find(req_id);
    if (req_it != dp_state.preprocessing_reqs.end()) {
      return req_it->second;
    }
    return nullptr;
  };

  std::unordered_map<int64_t, size_t> req_id_to_start_hook_idx;
  for (size_t hook_idx = 0; hook_idx < hooks_.size(); ++hook_idx) {
    std::vector<AsyncTaskResult> tasks_for_hook;
    for (const auto& task : finished_tasks) {
      if (task.hook_idx >= 0 && static_cast<size_t>(task.hook_idx) == hook_idx) {
        tasks_for_hook.push_back(task);
      }
    }
    if (tasks_for_hook.empty()) {
      continue;
    }

    ReqIdToNextStep next_steps = hooks_[hook_idx]->UpdateAsyncResult(tasks_for_hook, context);
    for (auto& kv : next_steps) {
      InferRequestPtr req = find_preprocessing_req(kv.first);
      if (req == nullptr) {
        KLLM_LOG_ERROR << "Invalid async finished state: req_id " << kv.first << " from hook_idx " << hook_idx
                       << " is not in preprocessing_reqs";
        continue;
      }
      if (kv.second.action == NextStepAction::kContinue) {
        req_id_to_start_hook_idx[kv.first] = hook_idx + 1;
      } else {
        result[kv.first] = kv.second;
      }
    }
  }

  if (req_id_to_start_hook_idx.empty()) {
    return result;
  }

  std::vector<InferRequestPtr> continued;
  continued.reserve(req_id_to_start_hook_idx.size());
  for (const auto& kv : req_id_to_start_hook_idx) {
    InferRequestPtr req = find_preprocessing_req(kv.first);
    if (req != nullptr) {
      continued.push_back(req);
    }
  }
  if (continued.empty()) {
    return result;
  }

  ReqIdToNextStep chain_result = ProcessInternal(continued, context, req_id_to_start_hook_idx);
  for (auto& kv : chain_result) {
    result[kv.first] = kv.second;
  }
  return result;
}

}  // namespace ksana_llm
