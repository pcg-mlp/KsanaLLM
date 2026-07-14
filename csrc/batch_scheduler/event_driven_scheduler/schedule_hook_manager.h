/* Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * 事件驱动调度器：Hook 责任链管理器实现。
 * ==============================================================================*/
#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "batch_scheduler/event_driven_scheduler/event_driven_types.h"
#include "batch_scheduler/event_driven_scheduler/schedule_context.h"
#include "batch_scheduler/event_driven_scheduler/schedule_hook_interface.h"
#include "batch_scheduler/event_driven_scheduler/schedule_hook_manager_interface.h"

namespace ksana_llm {

class ScheduleHookManager : public ScheduleHookManagerInterface {
 public:
  ScheduleHookManager() = default;
  ~ScheduleHookManager() override = default;

  void AddHook(ScheduleHookPtr hook) override;

  ReqIdToNextStep Process(const std::vector<InferRequestPtr>& requests, ScheduleContext& context) override;
  ReqIdToNextStep UpdateAsyncResult(const std::vector<AsyncTaskResult>& finished_tasks,
                                    ScheduleContext& context) override;

  void SetAsyncTaskFinishedCallback(AsyncTaskFinishedCallback callback) override;

 private:
  // req_id -> 从哪个 hook_idx 开始调用；按 hook_idx 从小到大依次调用，根据该 map 与 cur_hook_idx 决定是否加入
  // hook->Process 的输入来处理这个请求，从而实现异步结果回调时从特定 Hook 继续处理。
  ReqIdToNextStep ProcessInternal(const std::vector<InferRequestPtr>& requests, ScheduleContext& context,
                                  const std::unordered_map<int64_t, size_t>& req_id_to_start_hook_idx);

  std::vector<ScheduleHookPtr> hooks_;
  AsyncTaskFinishedCallback async_task_finished_callback_;
};

}  // namespace ksana_llm
