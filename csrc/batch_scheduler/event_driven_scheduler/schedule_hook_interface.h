/* Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * 事件驱动调度器：Hook 对象接口，与设计文档 5.1 节对应。
 * ==============================================================================*/
#pragma once

#include <unordered_map>
#include <vector>

#include "batch_scheduler/event_driven_scheduler/event_driven_types.h"
#include "batch_scheduler/event_driven_scheduler/schedule_context.h"
#include "runtime/infer_request.h"

namespace ksana_llm {

// 责任链上对单个请求的下一步指引（设计文档 5.1.1）
enum class NextStepAction {
  kContinue,         // 继续交给后续 Hook 处理
  kWaitAsyncResult,  // 等待异步结果，当前不继续向后传递
  kFailAndEnd,       // 失败并结束处理
};

struct NextStep {
  NextStepAction action = NextStepAction::kContinue;
  // 可携带与 action 相关的附加信息，如 task_id 等
  int task_id = 0;
};

// req_id -> NextStep，描述每个请求经责任链处理后的下一步
using ReqIdToNextStep = std::unordered_map<int64_t, NextStep>;

// 单个 Hook 的抽象（设计文档 5.1.2）
class ScheduleHookInterface {
 public:
  using InferRequestPtr = std::shared_ptr<InferRequest>;

  virtual ~ScheduleHookInterface() = default;

  // 同步处理：基于 requests 做本 Hook 的逻辑，返回每个 req_id 的下一步指引。
  virtual ReqIdToNextStep Process(const std::vector<InferRequestPtr>& requests, ScheduleContext& context) = 0;

  // 异步结果回调：根据已完成的异步任务信息更新状态，返回每个 req_id 的下一步指引。
  virtual ReqIdToNextStep UpdateAsyncResult(const std::vector<AsyncTaskResult>& finished_tasks,
                                            ScheduleContext& context) = 0;

  // 由 HookManager 下发异步完成回调，Hook 在异步任务完成时调用。
  virtual void SetAsyncTaskFinishedCallback(AsyncTaskFinishedCallback callback) = 0;
};

}  // namespace ksana_llm
