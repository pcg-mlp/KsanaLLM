/* Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * 事件驱动调度器：Hook 对象管理器接口，与设计文档 5.2 节对应。
 * ==============================================================================*/
#pragma once

#include <memory>
#include <vector>

#include "batch_scheduler/event_driven_scheduler/event_driven_types.h"
#include "batch_scheduler/event_driven_scheduler/schedule_context.h"
#include "batch_scheduler/event_driven_scheduler/schedule_hook_interface.h"
#include "runtime/infer_request.h"

namespace ksana_llm {

class ScheduleHookManagerInterface {
 public:
  using InferRequestPtr = std::shared_ptr<InferRequest>;
  using ScheduleHookPtr = std::shared_ptr<ScheduleHookInterface>;

  virtual ~ScheduleHookManagerInterface() = default;

  // 添加 Hook 对象, 按照AddHook调用的顺序，分配hook_idx，调度器调用时按照 hook_idx 从小到大顺序调用 Hook 的 Process /
  // UpdateAsyncResult。HookManager 持有 shared_ptr，确保 Hook 生命周期至少与 HookManager 一致。
  virtual void AddHook(ScheduleHookPtr hook) = 0;

  // 沿 Hook 链调用各 Hook 的 Process，返回每个请求的最终 NextStep，由调用方根据结果做请求移动。
  virtual ReqIdToNextStep Process(const std::vector<InferRequestPtr>& requests, ScheduleContext& context) = 0;

  // 异步结果到达时调用：沿链调用各 Hook 的 UpdateAsyncResult，返回 req_id -> NextStep。
  // 在 UpdateAsyncResult 语境下，各 action 的语义约定如下：
  // 1) kContinue: 当前异步结果已就绪，请求可继续沿 Hook 链推进；若后续无阻塞，则由调用方从 preprocessing_reqs 移入
  //    prefilling_reqs。
  // 2) kWaitAsyncResult: 请求仍需等待后续异步结果；调用方应保持请求在 preprocessing_reqs，不做状态迁移。
  // 3) kFailAndEnd: 请求在异步处理阶段失败；调用方应执行失败收敛/结束流程（例如释放资源并返回错误）。
  virtual ReqIdToNextStep UpdateAsyncResult(const std::vector<AsyncTaskResult>& finished_tasks,
                                            ScheduleContext& context) = 0;

  // 设置 Hook 异步任务完成时的回调，通常由调度器设为 UpdateAfterAsyncTaskFinished，
  // HookManager 会将该回调下发给各 Hook，Hook 在异步执行完成时调用。
  virtual void SetAsyncTaskFinishedCallback(AsyncTaskFinishedCallback callback) = 0;
};

}  // namespace ksana_llm
