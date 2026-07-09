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

  // 通用化的事件通知入口: 调度器以"事件"的方式通知 Hook 状态变化 (KV blocks 释放 /
  // 请求离场 / ...)，Hook 的行为由事件驱动。每个 Hook 只处理自己关心的事件类型，
  // 不关心的事件直接忽略 (基类默认空实现)。相比为每个具体操作定义一个专用接口，
  // 这样新增事件类型时无需改动接口签名，也不强制所有 Hook 实现无关的方法。
  //   - kBlocksFreed: 例如 GlobalCacheQueryHook 借此唤醒 deferred admission 队列重试
  //     AllocateRequestBlocks。
  //   - kRequestExited: 例如 GlobalCacheQueryHook 借此维护 "已准入但未离场" 的 in-flight
  //     req 集合 (幂等: 按 req_id erase, 不在集合中即 no-op), 给 pd_v2 Decode 的
  //     awaiting-prefill backlog 设并发上限, 避免大量在飞 req 预占 KV 块把池抽干。
  virtual void OnEvent(const HookEvent& /*event*/) {}
};

}  // namespace ksana_llm
