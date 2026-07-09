/* Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * 事件驱动调度器：事件类型与 payload 定义，与设计文档 4 / 3.6 节对应。
 * ==============================================================================*/
#pragma once

#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "runtime/infer_request.h"
#include "runtime/schedule_output.h"
#include "utils/status.h"

namespace ksana_llm {

// ----- 调度事件类型（设计文档 3.6） -----
enum class SchedEventType {
  EventAddRequest,
  EventUpdateAfterGeneration,
  EventAsyncTaskFinished,
  EventGetScheduleResult,  // 消费者请求下一个结果；主线程收到后才将 PlanningBatch 组织成 ScheduleResult 放入
                           // result_queue_
};

inline std::string SchedEventType2Str(SchedEventType e) {
  switch (e) {
    case SchedEventType::EventAddRequest:
      return "EventAddRequest";
    case SchedEventType::EventUpdateAfterGeneration:
      return "EventUpdateAfterGeneration";
    case SchedEventType::EventAsyncTaskFinished:
      return "EventAsyncTaskFinished";
    case SchedEventType::EventGetScheduleResult:
      return "EventGetScheduleResult";
    default:
      return "Wrong type";
  }
}

// ----- Hook 通知事件（通用化的 Hook 通知机制） -----
// 调度器以"事件"的方式通知 Hook 状态发生了变化，而不是为每个具体操作定义一个专用接口。
// Hook 按需处理自己关心的事件类型，不关心的事件默认忽略 (基类 OnEvent 空实现)。
// 事件驱动 Hook 行为: 资源被释放、请求离场等都表达为一个事件，未来新增事件类型无需改动接口签名。
enum class HookEventType {
  kBlocksFreed,    // KV blocks 被释放 (Stop / Recompute / Swapout)，资源可用，可重试分配等。
  kRequestExited,  // 某个请求彻底离场 (正常完成 / 失败结束 / pd_v2 decode 抢占重启释放本地 KV)。
};

inline std::string HookEventType2Str(HookEventType e) {
  switch (e) {
    case HookEventType::kBlocksFreed:
      return "kBlocksFreed";
    case HookEventType::kRequestExited:
      return "kRequestExited";
    default:
      return "Wrong type";
  }
}

// 通用 Hook 事件载体。不同事件类型按需读取相关字段，无关字段忽略。
struct HookEvent {
  HookEventType type;
  // kRequestExited 携带离场的 req_id；其他事件类型忽略此字段。
  int64_t req_id = 0;

  explicit HookEvent(HookEventType t, int64_t rid = 0) : type(t), req_id(rid) {}
};

// ----- 异步任务结果（设计文档 4.4） -----
struct AsyncTaskResult {
  int hook_idx = 0;  // 标识从哪个 Hook 开始调用，调用顺序为 hook_idx 从小到大
  int task_id = 0;   // 标识hook内的具体异步任务，供Hook内部使用
  int task_ret_state = 0;
};

// ----- 事件 payload 基类（继承多态） -----
struct SchedEventInfoBase {
  virtual ~SchedEventInfoBase() = default;
  virtual SchedEventType GetType() const = 0;
};

// ----- 各事件 payload（设计文档 4.1 ~ 4.4） -----
struct EventAddRequestInfo : public SchedEventInfoBase {
  std::vector<std::shared_ptr<InferRequest>> reqs;
  SchedEventType GetType() const override { return SchedEventType::EventAddRequest; }
};

struct EventUpdateAfterGenerationInfo : public SchedEventInfoBase {
  GenerationOutputGroup gen_results;
  SchedEventType GetType() const override { return SchedEventType::EventUpdateAfterGeneration; }
};

struct EventAsyncTaskFinishedInfo : public SchedEventInfoBase {
  std::vector<AsyncTaskResult> task_results;
  SchedEventType GetType() const override { return SchedEventType::EventAsyncTaskFinished; }
};

struct EventGetScheduleResultInfo : public SchedEventInfoBase {
  SchedEventType GetType() const override { return SchedEventType::EventGetScheduleResult; }
};

// ----- 调度事件（设计文档 3.6） -----
struct SchedEvent {
  std::shared_ptr<SchedEventInfoBase> event_info;

  SchedEvent() = default;

  template <typename EventInfoType, typename = std::enable_if_t<std::is_base_of_v<SchedEventInfoBase, EventInfoType>>>
  explicit SchedEvent(std::shared_ptr<EventInfoType> info) : event_info(std::move(info)) {}

  bool IsEmpty() const { return event_info == nullptr; }

  SchedEventType GetType() const {
    return event_info == nullptr ? SchedEventType::EventGetScheduleResult : event_info->GetType();
  }
};

}  // namespace ksana_llm
