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
