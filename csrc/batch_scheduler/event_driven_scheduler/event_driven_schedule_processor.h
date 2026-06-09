/* Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * 事件驱动调度器：单主线程 + 事件队列，实现 ScheduleProcessorInterface。
 * ==============================================================================*/
#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "batch_scheduler/event_driven_scheduler/event_driven_state.h"
#include "batch_scheduler/event_driven_scheduler/event_driven_types.h"
#include "batch_scheduler/event_driven_scheduler/schedule_backend_interface.h"
#include "batch_scheduler/event_driven_scheduler/schedule_context.h"
#include "batch_scheduler/event_driven_scheduler/schedule_hook_manager.h"
#include "batch_scheduler/event_driven_scheduler/schedule_hook_manager_interface.h"
#include "batch_scheduler/event_driven_scheduler/workload_balance/workload_balancer_interface.h"
#include "batch_scheduler/global_cache_put_tracker.h"
#include "batch_scheduler/schedule_processor_interface.h"
#include "cache_manager/cache_manager_interface.h"
#include "utils/blocking_queue.h"
#include "utils/status.h"

namespace ksana_llm {

class EventDrivenScheduleProcessor : public ScheduleProcessorInterface {
 public:
  explicit EventDrivenScheduleProcessor(size_t dp_num, size_t max_pp_batch_num);
  ~EventDrivenScheduleProcessor() override;

  void Initialize(std::shared_ptr<BatchSchedulerInterface> batch_scheduler) override;

  // Set schedule backend explicitly.
  // NOTE: backend_ will NOT be initialized in Initialize(). Caller must set it before Start().
  void SetScheduleBackend(std::shared_ptr<ScheduleBackendInterface> backend);

  // Set all dp cache managers explicitly.
  // After this is set, EventDrivenScheduleProcessor will no longer fetch cache managers from batch_scheduler_.
  void SetCacheManagers(std::vector<std::shared_ptr<CacheManagerInterface>> dp_cache_managers);

  Status AddInferRequest(std::vector<std::shared_ptr<InferRequest>>& infer_request_group) override;

  std::shared_ptr<ScheduleResult> GetNextScheduleResult() override;

  void UpdateWithGenerationResult(const GenerationOutputGroup& generation_output) override;
  void Lock() override;
  void Unlock() override;

  // 设计文档 5.3.2：异步 task 完成时由调度主线程调用
  void UpdateAfterAsyncTaskFinished(const EventAsyncTaskFinishedInfo& event_info);

  void Start() override;
  void Stop() override;

  // 设置外部 Hook 管理器；若为 nullptr 则使用内置的 default_schedule_hook_manager_。
  // 设置时会自动将 UpdateAfterAsyncTaskFinished 注册为该管理器的异步完成回调。
  void SetScheduleHookManager(std::shared_ptr<ScheduleHookManagerInterface> hook_manager);
  void SetWorkloadBalancer(std::shared_ptr<WorkloadBalancerInterface> workload_balancer);

  // 设置全局 cache put 追踪器。设置后 HandleStopRequest 等会在释放 block 前
  // 检查是否有 pending put 操作，避免 put 正在进行时释放 block 导致 use-after-free。
  void SetGlobalCachePutTracker(std::shared_ptr<GlobalCachePutTracker> tracker);

  ScheduleState* GetScheduleState() { return &schedule_state_; }

 private:
  void AsyncSchedulerLoop();
  void DispatchEvent(SchedEvent& event);
  void OnEventAddRequest(const std::shared_ptr<EventAddRequestInfo>& info);
  void OnEventUpdateAfterGeneration(const std::shared_ptr<EventUpdateAfterGenerationInfo>& info);
  void OnEventAsyncTaskFinished(const std::shared_ptr<EventAsyncTaskFinishedInfo>& info);
  void OnEventGetScheduleResult();  // 仅设置 pending_get_schedule_result_；在事件处理完后由 SchedulerLoop
                                    // 检查并尝试放入 launchable batch
  void EnqueueAsyncTaskFinished(const EventAsyncTaskFinishedInfo& event_info);

  // Backend event application helpers.
  void HandleStopRequest(const std::shared_ptr<InferRequest>& req, Status ret_status, size_t dp_idx);
  void HandleHookFailAndEnd(const std::shared_ptr<InferRequest>& req, size_t dp_idx, const std::string& reason);
  void HandleRecomputeRequest(const std::shared_ptr<InferRequest>& req, Status ret_status, size_t dp_idx);

  // return true if resource state changed.
  bool ApplyBackendEventsToState(const std::vector<ScheduleBackendEvent>& backend_events);
  bool ProcessDelayedBackendEvents();

  void RunScheduleForBatches();
  bool TryBuildAndLaunchScheduleResult(ScheduleContext& ctx);
  Status ProcessScheduleDataInternal(size_t multi_batch_id, ScheduleResult& result);

  std::atomic<bool> terminated_{false};
  const size_t max_pp_batch_num_;
  size_t dp_num_ = 0;
  std::thread sched_thread_;
  BlockingQueue<SchedEvent> event_queue_;
  BlockingQueue<std::shared_ptr<ScheduleResult>> result_queue_;

  ScheduleState schedule_state_;
  std::vector<std::shared_ptr<PlanningBatch>> planning_batches_;

  // Reused schedule context to avoid per-event construction.
  // NOTE: This object is only accessed/updated under infer_request_mutex_.
  ScheduleContext schedule_context_;

  std::shared_ptr<ScheduleHookManagerInterface> schedule_hook_manager_ = nullptr;
  std::shared_ptr<WorkloadBalancerInterface> workload_balancer_ = nullptr;

  // EventGetScheduleResult 时若无 launchable batch 则置 true；事件队列排空并执行 Schedule 后尝试放入 result_queue_
  bool pending_get_schedule_result_ = false;

  // 调度后端
  std::shared_ptr<ScheduleBackendInterface> backend_;

  // One DP corresponds to one cache manager, index is dp_idx.
  std::vector<std::shared_ptr<CacheManagerInterface>> dp_cache_managers_;

  // 外部 Lock/Unlock 与主线程修改 InferRequest 的互斥控制。
  std::mutex infer_request_mutex_;

  // 延迟执行的 Stop/Recompute 请求级操作（等待 HasInflightTask()==false）。
  std::vector<ScheduleBackendEvent> delayed_backend_events_;

  // 全局 cache put 追踪器（延迟析构，防止 put 进行中释放 block）
  std::shared_ptr<GlobalCachePutTracker> global_cache_put_tracker_;

  size_t schedule_id_ = DEFAULT_SCHEDULE_ID;
};

}  // namespace ksana_llm
