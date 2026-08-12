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
#include "batch_scheduler/event_driven_scheduler/workload_balance/pp_batch_distributor.h"
#include "batch_scheduler/event_driven_scheduler/workload_balance/workload_balancer_interface.h"
#include "batch_scheduler/global_cache_put_tracker.h"
#include "batch_scheduler/schedule_processor_interface.h"
#include "cache_manager/cache_manager_interface.h"
#include "utils/blocking_queue.h"
#include "utils/status.h"

namespace ksana_llm {

// Read-only snapshot of this engine's scheduler load, sourced from the
// EVENT_DRIVEN scheduler state. Used by the pd_v2 snapshot provider (Prefill
// side) to populate the load fields of PeerHealth. Kept in this header (no
// pd_v2 dependency) so batch_scheduler stays decoupled from the connector.
struct ScheduleLoadStat {
  uint32_t waiting_reqs = 0;            // queued (prefilling) request count
  uint32_t running_reqs = 0;            // running (decoding) request count
  uint64_t pending_prefill_tokens = 0;  // sum of planning sequence lengths over prefilling reqs
  uint32_t step_token_num = 0;          // EWMA of tokens processed per schedule step
  uint32_t step_latency_us = 0;         // EWMA of per-step wall time (microseconds)
  uint32_t step_latency_us_inst = 0;    // most recent (non-EWMA) per-step wall time; for the
                                        // 50ms TPOT gate, which must react to a load spike on the
                                        // very next decision instead of lagging behind the EWMA
};

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
  void SetPpBatchDistributor(std::shared_ptr<PpBatchDistributor> distributor);

  // 设置全局 cache put 追踪器。设置后 HandleStopRequest 等会在释放 block 前
  // 检查是否有 pending put 操作，避免 put 正在进行时释放 block 导致 use-after-free。
  void SetGlobalCachePutTracker(std::shared_ptr<GlobalCachePutTracker> tracker);

  // pd_v2 Decode 角色标记。PD decode 下 KV block 分配的唯一所有者是
  // GlobalCacheQueryHook (defer_admission_on_block_full_)，backend ProcessPrefillingQueue
  // 不负责分配。因此 recompute 不能把请求直接塞回 prefilling_reqs (那是非 PD 的分配模型)，
  // 必须经 HandlePdV2DecodeRestart 重新投递走完整 hook 链重新分配。由 engine bootstrap
  // 用 (fetcher_role == "pd_v2_decode") 设置，与 GlobalCacheQueryHook 的
  // defer_admission_on_block_full 同源。
  void SetPdV2DecodeMode(bool enabled) { pd_v2_decode_mode_ = enabled; }

  ScheduleState* GetScheduleState() { return &schedule_state_; }

  // Read-only load snapshot for the pd_v2 Prefill snapshot provider. The
  // queue three counts (waiting/running/pending_prefill_tokens) iterate
  // prefilling_reqs, so they are read under infer_request_mutex_ to avoid a
  // use-after-free against the scheduler thread's push_back/erase. The two
  // per-step EWMA values are lock-free atomics written by the scheduler
  // thread. Called from the connector heartbeat / completion threads.
  ScheduleLoadStat GetLocalLoad();

  // Same read as GetLocalLoad but WITHOUT taking infer_request_mutex_. The
  // caller MUST already hold infer_request_mutex_. This exists for the kLinUCB
  // PdV2DecodeHook, which calls back here from inside DispatchEvent — a context
  // that already holds the lock (AsyncSchedulerLoop). Using the locking
  // GetLocalLoad() there self-deadlocks (infer_request_mutex_ is non-recursive).
  ScheduleLoadStat GetLocalLoadUnlocked();

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

  // Look up the PP batch index for req_id in req_to_batch_idx.
  // Throws KLLM_THROW with caller_prefix if not found (state machine bug).
  // The req_to_batch_idx mapping is immutable after assignment — see event_driven_state.h.
  size_t LookupReqBatchIdx(int64_t req_id, const char* caller) const;

  // pd_v2 Decode-side preempt path (swap removal). A kv_from_remote request
  // selected as a recompute victim cannot be re-prefilled locally (would
  // overwrite peer-written KV with a divergent attention path). Instead we
  // release its local KV blocks, reset scheduling/PD state while preserving
  // output_tokens (already-generated tokens), and re-inject it through
  // AddInferRequest so the full hook chain (GlobalCacheQueryHook +
  // PdV2DecodeHook) re-runs over the accumulated sequence and re-fetches KV
  // from a peer Prefill.
  void HandlePdV2DecodeRestart(const std::shared_ptr<InferRequest>& req, size_t dp_idx);

  // return true if resource state changed.
  bool ApplyBackendEventsToState(const std::vector<ScheduleBackendEvent>& backend_events);
  bool ProcessDelayedBackendEvents();

  // Ops load snapshot every 10s (INFO), field names/format aligned with
  // BatchScheduler::ReportTotalState for the same online monitoring path.
  // Mapping today is transitional; will later switch to event-driven-native
  // counters (see implementation comment for running/decoding overlap).
  void ReportEventDrivenState() const;

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
  std::shared_ptr<PpBatchDistributor> pp_batch_distributor_ = nullptr;

  // True when this node is a pd_v2 Decode role (GlobalCacheQueryHook owns block
  // allocation). Gates the recompute reinjection path in HandleRecomputeRequest.
  bool pd_v2_decode_mode_ = false;

  // EventGetScheduleResult 时若无 launchable batch 则置 true；事件队列排空并执行 Schedule 后尝试放入 result_queue_
  bool pending_get_schedule_result_ = false;

  // 调度后端
  std::shared_ptr<ScheduleBackendInterface> backend_;

  // One DP corresponds to one cache manager, index is dp_idx.
  std::vector<std::shared_ptr<CacheManagerInterface>> dp_cache_managers_;

  // 外部 Lock/Unlock 与主线程修改 InferRequest 的互斥控制。
  std::mutex infer_request_mutex_;

  // Global pipeline depth counter. Limits the number of in-flight ScheduleResult batches
  // to prevent exceeding the executor's ModelInput slot count (max 2).
  // Protected by the single-threaded AsyncSchedulerLoop — no atomic needed.
  static constexpr size_t kMaxPipelineDepth = 2;
  size_t pipeline_depth_{0};

  // Thread id of AsyncSchedulerLoop, captured at loop entry. GetLocalLoad()
  // uses it to detect re-entry from the scheduler thread (which already holds
  // infer_request_mutex_ during DispatchEvent) and skip re-locking the
  // non-recursive mutex. Default id (no loop running) compares unequal to any
  // real thread, so off-thread callers always take the lock.
  std::atomic<std::thread::id> async_loop_tid_{};

  // 延迟执行的 Stop/Recompute 请求级操作（等待 HasInflightTask()==false）。
  std::vector<ScheduleBackendEvent> delayed_backend_events_;

  // 全局 cache put 追踪器（延迟析构，防止 put 进行中释放 block）
  std::shared_ptr<GlobalCachePutTracker> global_cache_put_tracker_;

  size_t schedule_id_ = DEFAULT_SCHEDULE_ID;

  // Per-step throughput/latency EWMA feeding ScheduleLoadStat. Written only
  // by the scheduler thread (launch records the timestamp + token count,
  // generation-result return folds them into the EWMA), read lock-free by
  // GetLocalLoad() on other threads.
  std::atomic<uint32_t> step_token_ewma_{0};
  std::atomic<uint32_t> step_latency_us_ewma_{0};
  // Most recent (non-EWMA) per-step wall time, published for the TPOT gate.
  std::atomic<uint32_t> step_latency_us_inst_{0};
  // Scheduler-thread-only scratch: timestamp and token count of the most
  // recently launched step, consumed when its generation result returns.
  int64_t last_step_launch_ms_ = 0;
  uint32_t last_step_token_num_ = 0;

  // Fold the most recently launched step's token count and measured wall
  // time into the EWMA atomics. Scheduler-thread only.
  void RecordStepLatency();
};

}  // namespace ksana_llm
