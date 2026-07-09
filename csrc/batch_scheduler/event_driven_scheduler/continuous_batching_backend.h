/* Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * 事件驱动调度器：参考 ContinuousBatchingStrategy 的 ProcessDecodingQueue / ProcessPrefillingQueue
 * 实现的 ScheduleBackendInterface，仅处理 decoding_reqs 与 prefilling_reqs，调度结果暂存于 DpBatch.running_reqs，
 * 当某 batch idx 下所有 DpBatch 的 running_reqs 均 Launchable 时再写入 schedule_output->running_reqs。
 * ==============================================================================*/
#pragma once

#include <atomic>
#include <unordered_set>

#include "batch_scheduler/event_driven_scheduler/event_driven_state.h"
#include "batch_scheduler/event_driven_scheduler/schedule_backend_interface.h"

namespace ksana_llm {

class ContinuousBatchingBackend : public ScheduleBackendInterface {
 public:
  explicit ContinuousBatchingBackend(const BatchSchedulerConfig& batch_scheduler_config = {});
  ~ContinuousBatchingBackend() override = default;
  void Schedule(const ScheduleContext& context, std::vector<std::shared_ptr<PlanningBatch>>& planning_batches,
                std::vector<ScheduleBackendEvent>& events) override;
  void UpdateWithGenerationResult(size_t multi_batch_id, const GenerationOutputGroup& generation_output,
                                  const ScheduleContext& context, std::vector<ScheduleBackendEvent>& events) override;
  void Stop() override;

 private:
  void ProcessDecodingQueue(const ScheduleContext& context, PlanningBatch& planning_batch,
                            std::vector<ScheduleBackendEvent>& events,
                            std::unordered_set<uint64_t>& emitted_event_keys);
  void ProcessPrefillingQueue(const ScheduleContext& context, PlanningBatch& planning_batch,
                              std::vector<ScheduleBackendEvent>& events,
                              std::unordered_set<uint64_t>& emitted_event_keys);

  // pd_v2 下，preprocessing_reqs 可能包含正在等待对端 PrefillComplete 的请求。
  // 这类请求被客户端取消时，只标记 pending_stop，不立即发 kStopRequest，避免 RDMA 尚未完成时释放 KV。
  // PdV2DecodeHook 收到 OnPrefillComplete 后再统一 FailAndEnd。
  // 非PD模式下的 preprocessing 队列中的取消请求，仍在这里立即发 kStopRequest。
  void ScanAbortedPreprocessingReqs(const ScheduleContext& context, const PlanningBatch& planning_batch,
                                    std::vector<ScheduleBackendEvent>& events,
                                    std::unordered_set<uint64_t>& emitted_event_keys);

 private:
  std::atomic<bool> stopped_{false};
};

}  // namespace ksana_llm
