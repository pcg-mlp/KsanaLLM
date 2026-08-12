/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "batch_scheduler/schedule_processor_interface.h"
#include "utils/blocking_queue.h"
#include "utils/status.h"

namespace ksana_llm {

// The schedule processor.
class ScheduleProcessor : public ScheduleProcessorInterface {
 public:
  explicit ScheduleProcessor(const bool enable_async, const size_t max_pp_batch_num = 1)
      : enable_async_(enable_async), max_pp_batch_num_(max_pp_batch_num) {}
  ~ScheduleProcessor() override = default;

  void Initialize(std::shared_ptr<BatchSchedulerInterface> batch_scheduler) override;

  Status AddInferRequest(std::vector<std::shared_ptr<InferRequest>>& infer_request_group) override;

  // The sync mode: call Schedule directly -> check for running requests -> wait or process data
  std::shared_ptr<ScheduleResult> GetNextScheduleResult() override;

  void UpdateWithGenerationResult(const GenerationOutputGroup& generation_output) override;

  // After a ScheduleOutput is dispatched, release the inflight task for
  // every chunked-mid req it carried, then consume any parked
  // ScheduleResult on the next multi-batch that may now be launchable.
  void NotifyNextPPMultiBatch(size_t multi_batch_id, const std::shared_ptr<ScheduleOutput>& schedule_output) override;

  void Lock() override;
  void Unlock() override;

  void Start() override;
  void Stop() override;

 protected:
  void AsyncScheduleThread(size_t multi_batch_id);
  void SyncScheduleThread(size_t multi_batch_id);

  std::shared_ptr<ScheduleResult> GetScheduleResultInSync(size_t multi_batch_id);

  std::shared_ptr<ScheduleResult> Schedule(size_t multi_batch_id);

  // Internal helper method: processes data for next forwarding.
  virtual Status ProcessScheduleDataInternal(size_t multi_batch_id, ScheduleResult& result);

  // Try to consume a parked planning_sched_results_[multi_batch_id]
  // if it is now launchable.
  bool TryConsumeScheduleResult(size_t multi_batch_id);

 private:
  std::atomic<bool> terminated_{false};

  const bool enable_async_;
  const size_t max_pp_batch_num_;
  std::vector<std::thread> sched_threads_;
  BlockingQueue<std::shared_ptr<ScheduleResult>> sched_result_queue_;

  // To guard whether ready to produce next schedule result.
  std::mutex schedule_guard_;
  std::condition_variable schedule_guard_cv_;

  // To guard schedule results for sync mode.
  std::vector<bool> sync_generating_finished_;

  // To cache planning results for async mode.
  std::vector<std::shared_ptr<ScheduleResult>> planning_sched_results_;
};

}  // namespace ksana_llm
