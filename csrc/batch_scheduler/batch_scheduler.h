/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <chrono>
#include <condition_variable>
#include <memory>
#include <vector>

#include "batch_scheduler/batch_scheduler_balance_reqs_algo.h"
#include "batch_scheduler/batch_scheduler_interface.h"
#include "batch_scheduler/state/batch_state.h"
#include "batch_scheduler/state/scheduler_shared_counter.h"
#include "batch_scheduler/state/scheduler_tick_tok.h"
#include "batch_scheduler/strategy/strategy_factory.h"
#include "batch_scheduler/workload_balance/pp_multibatch_balancer.h"
#include "runtime/infer_request.h"

#include "cache_manager/cache_manager_interface.h"

#include "batch_scheduler/schedule_output_process.h"
#include "configure/environment.h"
#include "runtime/threadpool.h"
#include "transfer/transfer_engine.h"
#include "utils/status.h"

namespace ksana_llm {

class GlobalCachePutTracker;

class BatchScheduler : public BatchSchedulerInterface {
 public:
  BatchScheduler(const BatchSchedulerConfig &batch_scheduler_config, const RuntimeConfig &runtime_config,
                 std::shared_ptr<ScheduleOutputPool> schedule_output_pool);
  ~BatchScheduler();

  // Get the next infer reqs that ready to run.
  std::shared_ptr<ScheduleOutputGroup> Schedule(size_t multi_batch_id) override;

  bool TryToLaunchPlannedScheduleOutput(size_t multi_batch_id, ScheduleOutput &planned_schedule_output,
                                        std::vector<std::shared_ptr<InferRequest>> &stopped_reqs) override;

  void UpdateWithGenerationResult(size_t multi_batch_id, const GenerationOutputGroup &generation_output) override;

  // Add infer request to waiting list.
  Status AddInferRequest(std::vector<std::shared_ptr<InferRequest>> &infer_request_group) override;

  // Set the cache manager instance of batch scheduler.
  void SetCacheManager(std::shared_ptr<CacheManagerInterface> cache_manager, int attn_dp_idx) override;

  std::shared_ptr<CacheManagerInterface> &GetCacheManager(int attn_dp_idx) override;

  // Set the transfer engine instance for all schedule strategies.
  void SetTransferEngine(std::shared_ptr<TransferEngine> transfer_engine);

  // 设置全局 cache put 追踪器，用于延迟析构。
  void SetGlobalCachePutTracker(std::shared_ptr<GlobalCachePutTracker> tracker);

  // Whether the scheduler is idle, that is, waiting buffer is empty.
  bool IsIdle(size_t multi_batch_id) override;

  void WaitUntilHaveReqs(size_t multi_batch_id) override;

  std::vector<std::shared_ptr<InferRequest>> GetMockRequest() { return mock_request_group_; }

  void Stop() override;

 private:
  // Add infer requests to waiting buffer queue, and reject requests if the queue is full.
  Status EnqueueWaitingBufferQueue(std::vector<std::shared_ptr<InferRequest>> &infer_request_group);

  // balance waiting reqs to dp_waiting_reqs_ by batch_state_
  // the output is dp_waiting_reqs_
  void BalanceWaitingReqs();

  void BalancePPMultiBatchReqs(size_t multi_batch_id);

  // Shift chunked-prefill reqs from `multi_batch_id` onto the next multi_batch's
  // waiting_queue front, so the next chunk runs on a different multi_batch
  // for better PP pipeline overlap.
  void ShiftChunkedReqsAcrossPPMultiBatch(size_t multi_batch_id);

  void ReportBatchState(std::shared_ptr<BatchState> batch_state, size_t dp_idx);

  Status CreateMockReq(const RuntimeConfig &runtime_config,
                       std::vector<std::shared_ptr<InferRequest>> &infer_request_group);

  // report the state of all instance
  void ReportTotalState();

  // ADP Balance related methods
  void BalanceADPRequests(size_t multi_batch_id);
  void CalculatePrefillWorkload(std::vector<float> &workload);
  void DistributeRequestsByToken(
      const std::vector<std::pair<size_t, std::shared_ptr<InferRequest>>> &requests_with_tokens);
  bool IsADPBalanceTimeout();
  void UpdateQPS(size_t request_count);
  size_t GetCurrentPrefillAccumulationMaxSteps();
  size_t GetCurrentPrefillAccumulationMaxTimeMs();

 private:
  // The config of batch scheduler.
  BatchSchedulerConfig batch_scheduler_config_;

  size_t dp_num_;
  size_t pp_batch_num_;
  bool always_return_launchable_req_ = false;  // Always return launchable request when expert parallel world size>1

  // object pool of schedule output buffer.
  std::shared_ptr<ScheduleOutputPool> schedule_output_pool_ = nullptr;

  // The thread pool of batch scheduler.
  std::unique_ptr<ThreadPool> threadpool_ = nullptr;

  // The batch state informations, include some queues and mutexes. [dp_idx, multi_batch_id]
  std::vector<std::vector<std::shared_ptr<BatchState>>> batch_states_;

  // The buffer queue needed be scheduled in strategy.
  std::vector<std::shared_ptr<InferRequest>> waiting_reqs_;
  // Protect the waiting_reqs_.
  std::mutex waiting_reqs_mutex_;
  // Signalled after BalanceWaitingReqs drains waiting_reqs_, so producers blocked in
  // EnqueueWaitingBufferQueue waiting for staging space wake up on the next schedule step.
  std::condition_variable waiting_reqs_cv_;

  // The buffer queue needed be scheduled in strategy
  std::vector<std::vector<std::shared_ptr<InferRequest>>> dp_waiting_reqs_;

  // The batch strategy implementations.
  std::vector<std::shared_ptr<BaseScheduleStrategy>> schedule_strategies_;

  // 全局 cache put 追踪器（延迟析构）
  std::shared_ptr<GlobalCachePutTracker> global_cache_put_tracker_;

  // Balance requests algorithm
  std::unique_ptr<BalanceReqsAlgo> balance_reqs_algo_ = nullptr;

  // Balance requests among multiple batchs in pipeline parallel.
  // One instance per dp so per-dp state (e.g. starvation hold_iter and
  // held_waiting_reqs_) does not leak across dp groups.
  std::vector<std::unique_ptr<PPMultibatchWorkloadBalancer>> pp_multibatch_wl_balancers_;

  // NOTE(karlluo, jackyjtang): The thread pool is not thread safe, so we need to be temp variable
  // group of all strategy schedule outputs
  std::shared_ptr<ScheduleOutputGroup> schedule_output_group_;

  bool terminating_ = false;
  std::vector<std::shared_ptr<InferRequest>> mock_request_group_;

  // To avoid variables destruction， while mock req will reference KsanaPythonInput and Request.
  std::shared_ptr<Request> alias_mock_request_;
  std::shared_ptr<KsanaPythonInput> alias_python_input_;

  std::shared_ptr<SchedulerSharedCounter> scheduler_shared_counter_ = nullptr;
  std::shared_ptr<SchedulerTickTok> scheduler_ticktok_ = nullptr;

  // ADP Balance related members
  size_t adp_balance_waiting_iters_ = 0;
  size_t adp_balance_context_batches_ = 0;
  std::chrono::steady_clock::time_point adp_balance_start_time_;

  // QPS tracking members
  std::chrono::steady_clock::time_point last_qps_update_time_;
  size_t request_count_window_ = 0;
  double current_qps_ = 0.0;
  static constexpr size_t QPS_UPDATE_INTERVAL_SECONDS = 10;

  // Executor 全局流水线深度（build-ahead 路径）: 已 LaunchScheduleOutput、尚未 sampling 回流的 schedule 数。
  // 与 ModelInput 双 slot / max_depth=2 对齐：在途 launch 不得超过 2，launch(N+1) 前须 retire(N-1)。
  size_t executor_pipeline_depth_ = 0;
  static constexpr size_t kExecutorPipelineMaxDepth = 2;

  // Balance stats
  std::vector<std::vector<size_t>> last_step_tokens_;  // last observed step_token_num per (dp_idx, multi_batch_id)
  double accumulated_balance_ratio_ = 0.0;             // sum of min/max(step_token) ratio across rounds
  size_t accumulated_balance_round_ = 0;
};

}  // namespace ksana_llm
