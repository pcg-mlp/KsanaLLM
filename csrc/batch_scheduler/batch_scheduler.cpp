/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "batch_scheduler/batch_scheduler.h"

#include <algorithm>
#include <chrono>
#include <future>
#include <iomanip>
#include <limits>
#include <memory>
#include <numeric>
#include <thread>
#include <utility>
#include <vector>

#include "runtime/speculative/common/speculative_resource_profile.h"

#include "batch_scheduler/global_cache_put_tracker.h"
#include "batch_scheduler/request_admission.h"
#include "batch_scheduler/strategy/continuous_batching.h"
#include "profiler/reporter.h"
#include "profiler/sched_event_tracer.h"
#include "runtime/fast_path_controller.h"
#include "runtime/infer_request.h"
#include "utils/channel.h"
#include "utils/logger.h"
#include "utils/memory_utils.h"
#include "utils/ret_code.h"
#include "utils/singleton.h"
#include "utils/status.h"
#include "utils/string_utils.h"

namespace ksana_llm {

BatchScheduler::BatchScheduler(const BatchSchedulerConfig& batch_scheduler_config, const RuntimeConfig& runtime_config,
                               std::shared_ptr<ScheduleOutputPool> schedule_output_pool)
    : batch_scheduler_config_(batch_scheduler_config),
      dp_num_(runtime_config.parallel_basic_config.attn_data_parallel_size),
      schedule_output_pool_(schedule_output_pool) {
  pp_batch_num_ = batch_scheduler_config_.max_pp_batch_num > 0 ? batch_scheduler_config_.max_pp_batch_num : 1;

  // max_waiting_queue_len is for each strategy
  waiting_reqs_.reserve(batch_scheduler_config_.max_waiting_queue_len * dp_num_);

  schedule_output_group_ = std::make_shared<ScheduleOutputGroup>(dp_num_);
  KLLM_LOG_DEBUG << "pp_batch_num_=" << pp_batch_num_ << ", batch_scheduler_config_.pp_multibatch_wb_strategy.enable="
                 << batch_scheduler_config_.pp_multibatch_wb_strategy.enable;
  if (pp_batch_num_ > 1 && batch_scheduler_config_.pp_multibatch_wb_strategy.enable) {
    pp_multibatch_wl_balancers_.reserve(dp_num_);
    for (size_t i = 0; i < dp_num_; ++i) {
      pp_multibatch_wl_balancers_.emplace_back(std::make_unique<PPMultibatchWorkloadBalancer>(batch_scheduler_config_));
    }
  }
  balance_reqs_algo_ = std::make_unique<BalanceReqsAlgo>();
  threadpool_ = std::make_unique<ThreadPool>(dp_num_);
  threadpool_->Start();

  scheduler_shared_counter_ = std::make_shared<SchedulerSharedCounter>(dp_num_);
  scheduler_ticktok_ = std::make_shared<SchedulerTickTok>(dp_num_);

  schedule_strategies_.resize(dp_num_);
  batch_states_.resize(dp_num_);
  dp_waiting_reqs_.resize(dp_num_);

  for (int i = 0; i < dp_num_; i++) {
    schedule_strategies_[i] = ScheduleStrategyFactory::CreateScheduleStrategy(batch_scheduler_config_, runtime_config);
    schedule_strategies_[i]->SetSharedCounter(scheduler_shared_counter_);
    schedule_strategies_[i]->SetSchedulerTickTok(scheduler_ticktok_);
    schedule_strategies_[i]->SetDataParaGroupId(i);

    batch_states_[i].resize(pp_batch_num_);
    for (size_t j = 0; j < pp_batch_num_; j++) {
      batch_states_[i][j] = std::make_shared<BatchState>(j, batch_scheduler_config_, schedule_output_pool_);
    }
    dp_waiting_reqs_[i].reserve(batch_scheduler_config_.max_waiting_queue_len);
  }

  // Initialize PP balance stats: last_step_tokens_[dp_idx][multi_batch_id]
  last_step_tokens_.assign(dp_num_, std::vector<size_t>(pp_batch_num_, 0));

  // Create mock request to make sure there are always launchable requests.
  always_return_launchable_req_ = runtime_config.parallel_basic_config.expert_world_size > 1;
  if (always_return_launchable_req_) {
    CreateMockReq(runtime_config, mock_request_group_);
    if (mock_request_group_.size() >= 1) {
      for (int i = 0; i < pp_batch_num_; i++) {
        batch_states_[0][i]->mock_queue.push_back(mock_request_group_[0]);
        std::shared_ptr<InferRequest> req = *batch_states_[0][i]->mock_queue.begin();

        KLLM_LOG_DEBUG << "req_id " << req->req_id << ", input_tokens_num " << req->input_tokens.size()
                       << ", output_tokens_num " << req->output_tokens.size() << ", InferRequest addr " << req
                       << ", output_tokens addr" << req->output_tokens.data() << ", input_tokens addr "
                       << req->input_tokens.data();
      }
    }
  }

  // Initialize QPS tracking
  last_qps_update_time_ = std::chrono::steady_clock::now();
  request_count_window_ = 0;
  current_qps_ = 0.0;
}

BatchScheduler::~BatchScheduler() { threadpool_->Stop(); }

void BatchScheduler::Stop() {
  {
    // Set the flag under the lock the waiters use, so a producer cannot evaluate its predicate
    // and go back to sleep just before the notify.
    std::lock_guard<std::mutex> guard(waiting_reqs_mutex_);
    terminating_ = true;
  }
  // Wake producers blocked waiting for staging space so shutdown does not have to wait out
  // their timeout. They observe terminating_ and fail their request.
  waiting_reqs_cv_.notify_all();
}

void BatchScheduler::SetCacheManager(std::shared_ptr<CacheManagerInterface> cache_manager, int dp_idx) {
  KLLM_CHECK_WITH_INFO(dp_idx < dp_num_, FormatStr("dp_idx %d is out of range, dp_num_ %zu.", dp_idx, dp_num_));
  schedule_strategies_.at(dp_idx)->SetCacheManager(cache_manager);
}

std::shared_ptr<CacheManagerInterface>& BatchScheduler::GetCacheManager(int dp_idx) {
  KLLM_CHECK_WITH_INFO(dp_idx < dp_num_, FormatStr("dp_idx %d is out of range, dp_num_ %zu.", dp_idx, dp_num_));
  return schedule_strategies_[dp_idx]->GetCacheManager();
}

void BatchScheduler::SetTransferEngine(std::shared_ptr<TransferEngine> transfer_engine) {
  for (size_t i = 0; i < dp_num_; i++) {
    schedule_strategies_[i]->SetTransferEngine(transfer_engine);
  }
}

void BatchScheduler::SetGlobalCachePutTracker(std::shared_ptr<GlobalCachePutTracker> tracker) {
  global_cache_put_tracker_ = tracker;
  for (auto& strategy : schedule_strategies_) {
    strategy->SetGlobalCachePutTracker(tracker);
  }
}

Status BatchScheduler::AddInferRequest(std::vector<std::shared_ptr<InferRequest>>& infer_request_group) {
  std::shared_ptr<InferRequest>& infer_request = infer_request_group[0];
  KLLM_LOG_DEBUG << "batch scheduler add infer req " << infer_request->req_id << ", max_new_tokens "
                 << infer_request->sampling_config.max_new_tokens;

  const Status finish_status = ValidateRequestAdmission(infer_request, batch_scheduler_config_);
  if (!finish_status.OK()) {
    KLLM_LOG_ERROR << "req_id: " << infer_request->req_id
                   << " request admission failed: " << finish_status.GetMessage();

    infer_request->finish_status = finish_status;
    StampOnce(infer_request->finish_time_us);
    for (auto& infer_request : infer_request_group) {
      infer_request->finished = true;
    }
    infer_request->Notify();
    return finish_status;
  }

  UpdateQPS(infer_request_group.size());

  return EnqueueWaitingBufferQueue(infer_request_group);
}

bool BatchScheduler::IsIdle(size_t multi_batch_id) {
  {
    std::lock_guard<std::mutex> guard(waiting_reqs_mutex_);
    if (!waiting_reqs_.empty()) {
      return false;
    }
  }

  // Check if any dp's PP balancer is holding waiting reqs in its private queue.
  for (auto& balancer : pp_multibatch_wl_balancers_) {
    if (balancer->HasHeldWaitingReqs()) {
      return false;
    }
  }

  for (auto& dp_batch_states : batch_states_) {
    auto& batch_state = dp_batch_states[multi_batch_id];
    std::lock_guard<std::mutex> guard(batch_state->queue_mutex);
    bool batch_state_queue_empty = batch_state->decoding_queue.empty() && batch_state->waiting_queue.empty() &&
                                   batch_state->transfer_queue.empty();

    bool have_free_block = batch_state->async_recomputed_reqs.size() == 0;
    // have request and have space
    if ((!batch_state_queue_empty) && have_free_block) {
      return false;
    }
  }

  return true;
}

void BatchScheduler::WaitUntilHaveReqs(size_t multi_batch_id) {
  while (IsIdle(multi_batch_id) && !terminating_) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    ReportTotalState();
    {
      std::lock_guard<std::mutex> guard(schedule_mutex_);
      for (size_t i = 0; i < dp_num_; i++) {
        schedule_strategies_[i]->SetBatchState(batch_states_[i][multi_batch_id]);
      }
    }
  }
}

Status BatchScheduler::EnqueueWaitingBufferQueue(std::vector<std::shared_ptr<InferRequest>>& infer_request_group) {
  std::unique_lock<std::mutex> lock(waiting_reqs_mutex_);

  // waiting_reqs_ is a staging buffer, not the queue requests actually wait in: BalanceWaitingReqs
  // empties it completely once per schedule step, into dp_waiting_reqs_, which has no size limit at
  // all (it is only reserve()d). So a full staging buffer means "more requests arrived between two
  // schedule steps than fit in one drain", which is arrival timing -- not a capacity limit. The
  // engine can be entirely idle and still reject here.
  //
  // Rejecting that burst is wrong. Wait for the next drain instead. The wait is bounded so that a
  // genuinely wedged scheduler still surfaces an error rather than hanging forever; one drain per step
  // means the normal wait is a single step, and the timeout below is several hundred steps of slack. On
  // timeout we return RET_EXCEED_CAPACITY exactly as before, so callers that already treat that code as
  // backpressure (see _submit_with_backpressure in python/ksana_llm/ksana_engine.py) keep working
  // unchanged.
  //
  // Raising max_waiting_queue_len is a complement to this, not a substitute: it widens one drain so a
  // burst never has to wait, but it only helps launch paths that actually set it, and this wait is what
  // keeps every other path from rejecting. (Raising it used to also cost hundreds of MB of device memory
  // per doubling, because it sized the worst-case layers_slot_mapping / layers_block_table in ModelInput.
  // That coupling is gone -- those tensors are now sized by what one forward step can hold, see
  // csrc/models/base/model_input.cpp -- so the cap is free to raise.)
  constexpr auto kStagingBufferWaitTimeout = std::chrono::seconds(10);
  const size_t group_size = infer_request_group.size();
  // A group that cannot fit in an empty buffer will never satisfy the predicate, so waiting would
  // only burn the timeout before failing. Reject it immediately, as before this change.
  const bool fits_at_all = group_size <= batch_scheduler_config_.max_waiting_queue_len;
  const bool has_space = fits_at_all && waiting_reqs_cv_.wait_for(lock, kStagingBufferWaitTimeout, [&] {
    return terminating_ || waiting_reqs_.size() + group_size <= batch_scheduler_config_.max_waiting_queue_len;
  });

  if (!has_space || terminating_) {
    std::shared_ptr<InferRequest>& infer_request = infer_request_group[0];
    KLLM_LOG_ERROR << "waiting queue is full, req " << infer_request << " failed."
                   << " waiting queue size: " << waiting_reqs_.size()
                   << ", max_waiting_queue_len: " << batch_scheduler_config_.max_waiting_queue_len
                   << ", infer_request_group_size: " << group_size << ", terminating: " << terminating_;

    auto finish_status = Status(RET_EXCEED_CAPACITY, "waiting queue is full.");
    infer_request->finish_status = finish_status;
    StampOnce(infer_request->finish_time_us);
    for (auto& req : infer_request_group) {
      req->finished = true;
    }
    infer_request->Notify();
    return finish_status;
  }

  for (const auto& infer_request : infer_request_group) {
    infer_request->output_tokens = infer_request->input_tokens;
    infer_request->ResetPrefillingTokens();
    waiting_reqs_.push_back(infer_request);
  }
  return Status();
}

void BatchScheduler::BalanceWaitingReqs() {
  std::vector<std::pair<size_t, std::shared_ptr<InferRequest>>> waiting_reqs_with_index;
  {
    std::lock_guard<std::mutex> guard(waiting_reqs_mutex_);
    // inputs are waiting_reqs_ and batch_states_
    // output is dp_waiting_reqs_
    if (waiting_reqs_.empty()) {
      KLLM_LOG_SCHEDULER << "waiting_reqs_ is empty";
      return;
    }

    if (waiting_reqs_.size() == 1 && dp_waiting_reqs_.size() == 1) {
      dp_waiting_reqs_[0].insert(dp_waiting_reqs_[0].end(), waiting_reqs_.begin(), waiting_reqs_.end());
      waiting_reqs_.clear();
      waiting_reqs_cv_.notify_all();
      KLLM_LOG_SCHEDULER << "waiting_reqs_ size is 1";
      return;
    }

    for (auto& req : waiting_reqs_) {
      int64_t tokens_num = 0;
      if (req->forwarding_tokens.size() > 0) {
        tokens_num = req->forwarding_tokens.size() - req->kv_cached_token_num;
      } else {
        // forwarding_tokens is empty at first time
        tokens_num = req->input_tokens.size() - req->kv_cached_token_num;
      }
      tokens_num = tokens_num > 0 ? tokens_num : 1;
      waiting_reqs_with_index.emplace_back(
          std::make_pair<size_t, std::shared_ptr<InferRequest>>(static_cast<size_t>(tokens_num), std::move(req)));
    }
    waiting_reqs_.clear();
    // Staging space is free again; release any producer blocked in EnqueueWaitingBufferQueue.
    waiting_reqs_cv_.notify_all();
  }

  std::vector<float> workload(dp_num_, 0);
  for (size_t i = 0; i < dp_num_; ++i) {
    auto& dp_waiting_reqs = dp_waiting_reqs_[i];
    for (auto& req : dp_waiting_reqs) {
      int64_t tokens_num = 0;
      if (req->forwarding_tokens.size() > 0) {
        tokens_num = req->forwarding_tokens.size() - req->kv_cached_token_num;
      } else {
        // forwarding_tokens is empty at first time
        tokens_num = req->input_tokens.size() - req->kv_cached_token_num;
      }
      tokens_num = tokens_num > 0 ? tokens_num : 1;
      waiting_reqs_with_index.emplace_back(
          std::make_pair<size_t, std::shared_ptr<InferRequest>>(static_cast<size_t>(tokens_num), std::move(req)));
    }
    dp_waiting_reqs.clear();

    size_t running_size = 0;
    size_t waiting_size = 0;
    for (int j = 0; j < pp_batch_num_; j++) {
      auto& batch_state = batch_states_[i][j];
      std::lock_guard<std::mutex> guard(batch_state->queue_mutex);

      // Note(TJ): 最好可以使用每个req的tokens总和
      running_size += batch_state->schedule_output->running_reqs.size();
      running_size += batch_state->decoding_queue.size();
      waiting_size += batch_state->waiting_queue.size();
    }
    // 计算负载，根据优先级分配不同权重，数值越低，权重越低
    workload[i] = running_size * 0.7f + waiting_size;
  }

  balance_reqs_algo_->BalanceReqs(workload, waiting_reqs_with_index, dp_waiting_reqs_);
}

void BatchScheduler::BalancePPMultiBatchReqs(size_t multi_batch_id) {
  if (pp_multibatch_wl_balancers_.empty()) {
    return;
  }

  for (size_t i = 0; i < dp_num_; ++i) {
    pp_multibatch_wl_balancers_[i]->BalancePPMultiBatchReqs(multi_batch_id, dp_waiting_reqs_[i], batch_states_[i]);
  }
}

void BatchScheduler::ShiftChunkedReqsAcrossPPMultiBatch(size_t multi_batch_id) {
  if (pp_multibatch_wl_balancers_.empty()) {
    return;
  }

  for (size_t i = 0; i < dp_num_; ++i) {
    pp_multibatch_wl_balancers_[i]->ShiftChunkedReqs(multi_batch_id, batch_states_[i]);
  }
}

void BatchScheduler::ReportBatchState(std::shared_ptr<BatchState> batch_state, size_t dp_idx) {
  const std::string multi_batch_idx_str = std::to_string(batch_state->multi_batch_id_);
  const std::string dp_idx_str = std::to_string(dp_idx);
  const size_t batch_size = batch_state->schedule_output->running_reqs.size();
  REPORT_METRIC("batch_scheduler_batch_size_" + multi_batch_idx_str + "_" + dp_idx_str, batch_size);
  REPORT_METRIC("batch_scheduler_waiting_size_" + multi_batch_idx_str + "_" + dp_idx_str,
                batch_state->waiting_queue.size());
  REPORT_METRIC("batch_scheduler_batch_size", batch_size);
  REPORT_METRIC("batch_scheduler_waiting_size", batch_state->waiting_queue.size());

  if (batch_size > 0) {
    size_t step_token_num = 0;
    const auto current_time = ProfileTimer::GetCurrentTimeInUs();
    for (const auto& req : batch_state->schedule_output->running_reqs) {
      const ScheduleTaskWorkload& w = req->GetPlanningTask().workload;
      const size_t step_tokens =
          w.prefill_token_num > 0 ? w.prefill_token_num : w.generated_token_num + w.draft_token_num;
      step_token_num += step_tokens;

      if (req->kv_cached_token_num == 0) {
        REPORT_METRIC("batch_manager_schedule_us", current_time - req->timestamp_in_us);
      }
    }
    REPORT_METRIC("step_tokens_to_schedule", step_token_num);

    // Track last step_token per (dp, multi_batch) for PP balance reporting.
    last_step_tokens_[dp_idx][batch_state->multi_batch_id_] = step_token_num;
  }
}

bool BatchScheduler::TryToLaunchPlannedScheduleOutput(size_t multi_batch_id, ScheduleOutput& planned_schedule_output,
                                                      std::vector<std::shared_ptr<InferRequest>>& stopped_reqs) {
  stopped_reqs.clear();
  std::lock_guard<std::mutex> guard(schedule_mutex_);

  // Executor build-ahead 流水线深度由 FastPathController 在 Engine 启动期一次性判定:
  //   - 启用: max_depth=2, 允许同一 req 在上一 step 尚未 retire 时再 launch 一步 (占位 token + device 回填).
  //   - 禁用: max_depth=1，等价于 build-ahead 关闭时的 async 深度。
  const bool fast_path_active = FastPathController::GetInstance().IsEnabledAtStartup();
  const size_t max_depth = fast_path_active ? 2 : 1;

  if (!planned_schedule_output.IsLaunchable(max_depth)) {
    return false;
  }

  stopped_reqs.reserve(planned_schedule_output.running_reqs.size());
  // Remove finished requests in async mode
  for (auto it = planned_schedule_output.running_reqs.begin(); it != planned_schedule_output.running_reqs.end();) {
    auto req = *it;
    KLLM_LOG_SCHEDULER << req->ScheduleStateToStr();
    if (req->IsStopped()) {
      RemoveRequestFromQueue(planned_schedule_output.running_reqs, req);
      stopped_reqs.push_back(req);
      KLLM_LOG_SCHEDULER << "Drop finished req=" << req->req_id;
    } else {
      ++it;
    }
  }

  if (planned_schedule_output.running_reqs.empty()) {
    return true;
  }

  // per-req max_depth=2 只限制单请求 inflight 数, 不限制全局并发 launch 数.
  // Executor 侧 ModelInput 仅双 slot, 需额外限制在途 schedule 不超过 kExecutorPipelineMaxDepth,
  // 保证 launch(N+1) 前 N-1 已回流 (stop/EOS 决策可见), 且第三步不会抢占 ring slot.
  if (fast_path_active && executor_pipeline_depth_ >= kExecutorPipelineMaxDepth) {
    return false;
  }

  std::vector<ScheduleOutput*> outputs;
  for (size_t i = 0; i < dp_num_; ++i) {
    outputs.push_back(batch_states_[i][multi_batch_id]->schedule_output);
  }
  MergeScheduleInfoForWorkers(outputs, planned_schedule_output);

  // 末位 token 占位 vs 真实 token 由 LaunchScheduleOutput 按 req 判定 (HasInflightTask),
  // 不再要求 batch 成员/顺序与上一步一致.
  KLLM_CHECK(planned_schedule_output.IsLaunchable(max_depth));
  planned_schedule_output.LaunchScheduleOutput(max_depth);
  if (fast_path_active) {
    ++executor_pipeline_depth_;
  }

  // LaunchScheduleOutput() just moved chunk N from planning_task_
  // into inflight_task_, so planning slot is now empty and the chunked req
  // is eligible to be moved onto dst's waiting_queue.
  ShiftChunkedReqsAcrossPPMultiBatch(multi_batch_id);
  return true;
}

void BatchScheduler::UpdateWithGenerationResult(size_t multi_batch_id, const GenerationOutputGroup& generation_output) {
  std::lock_guard<std::mutex> guard(schedule_mutex_);
  // generation 回流后释放一个全局 pipeline 槽位, 与 TryToLaunch 中的 ++ 配对.
  if (FastPathController::GetInstance().IsEnabledAtStartup() && executor_pipeline_depth_ > 0) {
    --executor_pipeline_depth_;
  }
  uint64_t schedule_time_in_ms = GetCurrentTimeInMs();
  // Update running requests before workload balance
  for (size_t i = 0; i < dp_num_; i++) {
    schedule_strategies_[i]->SetBatchState(batch_states_[i][multi_batch_id]);
    batch_states_[i][multi_batch_id]->schedule_time_in_ms = schedule_time_in_ms;
    schedule_strategies_[i]->UpdateRunningRequests(generation_output.reqs[i]);
  }
}

std::shared_ptr<ScheduleOutputGroup> BatchScheduler::Schedule(size_t multi_batch_id) {
  PROFILE_EVENT_SCOPE(Schedule_, fmt::format("Schedule_{}", multi_batch_id));
  std::lock_guard<std::mutex> guard(schedule_mutex_);

  KLLM_LOG_SCHEDULER << "Try scheduler multi_batch_id=" << multi_batch_id
                     << ", waiting_reqs_size:" << waiting_reqs_.size();

  // Check if we need to add MockRequest to waiting_queue BEFORE scheduling
  // This ensures MockRequest is available when there are no other requests
  if (always_return_launchable_req_) {
    size_t total_running_size = 0;
    size_t total_waiting_size = 0;
    size_t total_mock_queue_size = 0;
    for (size_t i = 0; i < dp_num_; i++) {
      auto& batch_state = batch_states_[i][multi_batch_id];
      std::lock_guard<std::mutex> guard(batch_state->queue_mutex);
      total_running_size += batch_state->schedule_output->running_reqs.size();
      total_waiting_size += batch_state->waiting_queue.size();
    }

    // If no running requests and no waiting requests, add MockRequest to ensure we always have something to schedule
    if (total_running_size == 0 && total_waiting_size == 0) {
      for (size_t i = 0; i < dp_num_; i++) {
        auto& batch_state = batch_states_[i][multi_batch_id];
        std::lock_guard<std::mutex> guard(batch_state->queue_mutex);
        if (!batch_state->mock_queue.empty()) {
          auto it = batch_state->mock_queue.begin();
          auto mock_req = *it;
          batch_state->waiting_queue.push_back(mock_req);
          batch_state->mock_queue.erase(it);
        }
      }
    }
  }

  // ADP Balance: Apply balance strategy before normal balance
  if (batch_scheduler_config_.attention_dp_lb_config.enable_balance && dp_num_ > 1) {
    BalanceADPRequests(multi_batch_id);
  } else {
    // Normal balance process when ADP Balance is disabled
    BalanceWaitingReqs();
  }
  BalancePPMultiBatchReqs(multi_batch_id);

  std::vector<std::future<void>> futures;
  for (size_t i = 0; i < dp_num_; i++) {
    futures.push_back(threadpool_->Submit([this, i, multi_batch_id] {
      schedule_strategies_[i]->SetBatchState(batch_states_[i][multi_batch_id]);
      schedule_strategies_[i]->Schedule(dp_waiting_reqs_[i]);
    }));
  }

  for (auto& future : futures) {
    future.wait();
  }

  size_t total_running_size = 0;
  size_t total_waiting_size_in_batch_states = 0;
  size_t total_dp_waiting_queue_size = 0;
  size_t total_mock_queue_size = 0;
  for (size_t i = 0; i < dp_num_; i++) {
    auto& batch_state = batch_states_[i][multi_batch_id];
    ReportBatchState(batch_state, i);
    schedule_output_group_->outputs[i] = batch_state->schedule_output;
    total_running_size += batch_state->schedule_output->running_reqs.size();
    total_waiting_size_in_batch_states += batch_state->waiting_queue.size();
    total_dp_waiting_queue_size += dp_waiting_reqs_[i].size();
    total_mock_queue_size += batch_state->mock_queue.size();
  }

  schedule_output_group_->schedule_id++;

  // 为本轮调度中的 prefill 请求注册 pending put 计数
  if (global_cache_put_tracker_ && total_running_size > 0) {
    for (size_t i = 0; i < dp_num_; i++) {
      global_cache_put_tracker_->OnBatchDispatched(batch_states_[i][multi_batch_id]->schedule_output->running_reqs);
    }
  }

  KLLM_LOG_SCHEDULER << "Finish schedule. multi_batch_id=" << multi_batch_id
                     << ", schedule_id=" << schedule_output_group_->schedule_id
                     << ", running_req.size(): " << total_running_size
                     << ", total_waiting_size_in_batch_states=" << total_waiting_size_in_batch_states
                     << ", total_dp_waiting_queue_size=" << total_dp_waiting_queue_size
                     << ", total_mock_queue_size=" << total_mock_queue_size;

  // Balance: for each multi_batch, compare its step_token total against the
  // previous multi_batch's value.
  if (pp_batch_num_ > 1) {
    size_t cur_step_token_num = 0;
    for (size_t dp = 0; dp < dp_num_; ++dp) {
      cur_step_token_num += last_step_tokens_[dp][multi_batch_id];
    }
    const size_t prev_mb = (multi_batch_id + pp_batch_num_ - 1) % pp_batch_num_;
    size_t prev_step_token_num = 0;
    for (size_t dp = 0; dp < dp_num_; ++dp) {
      prev_step_token_num += last_step_tokens_[dp][prev_mb];
    }
    // Skip when either side is zero (first round before prev_mb seen, or idle
    // batch) to avoid div-by-zero and ratio pollution.
    if (cur_step_token_num > 0 && prev_step_token_num > 0) {
      const size_t min_v = std::min(cur_step_token_num, prev_step_token_num);
      const size_t max_v = std::max(cur_step_token_num, prev_step_token_num);
      const double ratio = static_cast<double>(min_v) / static_cast<double>(max_v);
      accumulated_balance_ratio_ += ratio;
      accumulated_balance_round_++;
    }
  }

  ReportTotalState();
  return schedule_output_group_;
}

void BatchScheduler::ReportTotalState() {
  // Report every 10 seconds.
  constexpr size_t kReportIntervalMs = 10000;
  static IntervalLogger interval_logger(kReportIntervalMs);
  if (!interval_logger.ShouldLog()) {
    return;
  }

  size_t total_running_size = 0;
  size_t total_decoding_size = 0;
  size_t total_waiting_size = 0;
  size_t total_transfer_size = 0;
  {
    std::lock_guard<std::mutex> guard(waiting_reqs_mutex_);
    total_waiting_size = waiting_reqs_.size();
    for (size_t dp_rank = 0; dp_rank < dp_num_; ++dp_rank) {
      total_waiting_size += dp_waiting_reqs_[dp_rank].size();
      auto& batch_states = batch_states_[dp_rank];
      for (size_t multi_batch_id = 0; multi_batch_id < pp_batch_num_; ++multi_batch_id) {
        auto& batch_state = batch_states[multi_batch_id];
        std::lock_guard<std::mutex> guard(batch_state->queue_mutex);
        total_running_size += batch_state->schedule_output->running_reqs.size();
        total_decoding_size += batch_state->decoding_queue.size();
        total_waiting_size += batch_state->waiting_queue.size();
        total_transfer_size += batch_state->transfer_queue.size();
      }
    }
  }

  size_t total_used_blocks_num = 0;
  size_t total_free_blocks_num = 0;

  for (size_t dp_rank = 0; dp_rank < dp_num_; ++dp_rank) {
    auto& cache_manager = schedule_strategies_[dp_rank]->GetCacheManager();
    total_used_blocks_num += cache_manager->GetUsedBlockNumber();
    total_free_blocks_num += cache_manager->GetUsableBlockNumber();
  }
  size_t total_block_num = total_used_blocks_num + total_free_blocks_num;
  KLLM_LOG_INFO << "running_req_num=" << total_running_size << ", decoding_req_num=" << total_decoding_size
                << ", waiting_req_num=" << total_waiting_size << ", transfer_req_num=" << total_transfer_size
                << ", free_block_num=" << total_free_blocks_num
                << ", block_utils=" << (total_used_blocks_num * 100 / total_block_num) << "% (" << total_used_blocks_num
                << "/" << total_block_num << ")";

  // Balance reporting: per-multi_batch last step_token, instantaneous
  // imbalance ratio (min/max), and accumulated average ratio over rounds.
  if (pp_batch_num_ > 1) {
    const double avg_ratio = (accumulated_balance_round_ > 0)
                                 ? accumulated_balance_ratio_ / static_cast<double>(accumulated_balance_round_)
                                 : 0.0;
    KLLM_LOG_INFO << "PP balance: accumulated avg balance ratio=" << avg_ratio << " over " << accumulated_balance_round_
                  << " rounds";
    REPORT_METRIC("pp_step_tokens_balance_ratio_avg", avg_ratio);

    // Reset metric
    accumulated_balance_ratio_ = 0.0;
    accumulated_balance_round_ = 0;
  }
}

Status BatchScheduler::CreateMockReq(const RuntimeConfig& runtime_config,
                                     std::vector<std::shared_ptr<InferRequest>>& infer_request_group) {
  // To avoid Mock requests being categorized as SingleTokenForward requests, we calculate the Mock request total
  // length as: Mock total length = MTP token count + SingleToken length + 1 (additional token)
  auto spec_profile = SpeculativeRegistry::Get().Aggregate(batch_scheduler_config_);
  const size_t mock_req_length = spec_profile.workload_multiplier + 1 + 1;
  auto mock_req_input = std::make_shared<KsanaPythonInput>();
  alias_python_input_ = mock_req_input;
  std::vector<int> input_tokens(mock_req_length, 0);
  for (int i = 0; i < mock_req_length; ++i) {
    input_tokens[i] = (i + 1) % 100;  // Fill with some dummy tokens.
  }

  mock_req_input->input_tokens = input_tokens;
  // Only do one token prefill.
  mock_req_input->sampling_config.max_new_tokens = 1;
  mock_req_input->sampling_config.ignore_eos = true;
  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  auto mock_req = std::make_shared<Request>(mock_req_input, req_ctx);
  alias_mock_request_ = mock_req;

  mock_req->waiter = std::make_shared<Waiter>(1);
  KLLM_LOG_DEBUG << "mock_req req_id " << mock_req->req_id << ", input_tokens_num " << mock_req->input_tokens.size()
                 << ", output_tokens addr " << mock_req->output_tokens.data();

  // mock_req->output_group.size() == 1.
  for (size_t i = 0; i < mock_req->output_group.size(); i++) {
    std::shared_ptr<InferRequest> infer_req = std::make_shared<InferRequest>(mock_req, i);
    infer_request_group.push_back(infer_req);
    RuntimeConfig runtime_config;
    Singleton<Environment>::GetInstance()->GetRuntimeConfig(runtime_config);
    infer_req->kv_cache_blocks.clear();
    CacheManagerConfig cache_manager_config;
    Singleton<Environment>::GetInstance()->GetCacheManagerConfig(cache_manager_config);
    infer_req->block_token_num = cache_manager_config.block_token_num;
    infer_req->req_id = mock_req->req_id;
    infer_req->is_mock_req = true;
    infer_req->ResetPrefillingTokens();
  }

  for (auto& infer_req : infer_request_group) {
    infer_req->SetReqGroup(infer_request_group);
    KLLM_LOG_DEBUG << "InferRequest output_tokens_num " << infer_req->output_tokens.size() << ", Addr " << infer_req
                   << ", output_tokens addr " << infer_req->output_tokens.data();
  }

  return Status();
}

void BatchScheduler::CalculatePrefillWorkload(std::vector<float>& workload) {
  workload.assign(dp_num_, 0);
  for (size_t i = 0; i < dp_num_; ++i) {
    float prefill_workload = 0;
    // Calculate PrefillWorkload: sum of (forwarding_tokens - kv_cached_token_num) for running requests
    for (int j = 0; j < pp_batch_num_; j++) {
      auto& batch_state = batch_states_[i][j];
      std::lock_guard<std::mutex> guard(batch_state->queue_mutex);
      for (const auto& req : batch_state->schedule_output->running_reqs) {
        int64_t tokens_num = 0;
        if (req->forwarding_tokens.size() > 0) {
          tokens_num = req->forwarding_tokens.size() - req->kv_cached_token_num;
        } else {
          tokens_num = req->input_tokens.size() - req->kv_cached_token_num;
        }
        prefill_workload += static_cast<float>(tokens_num > 0 ? tokens_num : 1);
      }
    }
    workload[i] = prefill_workload;
  }
}

void BatchScheduler::DistributeRequestsByToken(
    const std::vector<std::pair<size_t, std::shared_ptr<InferRequest>>>& requests_with_tokens) {
  // Calculate current workload for each DP rank
  std::vector<float> workload;
  CalculatePrefillWorkload(workload);

  // Sort requests by token count in descending order (largest first)
  // This is a key optimization: assign heavy requests first for better global balance
  auto sorted_requests = requests_with_tokens;
  std::sort(sorted_requests.begin(), sorted_requests.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });

  // Use minimum makespan algorithm (minimize maximum workload)
  // This is better than simple greedy as it considers global optimization
  for (const auto& [tokens_num, req] : sorted_requests) {
    // Find the rank that would result in minimum maximum workload after assignment
    size_t best_rank = 0;
    float min_max_workload = workload[0] + static_cast<float>(tokens_num);

    for (size_t i = 1; i < dp_num_; ++i) {
      float potential_workload = workload[i] + static_cast<float>(tokens_num);
      if (potential_workload < min_max_workload) {
        min_max_workload = potential_workload;
        best_rank = i;
      }
    }

    // Assign request to the best rank
    dp_waiting_reqs_[best_rank].push_back(req);

    // Update workload for the selected rank
    workload[best_rank] += static_cast<float>(tokens_num);
  }

  // Calculate and log final load balance metrics for debugging
  float max_workload = *std::max_element(workload.begin(), workload.end());
  float min_workload = *std::min_element(workload.begin(), workload.end());
  float avg_workload = std::accumulate(workload.begin(), workload.end(), 0.0f) / dp_num_;
  float load_imbalance = max_workload > 0 ? (max_workload - min_workload) / max_workload : 0.0f;

  KLLM_LOG_DEBUG << "ADP DistributeRequestsByToken: assigned " << sorted_requests.size()
                 << " requests, load_imbalance=" << load_imbalance << ", max_workload=" << max_workload
                 << ", min_workload=" << min_workload << ", avg_workload=" << avg_workload;
}

bool BatchScheduler::IsADPBalanceTimeout() {
  // Check time-based timeout first
  if (adp_balance_waiting_iters_ > 0) {
    auto current_time = std::chrono::steady_clock::now();
    auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(current_time - adp_balance_start_time_).count();
    if (elapsed_ms >= static_cast<int64_t>(GetCurrentPrefillAccumulationMaxTimeMs())) {
      return true;
    }
  }

  // Check iteration-based timeout
  return adp_balance_waiting_iters_ >= GetCurrentPrefillAccumulationMaxSteps();
}

void BatchScheduler::UpdateQPS(size_t request_count) {
  request_count_window_ += request_count;  // 累加实际的请求数量

  auto now = std::chrono::steady_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(now - last_qps_update_time_).count();

  // 每隔QPS_UPDATE_INTERVAL_SECONDS秒更新一次QPS
  if (elapsed >= static_cast<double>(QPS_UPDATE_INTERVAL_SECONDS)) {
    current_qps_ = request_count_window_ / elapsed;

    KLLM_LOG_DEBUG << "Updated QPS: " << current_qps_ << " req/s (based on " << request_count_window_ << " requests in "
                   << elapsed << "s)";

    last_qps_update_time_ = now;
    request_count_window_ = 0;
  }
}

size_t BatchScheduler::GetCurrentPrefillAccumulationMaxSteps() {
  double min_qps_for_waiting = batch_scheduler_config_.attention_dp_lb_config.min_qps_for_waiting;
  // QPS大于阈值时不做累积，否则使用配置值（阈值<0表示所有QPS都使用）
  return (min_qps_for_waiting >= 0 && current_qps_ > min_qps_for_waiting)
             ? 0
             : batch_scheduler_config_.attention_dp_lb_config.max_waiting_steps;
}

size_t BatchScheduler::GetCurrentPrefillAccumulationMaxTimeMs() {
  double min_qps_for_waiting = batch_scheduler_config_.attention_dp_lb_config.min_qps_for_waiting;
  // QPS大于阈值时不做累积，否则使用配置值（阈值<0表示所有QPS都使用）
  return (min_qps_for_waiting >= 0 && current_qps_ > min_qps_for_waiting)
             ? 0
             : batch_scheduler_config_.attention_dp_lb_config.max_waiting_time_in_ms;
}

void BatchScheduler::BalanceADPRequests(size_t multi_batch_id) {
  // Step 1: Collect all waiting requests (similar to BalanceWaitingReqs)
  std::vector<std::pair<size_t, std::shared_ptr<InferRequest>>> waiting_reqs_with_index;
  {
    std::lock_guard<std::mutex> guard(waiting_reqs_mutex_);

    // Collect from waiting_reqs_
    for (auto& req : waiting_reqs_) {
      int64_t tokens_num = 0;
      if (req->forwarding_tokens.size() > 0) {
        tokens_num = req->forwarding_tokens.size() - req->kv_cached_token_num;
      } else {
        tokens_num = req->input_tokens.size() - req->kv_cached_token_num;
      }
      tokens_num = tokens_num > 0 ? tokens_num : 1;
      waiting_reqs_with_index.emplace_back(
          std::make_pair<size_t, std::shared_ptr<InferRequest>>(static_cast<size_t>(tokens_num), std::move(req)));
    }
    waiting_reqs_.clear();
    // Staging space is free again; release any producer blocked in EnqueueWaitingBufferQueue.
    waiting_reqs_cv_.notify_all();
  }

  // Collect from dp_waiting_reqs_ and batch_states waiting queues
  for (size_t i = 0; i < dp_num_; ++i) {
    // From dp_waiting_reqs_
    for (auto& req : dp_waiting_reqs_[i]) {
      int64_t tokens_num = 0;
      if (req->forwarding_tokens.size() > 0) {
        tokens_num = req->forwarding_tokens.size() - req->kv_cached_token_num;
      } else {
        tokens_num = req->input_tokens.size() - req->kv_cached_token_num;
      }
      tokens_num = tokens_num > 0 ? tokens_num : 1;
      waiting_reqs_with_index.emplace_back(
          std::make_pair<size_t, std::shared_ptr<InferRequest>>(static_cast<size_t>(tokens_num), std::move(req)));
    }
    dp_waiting_reqs_[i].clear();

    // From batch_states waiting queues for the specific multi_batch_id
    auto& batch_state = batch_states_[i][multi_batch_id];
    std::lock_guard<std::mutex> guard(batch_state->queue_mutex);
    for (auto& req : batch_state->waiting_queue) {
      int64_t tokens_num = req->input_tokens.size() - req->kv_cached_token_num;
      tokens_num = tokens_num > 0 ? tokens_num : 1;
      waiting_reqs_with_index.emplace_back(
          std::make_pair<size_t, std::shared_ptr<InferRequest>>(static_cast<size_t>(tokens_num), std::move(req)));
    }
    batch_state->waiting_queue.clear();
  }

  // Step 2: All waiting requests are prefill requests
  // Check balance conditions:
  // 1. waiting_reqs_with_index.size() >= dp_num_ (enough requests for all ranks)
  // 2. Not all ranks have generation requests
  bool all_ranks_have_ctx_requests = waiting_reqs_with_index.size() >= dp_num_;
  bool all_ranks_have_gen_requests = true;

  // Check if all ranks have generation requests
  for (size_t i = 0; i < dp_num_; i++) {
    auto& batch_state = batch_states_[i][multi_batch_id];
    if (batch_state->schedule_output->running_reqs.size() == 0) {
      all_ranks_have_gen_requests = false;
      break;
    }
  }

  // Step 3: Check timeout condition
  bool timeout_reached = IsADPBalanceTimeout();

  // Step 4: Decide whether to proceed with prefill
  bool should_proceed_prefill = all_ranks_have_ctx_requests || !all_ranks_have_gen_requests || timeout_reached;
  // Step 5: Apply decision and distribute requests
  if (should_proceed_prefill) {
    // Proceed with prefill: distribute requests based on current workload
    DistributeRequestsByToken(waiting_reqs_with_index);

    adp_balance_waiting_iters_ = 0;
    adp_balance_context_batches_ = 0;

    if (timeout_reached) {
      KLLM_LOG_DEBUG << "ADP Balance: Proceeding with prefill due to timeout - waiting_iters="
                     << adp_balance_waiting_iters_ << ", waiting_reqs_count=" << waiting_reqs_with_index.size();
    } else {
      KLLM_LOG_DEBUG << "ADP Balance: Proceeding with prefill - waiting_reqs_count=" << waiting_reqs_with_index.size()
                     << ", dp_num=" << dp_num_ << ", all_ranks_have_gen=" << all_ranks_have_gen_requests;
    }
  } else {
    // Don't proceed with prefill: save all waiting requests back for next iteration
    {
      std::lock_guard<std::mutex> guard(waiting_reqs_mutex_);
      for (auto& [tokens_num, req] : waiting_reqs_with_index) {
        waiting_reqs_.push_back(req);
      }
    }

    // Initialize start time for first wait iteration
    if (adp_balance_waiting_iters_ == 0) {
      adp_balance_start_time_ = std::chrono::steady_clock::now();
    }
    adp_balance_waiting_iters_++;
  }
}

}  // namespace ksana_llm
