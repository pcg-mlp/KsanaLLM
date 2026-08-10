/* Copyright 2026 Tencent Inc.  All rights reserved.

 White-box unit tests for EventDrivenScheduleProcessor.

 The existing event_driven_scheduler_test.cpp drives the processor end-to-end
 through ParallelTester (real backend + real hook chain), which only exercises
 the happy path. This file targets the remaining branches — pd_v2 decode
 restart, async hook completion, deferred backend events, hook fail-and-end,
 the cache-put-tracker defer path, and the load snapshot — by driving the
 private handlers directly on a single thread via a friend peer, plus a couple
 of thread-based tests for AsyncSchedulerLoop / Start / Stop.
 ==============================================================================*/

#include "batch_scheduler/event_driven_scheduler/event_driven_schedule_processor.h"

#include <atomic>
#include <memory>
#include <thread>
#include <vector>

#include "batch_scheduler/event_driven_scheduler/schedule_backend_interface.h"
#include "batch_scheduler/event_driven_scheduler/schedule_hook_manager_interface.h"
#include "batch_scheduler/event_driven_scheduler/workload_balance/pp_batch_distributor.h"
#include "batch_scheduler/global_cache_put_tracker.h"
#include "cache_manager/cache_manager_interface.h"
#include "profiler/timer.h"
#include "runtime/infer_request.h"
#include "utils/request.h"

#include "loguru.hpp"
#include "test.h"

namespace ksana_llm {
namespace {

// Bump loguru verbosity so SCHEDULER/INFO lines along the tested paths execute.
class MaxLogVerbosityEnv : public ::testing::Environment {
 public:
  void SetUp() override { loguru::g_stderr_verbosity = loguru::Verbosity_MAX; }
};
static auto* unused_log_env = ::testing::AddGlobalTestEnvironment(new MaxLogVerbosityEnv());

// ----- No-op cache manager (DestroyFinishedRequest is the only call we hit) -----
class NoopCacheManager : public CacheManagerInterface {
 public:
  void InitializeCachedBlocks() override {}
  std::shared_ptr<BlockAllocatorGroupInterface> GetBlockAllocatorGroup() const override { return nullptr; }
  size_t GetUsableBlockNumber() override { return 0; }
  size_t GetHostFreeBlockNumber() override { return 0; }
  size_t GetUsedBlockNumber() override { return 0; }
  size_t GetRequestStepBlockNumber(int64_t, size_t, size_t) override { return 0; }
  size_t GetRequestStepBlockNumberForOneNextToken(int64_t) override { return 0; }
  size_t GetRequestUsableBlockNumber(int64_t) override { return 0; }
  Status GetRequestPrefixBlockNumber(int64_t, const std::vector<int>&, size_t, size_t&, size_t&, size_t&) override {
    return Status();
  }
  Status GetRequestPrefixBlockNumberBeforeRemoteCacheLoading(int64_t, const std::vector<int>&, size_t, size_t&, size_t&,
                                                             size_t&, size_t&, size_t&, size_t&) override {
    return Status();
  }
  Status AllocateBlocksForRemoteCache(int64_t, std::vector<int>&, size_t, size_t, std::vector<int>&,
                                      std::vector<int>&) override {
    return Status();
  }
  Status ConfirmRemoteCacheLoadingResult(int64_t, const std::vector<int>&, size_t,
                                         const std::vector<size_t>&) override {
    return Status();
  }
  Status AllocateRequestBlocks(int64_t, size_t, std::vector<int>&, std::vector<int>&, std::vector<size_t>&,
                               int&) override {
    return Status();
  }
  void UpdateFlexibleCache(int64_t, const std::vector<int>&, int, std::vector<FlexibleCachedCopyTask>&,
                           size_t&) override {}
  Status UpdateRequestTokens(int64_t, const std::vector<int>&, size_t, std::vector<int>&, bool&, std::vector<int>&,
                             std::vector<size_t>&) override {
    return Status();
  }
  void DestroyFinishedRequest(int64_t req_id) override { destroyed.push_back(req_id); }
  bool IsPrefixCachingEnabled() override { return false; }

  std::vector<int64_t> destroyed;
};

// ----- Controllable hook manager -----
class FakeHookManager : public ScheduleHookManagerInterface {
 public:
  void AddHook(ScheduleHookPtr) override {}

  ReqIdToNextStep Process(const std::vector<InferRequestPtr>& requests, ScheduleContext&) override {
    process_called = true;
    if (set_decode_stage) {
      // pd_v2: hook stages KV from peer Prefill and flips the request to decode.
      for (const auto& r : requests) {
        r->infer_stage = InferStage::kDecode;
      }
    }
    if (use_process_override) {
      return process_result;
    }
    ReqIdToNextStep out;
    for (const auto& r : requests) {
      out[r->req_id] = NextStep{default_action, 0};
    }
    return out;
  }

  ReqIdToNextStep UpdateAsyncResult(const std::vector<AsyncTaskResult>&, ScheduleContext&) override {
    update_async_called = true;
    return async_result;
  }

  void SetAsyncTaskFinishedCallback(AsyncTaskFinishedCallback cb) override { cb_ = std::move(cb); }

  void OnEvent(const HookEvent& event) override {
    switch (event.type) {
      case HookEventType::kBlocksFreed:
        ++notify_blocks_freed_cnt;
        break;
      case HookEventType::kRequestExited:
        exited.push_back(event.req_id);
        break;
      default:
        break;
    }
  }

  // knobs
  NextStepAction default_action = NextStepAction::kContinue;
  bool set_decode_stage = false;
  bool use_process_override = false;
  ReqIdToNextStep process_result;
  ReqIdToNextStep async_result;

  // observations
  bool process_called = false;
  bool update_async_called = false;
  int notify_blocks_freed_cnt = 0;
  std::vector<int64_t> exited;
  AsyncTaskFinishedCallback cb_;
};

// ----- Controllable backend -----
class FakeBackend : public ScheduleBackendInterface {
 public:
  explicit FakeBackend(const BatchSchedulerConfig& cfg) : ScheduleBackendInterface(cfg) {}

  void PreSchedule(const ScheduleContext&, const std::vector<std::shared_ptr<PlanningBatch>>&,
                   std::vector<ScheduleBackendEvent>& events) override {
    for (auto& e : preschedule_events) events.push_back(e);
    preschedule_events.clear();
  }

  void Schedule(const ScheduleContext&, std::vector<std::shared_ptr<PlanningBatch>>& planning_batches,
                std::vector<ScheduleBackendEvent>& events) override {
    ++schedule_calls;
    // Optionally make batch 0 launchable by adding the staged request.
    if (launch_req != nullptr && !launched_once) {
      planning_batches[0]->AddRunningReq(0, launch_req);
      launched_once = true;
    }
    for (auto& e : schedule_events) events.push_back(e);
    schedule_events.clear();
  }

  void UpdateWithGenerationResult(size_t, const GenerationOutputGroup&, const ScheduleContext&,
                                  std::vector<ScheduleBackendEvent>& events) override {
    for (auto& e : update_events) events.push_back(e);
    update_events.clear();
  }

  void Stop() override { ++stop_calls; }

  std::vector<ScheduleBackendEvent> preschedule_events;
  std::vector<ScheduleBackendEvent> schedule_events;
  std::vector<ScheduleBackendEvent> update_events;
  std::shared_ptr<InferRequest> launch_req = nullptr;
  bool launched_once = false;
  int schedule_calls = 0;
  int stop_calls = 0;
};

// ----- Minimal workload balancer (assigns dp 0; no-op otherwise) -----
class FakeWorkloadBalancer : public WorkloadBalancerInterface {
 public:
  void AddInferRequests(std::vector<InferRequestPtr>& requests, const ScheduleContext&) override {
    add_called = true;
    for (auto& r : requests) r->attn_dp_group_id = 0;
  }
  std::vector<WorkloadQuota> GetDpSchedulingOrderAndQuota(const PlanningBatch&, const ScheduleContext&,
                                                          InferStage) override {
    return {};
  }
  void RecordInflightBatch(const PlanningBatch&, const ScheduleContext&) override { record_called = true; }
  bool IsRequestAcceptable(const InferRequestPtr&, const PlanningBatch&, const WorkloadQuota&, size_t,
                           size_t) const override {
    return true;
  }
  bool add_called = false;
  bool record_called = false;
};

// Keep Request objects alive: InferRequest holds references into Request.
std::vector<std::shared_ptr<Request>> g_alive_requests;

std::shared_ptr<InferRequest> MakeReq(int64_t req_id, int token_num = 10) {
  auto input = std::make_shared<KsanaPythonInput>();
  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  auto request = std::make_shared<Request>(input, req_ctx);
  g_alive_requests.push_back(request);
  auto infer_req = std::make_shared<InferRequest>(request, 0);
  infer_req->req_id = req_id;
  infer_req->infer_stage = InferStage::kContext;
  infer_req->block_token_num = 16;
  infer_req->attn_dp_group_id = 0;
  for (int i = 0; i < token_num; ++i) {
    infer_req->output_tokens.push_back(i + 100);
  }
  infer_req->ResetPrefillingTokens();
  return infer_req;
}

// Force an inflight task so HasInflightTask() == true.
void MakeInflight(const std::shared_ptr<InferRequest>& req) {
  req->ResetPrefillingTokens();
  req->SetPlanningTask();
  req->LaunchPlanningTask();
}

}  // namespace

// Friend peer: exposes private handlers / members for deterministic single-thread testing.
class EventDrivenScheduleProcessorWhiteBox : public testing::Test {
 protected:
  void SetUp() override {
    g_alive_requests.clear();
    cfg_.max_pp_batch_num = 1;
    cache_ = std::make_shared<NoopCacheManager>();
    hook_ = std::make_shared<FakeHookManager>();
    backend_ = std::make_shared<FakeBackend>(cfg_);

    proc_ = std::make_unique<EventDrivenScheduleProcessor>(/*dp_num=*/1, /*max_pp_batch_num=*/1);
    proc_->SetScheduleHookManager(hook_);
    proc_->SetCacheManagers({cache_});
    proc_->SetScheduleBackend(backend_);
    proc_->Initialize(nullptr);
  }

  void TearDown() override { proc_.reset(); }

  // Register a request into the global index + batch index (and optionally a queue).
  void RegisterReq(const std::shared_ptr<InferRequest>& req, size_t dp = 0, size_t batch = 0) {
    auto* st = proc_->GetScheduleState();
    st->req_id_to_req_ptr[req->req_id] = req;
    st->req_to_batch_idx[req->req_id] = batch;
  }

  DpScheduleState& DpState(size_t batch = 0, size_t dp = 0) {
    return proc_->GetScheduleState()->schedule_states[batch][dp];
  }

  // ---- private handler forwarders ----
  size_t Lookup(int64_t id) { return proc_->LookupReqBatchIdx(id, "test"); }
  void HandleStop(const std::shared_ptr<InferRequest>& r, Status s, size_t dp) { proc_->HandleStopRequest(r, s, dp); }
  void HandleFail(const std::shared_ptr<InferRequest>& r, size_t dp, const std::string& why) {
    proc_->HandleHookFailAndEnd(r, dp, why);
  }
  void HandleRecompute(const std::shared_ptr<InferRequest>& r, Status s, size_t dp) {
    proc_->HandleRecomputeRequest(r, s, dp);
  }
  void HandlePdRestart(const std::shared_ptr<InferRequest>& r, size_t dp) { proc_->HandlePdV2DecodeRestart(r, dp); }
  bool Apply(const std::vector<ScheduleBackendEvent>& evs) { return proc_->ApplyBackendEventsToState(evs); }
  bool ProcessDelayed() { return proc_->ProcessDelayedBackendEvents(); }
  void OnAdd(const std::shared_ptr<EventAddRequestInfo>& i) { proc_->OnEventAddRequest(i); }
  void OnUpdate(const std::shared_ptr<EventUpdateAfterGenerationInfo>& i) { proc_->OnEventUpdateAfterGeneration(i); }
  void OnAsync(const std::shared_ptr<EventAsyncTaskFinishedInfo>& i) { proc_->OnEventAsyncTaskFinished(i); }
  void OnGetResult() { proc_->OnEventGetScheduleResult(); }
  void Dispatch(SchedEvent& e) { proc_->DispatchEvent(e); }
  void UpdateAsync(const EventAsyncTaskFinishedInfo& i) { proc_->UpdateAfterAsyncTaskFinished(i); }
  void Enqueue(const EventAsyncTaskFinishedInfo& i) { proc_->EnqueueAsyncTaskFinished(i); }
  bool TryBuild() { return proc_->TryBuildAndLaunchScheduleResult(proc_->schedule_context_); }
  void RunSched() { proc_->RunScheduleForBatches(); }
  void RecordLatency() { proc_->RecordStepLatency(); }

  // ---- private member accessors ----
  std::vector<ScheduleBackendEvent>& Delayed() { return proc_->delayed_backend_events_; }
  size_t& PipelineDepth() { return proc_->pipeline_depth_; }
  bool& Pending() { return proc_->pending_get_schedule_result_; }
  std::vector<std::shared_ptr<PlanningBatch>>& Planning() { return proc_->planning_batches_; }
  int64_t& LastStepLaunchMs() { return proc_->last_step_launch_ms_; }
  uint32_t& LastStepTokenNum() { return proc_->last_step_token_num_; }
  uint32_t StepTokenEwma() { return proc_->step_token_ewma_.load(); }

  BatchSchedulerConfig cfg_;
  std::shared_ptr<NoopCacheManager> cache_;
  std::shared_ptr<FakeHookManager> hook_;
  std::shared_ptr<FakeBackend> backend_;
  std::unique_ptr<EventDrivenScheduleProcessor> proc_;
};

// NOTE: kept first on purpose. TryBuildAndLaunchScheduleResult's "gated"
// diagnostic block is throttled to ~1/s via a function-static timestamp; only
// the FIRST gated call in the whole process actually enters it. Running this
// test first (with a populated, inflight batch) ensures that single entry
// exercises the inner running_reqs scan instead of an empty-batch gated call.
TEST_F(EventDrivenScheduleProcessorWhiteBox, TryBuildNotLaunchableGated) {
  auto req = MakeReq(1);
  RegisterReq(req);
  Planning()[0]->AddRunningReq(0, req);  // sets planning task
  req->LaunchPlanningTask();             // -> inflight, no capacity -> not launchable

  EXPECT_FALSE(TryBuild());
}

// ========================= LookupReqBatchIdx =========================

TEST_F(EventDrivenScheduleProcessorWhiteBox, LookupReqBatchIdxFoundAndThrow) {
  auto req = MakeReq(1);
  RegisterReq(req, 0, 0);
  EXPECT_EQ(Lookup(1), 0u);
  EXPECT_THROW(Lookup(999), std::exception);
}

// ========================= HandleStopRequest =========================

TEST_F(EventDrivenScheduleProcessorWhiteBox, HandleStopRequestNormal) {
  auto req = MakeReq(1);
  RegisterReq(req);
  DpState().in_plan_schedule_ops.stop_req_ids.insert(1);

  HandleStop(req, Status(RET_SUCCESS), 0);

  EXPECT_TRUE(req->IsStopped());
  EXPECT_TRUE(req->finished);
  EXPECT_EQ(DpState().schedule_ops_for_nodes.stopped_req_ids.size(), 1u);
  EXPECT_EQ(cache_->destroyed.size(), 1u);  // DestroyFinishedRequest hit
  EXPECT_EQ(hook_->exited.size(), 1u);      // OnRequestExited hit
  EXPECT_EQ(proc_->GetScheduleState()->req_id_to_req_ptr.count(1), 0u);
  EXPECT_EQ(proc_->GetScheduleState()->req_to_batch_idx.count(1), 0u);
}

TEST_F(EventDrivenScheduleProcessorWhiteBox, HandleStopRequestAlreadyStoppedSkips) {
  auto req = MakeReq(1);
  RegisterReq(req);
  req->Stop();
  HandleStop(req, Status(RET_SUCCESS), 0);
  EXPECT_EQ(cache_->destroyed.size(), 0u);  // early return, nothing destroyed
}

TEST_F(EventDrivenScheduleProcessorWhiteBox, HandleStopRequestDeferredByPutTracker) {
  auto tracker = std::make_shared<GlobalCachePutTracker>(/*device_count=*/1, cache_);
  tracker->SetDpCacheManagers({cache_});
  proc_->SetGlobalCachePutTracker(tracker);

  auto req = MakeReq(1);
  RegisterReq(req);
  tracker->OnBatchDispatched({req});  // register a pending put -> defer destruction

  HandleStop(req, Status(RET_SUCCESS), 0);

  EXPECT_TRUE(req->finished);
  EXPECT_EQ(cache_->destroyed.size(), 0u);  // destruction deferred, not called inline
  EXPECT_TRUE(tracker->HasDeferredRequests());
}

// Access-log / pd_v2 phase metrics in HandleStopRequest. REPORT_METRIC is a no-op
// without a process profiler; these tests exercise every guarded branch so the
// new lines stay covered under gcov.
TEST_F(EventDrivenScheduleProcessorWhiteBox, HandleStopRequestReportsAccessLogAndPdV2Metrics) {
  auto req = MakeReq(1, /*token_num=*/4);  // 4 output tokens, empty input -> output_token_num=4
  RegisterReq(req);

  // Monotonic timeline (arrive is stamped in InferRequest ctor).
  const uint64_t arrive = req->timestamp_in_us;
  ASSERT_GT(arrive, 0u);
  req->scheduled_time_us.store(arrive + 300);
  req->first_token_time_us.store(arrive + 200);  // first token before decode schedule -> decode_queue
  // finish_time_us is StampOnce'd inside HandleStopRequest.

  // Remote Prefill path: dispatched > 0 enables the pd_v2 block.
  req->pd_v2.prefill_dispatched_time_us.store(arrive + 50);
  req->pd_v2.blocks_allocated_time_us.store(arrive + 40);   // admit_us
  req->pd_v2.prefill_complete_time_us.store(arrive + 180);  // remote_rtt
  req->pd_v2.pf_queue_us.store(20);
  req->pd_v2.pf_compute_us.store(30);
  req->pd_v2.pf_send_us.store(10);  // pf_total=60 < remote_rtt=130 -> kv_transfer

  HandleStop(req, Status(RET_SUCCESS), 0);

  EXPECT_TRUE(req->finished);
  EXPECT_GT(req->finish_time_us.load(), 0u);
  EXPECT_EQ(cache_->destroyed.size(), 1u);
}

TEST_F(EventDrivenScheduleProcessorWhiteBox, HandleStopRequestPdV2SkipsKvTransferWhenPfExceedsRtt) {
  auto req = MakeReq(2, /*token_num=*/3);
  RegisterReq(req);

  const uint64_t arrive = req->timestamp_in_us;
  req->scheduled_time_us.store(arrive + 10);
  req->first_token_time_us.store(arrive + 20);  // scheduled < first_token -> skip decode_queue
  req->pd_v2.prefill_dispatched_time_us.store(arrive + 5);
  req->pd_v2.blocks_allocated_time_us.store(0);            // skip admit (blocks_alloc < arrive)
  req->pd_v2.prefill_complete_time_us.store(arrive + 15);  // remote_rtt = 10
  req->pd_v2.pf_queue_us.store(100);                       // pf_total > remote_rtt -> skip kv_transfer
  req->pd_v2.pf_compute_us.store(0);
  req->pd_v2.pf_send_us.store(0);

  HandleStop(req, Status(RET_SUCCESS), 0);
  EXPECT_TRUE(req->finished);
}

TEST_F(EventDrivenScheduleProcessorWhiteBox, HandleStopRequestSkipsMetricsForPrefillDecodeTargets) {
  auto req = MakeReq(3, /*token_num=*/4);
  RegisterReq(req);
  // Prefill-side requests carry decode_targets; metrics gate requires empty.
  req->pd_v2.decode_targets.push_back(pd_v2::DecodeTargetLite{});
  req->pd_v2.prefill_dispatched_time_us.store(12345);

  HandleStop(req, Status(RET_SUCCESS), 0);
  EXPECT_TRUE(req->finished);
}

TEST_F(EventDrivenScheduleProcessorWhiteBox, HandleStopRequestSkipsMetricsOnErrorOrNoOutput) {
  // Error status: skip the whole metrics block.
  {
    auto req = MakeReq(4, /*token_num=*/4);
    RegisterReq(req);
    req->pd_v2.prefill_dispatched_time_us.store(1);
    HandleStop(req, Status(RET_RUNTIME_FAILED, "boom"), 0);
    EXPECT_TRUE(req->finished);
  }
  // output_token_num == 0 (output_tokens == input_tokens): also skip.
  {
    auto req = MakeReq(5, /*token_num=*/0);
    req->input_tokens = {1, 2, 3};
    req->output_tokens = req->input_tokens;
    RegisterReq(req);
    req->pd_v2.prefill_dispatched_time_us.store(1);
    HandleStop(req, Status(RET_SUCCESS), 0);
    EXPECT_TRUE(req->finished);
  }
}

TEST_F(EventDrivenScheduleProcessorWhiteBox, HandleStopRequestPdV2SkipsRemoteWhenCompleteBeforeDispatch) {
  auto req = MakeReq(6, /*token_num=*/2);
  RegisterReq(req);

  const uint64_t arrive = req->timestamp_in_us;
  req->scheduled_time_us.store(arrive + 50);
  req->first_token_time_us.store(arrive + 40);
  req->pd_v2.prefill_dispatched_time_us.store(arrive + 100);
  req->pd_v2.blocks_allocated_time_us.store(arrive + 10);
  req->pd_v2.prefill_complete_time_us.store(arrive + 50);  // complete < dispatched -> skip rtt block

  HandleStop(req, Status(RET_SUCCESS), 0);
  EXPECT_TRUE(req->finished);
}

// ========================= HandleHookFailAndEnd =========================

TEST_F(EventDrivenScheduleProcessorWhiteBox, HandleHookFailAndEndNullAndStopped) {
  HandleFail(nullptr, 0, "x");  // null -> ERROR + return (no crash)
  auto req = MakeReq(1);
  RegisterReq(req);
  req->Stop();
  HandleFail(req, 0, "x");  // already stopped -> skip
  EXPECT_EQ(cache_->destroyed.size(), 0u);
}

TEST_F(EventDrivenScheduleProcessorWhiteBox, HandleHookFailAndEndNormalSetsRuntimeFailed) {
  auto req = MakeReq(1);
  RegisterReq(req);
  DpState().prefilling_reqs.push_back(req);
  DpState().preprocessing_reqs[1] = req;

  HandleFail(req, 0, "boom");

  EXPECT_TRUE(req->finished);
  EXPECT_EQ(req->finish_status.GetCode(), RET_RUNTIME_FAILED);
  EXPECT_TRUE(DpState().prefilling_reqs.empty());
  EXPECT_EQ(DpState().preprocessing_reqs.count(1), 0u);
  EXPECT_EQ(hook_->exited.size(), 1u);
}

TEST_F(EventDrivenScheduleProcessorWhiteBox, HandleHookFailAndEndPreservesStatusAndSkipsDoubleNotify) {
  auto req = MakeReq(1);
  RegisterReq(req);
  // Simulate an already-finished, already-status'd request (e.g. timed-out / aborted).
  req->finished = true;
  req->finish_status = Status(RET_REQUEST_TIMEOUT, "timeout");

  HandleFail(req, 0, "boom");

  // Specific status preserved, not overwritten by RET_RUNTIME_FAILED.
  EXPECT_EQ(req->finish_status.GetCode(), RET_REQUEST_TIMEOUT);
}

// ========================= HandleRecomputeRequest =========================

TEST_F(EventDrivenScheduleProcessorWhiteBox, HandleRecomputeNullAndStopped) {
  HandleRecompute(nullptr, Status(), 0);  // null -> ERROR + return
  auto req = MakeReq(1);
  RegisterReq(req);
  req->Stop();
  HandleRecompute(req, Status(), 0);  // stopped -> drop
  EXPECT_EQ(cache_->destroyed.size(), 0u);
}

TEST_F(EventDrivenScheduleProcessorWhiteBox, HandleRecomputeNonPdV2RequeuesPrefill) {
  auto req = MakeReq(1);
  RegisterReq(req);
  DpState().in_plan_schedule_ops.recompute_req_ids.insert(1);

  HandleRecompute(req, Status(), 0);

  EXPECT_FALSE(req->IsStopped());  // Restart() un-stops
  ASSERT_EQ(DpState().prefilling_reqs.size(), 1u);
  EXPECT_EQ(DpState().prefilling_reqs[0]->req_id, 1);
  EXPECT_EQ(DpState().schedule_ops_for_nodes.recomputed_req_ids.size(), 1u);
  EXPECT_EQ(cache_->destroyed.size(), 1u);
}

TEST_F(EventDrivenScheduleProcessorWhiteBox, HandleRecomputeAlreadyInPrefillingNoDuplicate) {
  auto req = MakeReq(1);
  RegisterReq(req);
  // Pre-seed prefilling_reqs so HasReqInQueue() iterates a populated queue and
  // returns true -> recompute must NOT push a duplicate.
  DpState().prefilling_reqs.push_back(req);

  HandleRecompute(req, Status(), 0);

  EXPECT_EQ(DpState().prefilling_reqs.size(), 1u);
}

TEST_F(EventDrivenScheduleProcessorWhiteBox, HandleRecomputeKvFromRemoteGoesToPdRestart) {
  auto req = MakeReq(1);
  RegisterReq(req);
  req->pd_v2.kv_from_remote = true;  // forces HandlePdV2DecodeRestart path

  HandleRecompute(req, Status(), 0);

  // PD restart re-injects as an AddRequest event and re-registers the index.
  EXPECT_FALSE(req->pd_v2.kv_from_remote);
  EXPECT_TRUE(req->pd_v2.restart_resubmit);
  EXPECT_EQ(req->infer_stage, InferStage::kContext);
  EXPECT_EQ(DpState().schedule_ops_for_nodes.recomputed_req_ids.size(), 1u);
  // Not pushed into prefilling_reqs (the non-PD branch is skipped).
  EXPECT_TRUE(DpState().prefilling_reqs.empty());
}

TEST_F(EventDrivenScheduleProcessorWhiteBox, HandleRecomputePdV2DecodeModeGoesToPdRestart) {
  proc_->SetPdV2DecodeMode(true);
  auto req = MakeReq(1);
  RegisterReq(req);

  HandleRecompute(req, Status(), 0);

  EXPECT_TRUE(req->pd_v2.restart_resubmit);
  EXPECT_TRUE(DpState().prefilling_reqs.empty());
}

// ========================= HandlePdV2DecodeRestart =========================

TEST_F(EventDrivenScheduleProcessorWhiteBox, HandlePdV2DecodeRestartResetsState) {
  auto req = MakeReq(1);
  RegisterReq(req);
  req->pd_v2.kv_from_remote = true;
  req->pd_v2.awaiting_prefill = true;
  req->pd_v2.pending_stop = true;
  req->kv_cache_blocks = {1, 2, 3};
  req->swa_kv_cache_blocks = {4};
  req->kv_cached_token_num = 5;

  HandlePdRestart(req, 0);

  EXPECT_FALSE(req->pd_v2.kv_from_remote);
  EXPECT_FALSE(req->pd_v2.awaiting_prefill);
  EXPECT_FALSE(req->pd_v2.pending_stop);
  EXPECT_TRUE(req->kv_cache_blocks.empty());
  EXPECT_TRUE(req->swa_kv_cache_blocks.empty());
  EXPECT_EQ(req->kv_cached_token_num, 0u);
  EXPECT_TRUE(req->need_recreate_generator);
  EXPECT_TRUE(req->pd_v2.restart_resubmit);
  EXPECT_EQ(hook_->exited.size(), 1u);
  EXPECT_EQ(cache_->destroyed.size(), 1u);
}

// ========================= ApplyBackendEventsToState =========================

TEST_F(EventDrivenScheduleProcessorWhiteBox, ApplyEmptyAndNullReq) {
  EXPECT_FALSE(Apply({}));  // empty -> false

  ScheduleBackendEvent ev;
  ev.type = ScheduleBackendEventType::kStopRequest;
  ev.req_ptr = nullptr;
  EXPECT_FALSE(Apply({ev}));  // null req skipped, no resource released
}

TEST_F(EventDrivenScheduleProcessorWhiteBox, ApplyMoveToDecode) {
  auto req = MakeReq(1);
  auto other = MakeReq(7);
  RegisterReq(req);
  RegisterReq(other);
  DpState().prefilling_reqs.push_back(req);
  DpState().decoding_reqs.push_back(other);  // populate so HasReqInQueue iterates

  ScheduleBackendEvent ev;
  ev.type = ScheduleBackendEventType::kMoveToDecode;
  ev.dp_idx = 0;
  ev.req_ptr = req;
  EXPECT_FALSE(Apply({ev}));  // move is not a resource release

  EXPECT_TRUE(DpState().prefilling_reqs.empty());
  ASSERT_EQ(DpState().decoding_reqs.size(), 2u);
}

TEST_F(EventDrivenScheduleProcessorWhiteBox, ApplyStopImmediate) {
  auto req = MakeReq(1);
  RegisterReq(req);
  DpState().decoding_reqs.push_back(req);

  ScheduleBackendEvent ev;
  ev.type = ScheduleBackendEventType::kStopRequest;
  ev.dp_idx = 0;
  ev.req_ptr = req;
  ev.status = Status(RET_SUCCESS);

  EXPECT_TRUE(Apply({ev}));                      // resource released
  EXPECT_EQ(hook_->notify_blocks_freed_cnt, 1);  // NotifyBlocksFreed fired
  EXPECT_TRUE(req->IsStopped());
}

TEST_F(EventDrivenScheduleProcessorWhiteBox, ApplyStopWithNullPlanningBatch) {
  auto req = MakeReq(1);
  RegisterReq(req);
  DpState().decoding_reqs.push_back(req);
  Planning()[0] = nullptr;  // exercise the null planning_batch guard in remove_from_state_and_batches

  ScheduleBackendEvent ev;
  ev.type = ScheduleBackendEventType::kStopRequest;
  ev.dp_idx = 0;
  ev.req_ptr = req;
  EXPECT_TRUE(Apply({ev}));
  EXPECT_TRUE(req->IsStopped());
}

TEST_F(EventDrivenScheduleProcessorWhiteBox, ApplyRecomputeImmediate) {
  auto req = MakeReq(1);
  RegisterReq(req);

  ScheduleBackendEvent ev;
  ev.type = ScheduleBackendEventType::kRecomputeRequest;
  ev.dp_idx = 0;
  ev.req_ptr = req;

  EXPECT_TRUE(Apply({ev}));
  EXPECT_EQ(DpState().prefilling_reqs.size(), 1u);
}

TEST_F(EventDrivenScheduleProcessorWhiteBox, ApplyWithInflightDefersEvent) {
  auto req = MakeReq(1);
  RegisterReq(req);
  MakeInflight(req);
  ASSERT_TRUE(req->HasInflightTask());

  ScheduleBackendEvent ev;
  ev.type = ScheduleBackendEventType::kStopRequest;
  ev.dp_idx = 0;
  ev.req_ptr = req;

  EXPECT_FALSE(Apply({ev}));  // deferred, not applied -> no resource released
  EXPECT_EQ(Delayed().size(), 1u);
}

// ========================= ProcessDelayedBackendEvents =========================

TEST_F(EventDrivenScheduleProcessorWhiteBox, ProcessDelayedEmpty) { EXPECT_FALSE(ProcessDelayed()); }

TEST_F(EventDrivenScheduleProcessorWhiteBox, ProcessDelayedNullAndStillInflightAndReady) {
  // (1) null req entry -> erased.
  {
    ScheduleBackendEvent ev;
    ev.req_ptr = nullptr;
    Delayed().push_back(ev);
  }
  // (2) still-inflight req -> kept.
  auto busy = MakeReq(2);
  RegisterReq(busy);
  MakeInflight(busy);
  {
    ScheduleBackendEvent ev;
    ev.type = ScheduleBackendEventType::kStopRequest;
    ev.req_ptr = busy;
    Delayed().push_back(ev);
  }
  // (3) ready (no inflight) stop -> applied.
  auto ready = MakeReq(3);
  RegisterReq(ready);
  {
    ScheduleBackendEvent ev;
    ev.type = ScheduleBackendEventType::kStopRequest;
    ev.req_ptr = ready;
    Delayed().push_back(ev);
  }

  EXPECT_TRUE(ProcessDelayed());
  EXPECT_TRUE(ready->IsStopped());
  // null removed + ready removed; busy stays.
  ASSERT_EQ(Delayed().size(), 1u);
  EXPECT_EQ(Delayed()[0].req_ptr->req_id, 2);

  // Now retire busy's inflight and process the recompute-flavored delayed event.
  busy->ResetInflightTask();
  Delayed()[0].type = ScheduleBackendEventType::kRecomputeRequest;
  EXPECT_TRUE(ProcessDelayed());
  EXPECT_TRUE(Delayed().empty());
}

// ========================= OnEventAddRequest =========================

TEST_F(EventDrivenScheduleProcessorWhiteBox, OnAddNullAndEmpty) {
  OnAdd(nullptr);
  auto info = std::make_shared<EventAddRequestInfo>();
  OnAdd(info);  // empty reqs -> early return
  SUCCEED();
}

TEST_F(EventDrivenScheduleProcessorWhiteBox, OnAddContinueRoutesToPrefilling) {
  auto req = MakeReq(1);
  auto info = std::make_shared<EventAddRequestInfo>();
  info->reqs = {req};
  hook_->default_action = NextStepAction::kContinue;

  OnAdd(info);

  EXPECT_TRUE(hook_->process_called);
  ASSERT_EQ(DpState().prefilling_reqs.size(), 1u);
  EXPECT_EQ(proc_->GetScheduleState()->req_to_batch_idx.at(1), 0u);
}

TEST_F(EventDrivenScheduleProcessorWhiteBox, OnAddPdV2SkipPrefillRoutesToDecoding) {
  auto req = MakeReq(1);
  auto info = std::make_shared<EventAddRequestInfo>();
  info->reqs = {req};
  hook_->default_action = NextStepAction::kContinue;
  hook_->set_decode_stage = true;  // hook flips infer_stage to kDecode (post ResetPrefillingTokens)

  OnAdd(info);

  ASSERT_EQ(DpState().decoding_reqs.size(), 1u);
  EXPECT_TRUE(DpState().prefilling_reqs.empty());
}

TEST_F(EventDrivenScheduleProcessorWhiteBox, OnAddNoNextStepDefaultsToPrefilling) {
  auto req = MakeReq(1);
  auto info = std::make_shared<EventAddRequestInfo>();
  info->reqs = {req};
  hook_->use_process_override = true;
  hook_->process_result = {};  // no entry for req -> "no next_step" warning branch

  OnAdd(info);

  ASSERT_EQ(DpState().prefilling_reqs.size(), 1u);
}

TEST_F(EventDrivenScheduleProcessorWhiteBox, OnAddWaitAsyncMovesToPreprocessing) {
  auto req = MakeReq(1);
  auto info = std::make_shared<EventAddRequestInfo>();
  info->reqs = {req};
  hook_->default_action = NextStepAction::kWaitAsyncResult;

  OnAdd(info);

  EXPECT_EQ(DpState().preprocessing_reqs.count(1), 1u);
  EXPECT_TRUE(DpState().prefilling_reqs.empty());
}

TEST_F(EventDrivenScheduleProcessorWhiteBox, OnAddFailAndEndCleansUp) {
  auto req = MakeReq(1);
  auto info = std::make_shared<EventAddRequestInfo>();
  info->reqs = {req};
  hook_->default_action = NextStepAction::kFailAndEnd;

  OnAdd(info);

  EXPECT_TRUE(req->finished);
  EXPECT_EQ(req->finish_status.GetCode(), RET_RUNTIME_FAILED);
}

TEST_F(EventDrivenScheduleProcessorWhiteBox, OnAddKeepsOutputForRestartResubmit) {
  auto req = MakeReq(1, 4);
  req->output_tokens.push_back(777);  // accumulated generated token
  size_t before = req->output_tokens.size();
  req->pd_v2.restart_resubmit = true;
  auto info = std::make_shared<EventAddRequestInfo>();
  info->reqs = {req};

  OnAdd(info);

  // restart_resubmit must NOT reset output_tokens to input_tokens.
  EXPECT_EQ(req->output_tokens.size(), before);
  EXPECT_FALSE(req->pd_v2.restart_resubmit);  // consumed
}

TEST_F(EventDrivenScheduleProcessorWhiteBox, OnAddWithDistributor) {
  proc_->SetPpBatchDistributor(std::make_shared<PpBatchDistributor>(PPMultibatchWBStrategy::NO_WB));
  auto req = MakeReq(1);
  auto info = std::make_shared<EventAddRequestInfo>();
  info->reqs = {req};
  hook_->default_action = NextStepAction::kWaitAsyncResult;  // distributor placed -> moved to preprocessing

  OnAdd(info);

  EXPECT_EQ(DpState().preprocessing_reqs.count(1), 1u);
}

// ========================= UpdateAfterAsyncTaskFinished =========================

TEST_F(EventDrivenScheduleProcessorWhiteBox, UpdateAsyncEmptyReturns) {
  EventAsyncTaskFinishedInfo info;  // empty task_results
  UpdateAsync(info);
  SUCCEED();
}

TEST_F(EventDrivenScheduleProcessorWhiteBox, UpdateAsyncContinueToPrefilling) {
  auto req = MakeReq(1);
  RegisterReq(req);
  DpState().preprocessing_reqs[1] = req;
  hook_->async_result = {{1, NextStep{NextStepAction::kContinue, 0}}};

  EventAsyncTaskFinishedInfo info;
  info.task_results.push_back(AsyncTaskResult{});
  UpdateAsync(info);

  ASSERT_EQ(DpState().prefilling_reqs.size(), 1u);
  EXPECT_EQ(DpState().preprocessing_reqs.count(1), 0u);
}

TEST_F(EventDrivenScheduleProcessorWhiteBox, UpdateAsyncContinueToDecoding) {
  auto req = MakeReq(1);
  req->infer_stage = InferStage::kDecode;
  RegisterReq(req);
  DpState().preprocessing_reqs[1] = req;
  hook_->async_result = {{1, NextStep{NextStepAction::kContinue, 0}}};

  EventAsyncTaskFinishedInfo info;
  info.task_results.push_back(AsyncTaskResult{});
  UpdateAsync(info);

  ASSERT_EQ(DpState().decoding_reqs.size(), 1u);
}

TEST_F(EventDrivenScheduleProcessorWhiteBox, UpdateAsyncContinueOrphanWhenNotPreprocessing) {
  auto req = MakeReq(1);
  RegisterReq(req);
  // NOT in preprocessing_reqs -> orphan ERROR branch.
  hook_->async_result = {{1, NextStep{NextStepAction::kContinue, 0}}};

  EventAsyncTaskFinishedInfo info;
  info.task_results.push_back(AsyncTaskResult{});
  UpdateAsync(info);

  EXPECT_TRUE(DpState().prefilling_reqs.empty());
}

TEST_F(EventDrivenScheduleProcessorWhiteBox, UpdateAsyncFailAndEnd) {
  auto req = MakeReq(1);
  RegisterReq(req);
  DpState().preprocessing_reqs[1] = req;
  hook_->async_result = {{1, NextStep{NextStepAction::kFailAndEnd, 0}}};

  EventAsyncTaskFinishedInfo info;
  info.task_results.push_back(AsyncTaskResult{});
  UpdateAsync(info);

  EXPECT_TRUE(req->finished);
}

TEST_F(EventDrivenScheduleProcessorWhiteBox, UpdateAsyncWaitDefault) {
  auto req = MakeReq(1);
  RegisterReq(req);
  DpState().preprocessing_reqs[1] = req;
  hook_->async_result = {{1, NextStep{NextStepAction::kWaitAsyncResult, 0}}};

  EventAsyncTaskFinishedInfo info;
  info.task_results.push_back(AsyncTaskResult{});
  UpdateAsync(info);

  EXPECT_EQ(DpState().preprocessing_reqs.count(1), 1u);  // stays
}

// ========================= OnEventAsyncTaskFinished / Enqueue =========================

TEST_F(EventDrivenScheduleProcessorWhiteBox, OnAsyncEmptyAndNonEmpty) {
  OnAsync(nullptr);
  auto info = std::make_shared<EventAsyncTaskFinishedInfo>();
  OnAsync(info);  // empty -> early return

  auto req = MakeReq(1);
  RegisterReq(req);
  DpState().preprocessing_reqs[1] = req;
  hook_->async_result = {{1, NextStep{NextStepAction::kContinue, 0}}};
  info->task_results.push_back(AsyncTaskResult{});
  OnAsync(info);
  EXPECT_EQ(DpState().prefilling_reqs.size(), 1u);
}

TEST_F(EventDrivenScheduleProcessorWhiteBox, EnqueueEmptyAndNonEmpty) {
  EventAsyncTaskFinishedInfo empty;
  Enqueue(empty);  // empty -> no-op
  EventAsyncTaskFinishedInfo info;
  info.task_results.push_back(AsyncTaskResult{});
  Enqueue(info);  // enqueued to event_queue_
  SUCCEED();
}

// ========================= DispatchEvent =========================

TEST_F(EventDrivenScheduleProcessorWhiteBox, DispatchAllEventTypes) {
  // EventAddRequest
  {
    auto info = std::make_shared<EventAddRequestInfo>();
    info->reqs = {MakeReq(1)};
    SchedEvent e(info);
    Dispatch(e);
  }
  // EventUpdateAfterGeneration
  {
    auto info = std::make_shared<EventUpdateAfterGenerationInfo>();
    info->gen_results.multi_batch_id = 0;
    SchedEvent e(info);
    Dispatch(e);
  }
  // EventAsyncTaskFinished
  {
    auto info = std::make_shared<EventAsyncTaskFinishedInfo>();
    info->task_results.push_back(AsyncTaskResult{});
    SchedEvent e(info);
    Dispatch(e);
  }
  // EventGetScheduleResult (empty event)
  {
    SchedEvent e;
    Dispatch(e);
  }
  SUCCEED();
}

// ========================= OnEventUpdateAfterGeneration =========================

TEST_F(EventDrivenScheduleProcessorWhiteBox, OnUpdateDecrementsPipelineAndRecordsLatency) {
  PipelineDepth() = 1;
  auto req = MakeReq(1);
  RegisterReq(req);
  // Backend emits a stop event for this request.
  ScheduleBackendEvent ev;
  ev.type = ScheduleBackendEventType::kStopRequest;
  ev.dp_idx = 0;
  ev.req_ptr = req;
  backend_->update_events = {ev};

  auto info = std::make_shared<EventUpdateAfterGenerationInfo>();
  info->gen_results.multi_batch_id = 0;
  OnUpdate(info);

  EXPECT_EQ(PipelineDepth(), 0u);
  EXPECT_TRUE(req->IsStopped());
}

// ========================= TryBuildAndLaunchScheduleResult =========================

TEST_F(EventDrivenScheduleProcessorWhiteBox, TryBuildLaunches) {
  auto req = MakeReq(1);
  RegisterReq(req);
  Planning()[0]->AddRunningReq(0, req);
  DpState().schedule_ops_for_nodes.stopped_req_ids.push_back(42);

  EXPECT_TRUE(TryBuild());
  EXPECT_EQ(PipelineDepth(), 1u);
  EXPECT_FALSE(Pending());
  // schedule_ops_for_nodes reset after build.
  EXPECT_TRUE(DpState().schedule_ops_for_nodes.stopped_req_ids.empty());
}

TEST_F(EventDrivenScheduleProcessorWhiteBox, TryBuildPipelineFullPending) {
  auto req = MakeReq(1);
  RegisterReq(req);
  Planning()[0]->AddRunningReq(0, req);
  PipelineDepth() = 2;  // == kMaxPipelineDepth

  EXPECT_FALSE(TryBuild());
  EXPECT_TRUE(Pending());  // any_pending -> pending set
}

TEST_F(EventDrivenScheduleProcessorWhiteBox, TryBuildWithPutTrackerOnBatchDispatched) {
  auto tracker = std::make_shared<GlobalCachePutTracker>(1, cache_);
  tracker->SetDpCacheManagers({cache_});
  proc_->SetGlobalCachePutTracker(tracker);

  auto req = MakeReq(1);
  RegisterReq(req);
  Planning()[0]->AddRunningReq(0, req);

  EXPECT_TRUE(TryBuild());  // OnBatchDispatched path executed
}

// ========================= RunScheduleForBatches =========================

TEST_F(EventDrivenScheduleProcessorWhiteBox, RunScheduleNoEvents) {
  RunSched();  // PreSchedule + Schedule emit nothing -> single pass
  EXPECT_EQ(backend_->schedule_calls, 1);
}

TEST_F(EventDrivenScheduleProcessorWhiteBox, RunScheduleResourceReleasedLoops) {
  auto req = MakeReq(1);
  RegisterReq(req);
  // First Schedule() call emits a stop event -> resource released -> loop again.
  ScheduleBackendEvent ev;
  ev.type = ScheduleBackendEventType::kStopRequest;
  ev.dp_idx = 0;
  ev.req_ptr = req;
  backend_->schedule_events = {ev};

  RunSched();

  EXPECT_GE(backend_->schedule_calls, 2);  // looped at least twice
  EXPECT_TRUE(req->IsStopped());
}

// ========================= RecordStepLatency =========================

TEST_F(EventDrivenScheduleProcessorWhiteBox, RecordLatencyNoStepReturns) {
  LastStepLaunchMs() = 0;
  RecordLatency();  // early return
  EXPECT_EQ(StepTokenEwma(), 0u);
}

TEST_F(EventDrivenScheduleProcessorWhiteBox, RecordLatencySeedsThenEwma) {
  LastStepLaunchMs() = ProfileTimer::GetCurrentTimeInMs() - 5;
  LastStepTokenNum() = 100;
  RecordLatency();  // seed
  EXPECT_EQ(StepTokenEwma(), 100u);

  LastStepLaunchMs() = ProfileTimer::GetCurrentTimeInMs() - 5;
  LastStepTokenNum() = 200;
  RecordLatency();  // EWMA: 0.2*200 + 0.8*100 = 120
  EXPECT_EQ(StepTokenEwma(), 120u);
}

// ========================= GetLocalLoad =========================

TEST_F(EventDrivenScheduleProcessorWhiteBox, GetLocalLoadAggregates) {
  auto p = MakeReq(1);
  auto d = MakeReq(2);
  RegisterReq(p);
  RegisterReq(d);
  DpState().prefilling_reqs.push_back(p);
  DpState().decoding_reqs.push_back(d);

  // Off the scheduler thread -> takes the lock then aggregates.
  ScheduleLoadStat stat = proc_->GetLocalLoad();
  EXPECT_EQ(stat.waiting_reqs, 1u);
  EXPECT_EQ(stat.running_reqs, 1u);
  EXPECT_GT(stat.pending_prefill_tokens, 0u);
}

// ========================= Thread-based: Start / Stop / AsyncSchedulerLoop =========================

TEST_F(EventDrivenScheduleProcessorWhiteBox, StartProcessesAndStops) {
  // Stage a request so the backend makes batch 0 launchable on Schedule().
  auto req = MakeReq(1);
  RegisterReq(req);
  backend_->launch_req = req;

  proc_->Start();

  std::vector<std::shared_ptr<InferRequest>> group;
  auto add = MakeReq(2);
  group.push_back(add);
  proc_->AddInferRequest(group);

  auto result = proc_->GetNextScheduleResult();
  ASSERT_NE(result, nullptr);
  EXPECT_FALSE(result->schedule_output->running_reqs.empty());

  // Feed a generation result back through the loop.
  GenerationOutputGroup gen;
  gen.multi_batch_id = 0;
  proc_->UpdateWithGenerationResult(gen);

  proc_->Stop();
  EXPECT_GE(backend_->stop_calls, 1);
}

TEST_F(EventDrivenScheduleProcessorWhiteBox, GetNextScheduleResultUnblocksOnStop) {
  proc_->Start();
  // No launchable batch -> GetNextScheduleResult parks until Stop() pushes nullptr.
  std::shared_ptr<ScheduleResult> result;
  std::thread getter([&] { result = proc_->GetNextScheduleResult(); });
  std::this_thread::sleep_for(std::chrono::milliseconds(120));  // let the pending poll run
  proc_->Stop();
  getter.join();
  EXPECT_EQ(result, nullptr);
}

// ========================= Setters / WorkloadBalancer =========================

TEST_F(EventDrivenScheduleProcessorWhiteBox, SetWorkloadBalancerWiresBackend) {
  auto wb = std::make_shared<FakeWorkloadBalancer>();
  proc_->SetWorkloadBalancer(wb);       // backend_ already set -> SetWorkloadBalancer on backend
  proc_->SetScheduleBackend(backend_);  // backend set again with WB present -> other branch
  SUCCEED();
}

TEST_F(EventDrivenScheduleProcessorWhiteBox, OnAddRunsWorkloadBalancer) {
  auto wb = std::make_shared<FakeWorkloadBalancer>();
  proc_->SetWorkloadBalancer(wb);
  auto req = MakeReq(1);
  auto info = std::make_shared<EventAddRequestInfo>();
  info->reqs = {req};

  OnAdd(info);

  EXPECT_TRUE(wb->add_called);
  EXPECT_EQ(DpState().prefilling_reqs.size(), 1u);
}

TEST_F(EventDrivenScheduleProcessorWhiteBox, TryBuildRecordsInflightBatch) {
  auto wb = std::make_shared<FakeWorkloadBalancer>();
  proc_->SetWorkloadBalancer(wb);
  auto req = MakeReq(1);
  RegisterReq(req);
  Planning()[0]->AddRunningReq(0, req);

  EXPECT_TRUE(TryBuild());
  EXPECT_TRUE(wb->record_called);
}

TEST_F(EventDrivenScheduleProcessorWhiteBox, OnGetResultLaunchableClearsPending) {
  auto req = MakeReq(1);
  RegisterReq(req);
  Planning()[0]->AddRunningReq(0, req);

  OnGetResult();  // launchable -> pending_get_schedule_result_ = false
  EXPECT_FALSE(Pending());
}

TEST_F(EventDrivenScheduleProcessorWhiteBox, OnGetResultNotLaunchableSetsPending) {
  OnGetResult();  // nothing launchable -> pending = true
  EXPECT_TRUE(Pending());
}

// ========================= Lock / Unlock =========================

TEST_F(EventDrivenScheduleProcessorWhiteBox, LockUnlock) {
  proc_->Lock();
  proc_->Unlock();
  SUCCEED();
}

TEST_F(EventDrivenScheduleProcessorWhiteBox, AddInferRequestRejectsOverlengthInput) {
  auto req = MakeReq(1);
  req->input_tokens.resize(cfg_.max_token_len + 1);

  std::vector<std::shared_ptr<InferRequest>> group{req};
  Status status = proc_->AddInferRequest(group);

  EXPECT_EQ(status.GetCode(), RET_INPUT_LENGTH_EXCEEDED);
  EXPECT_EQ(req->finish_status.GetCode(), RET_INPUT_LENGTH_EXCEEDED);
  EXPECT_TRUE(req->finished);
  EXPECT_GT(req->finish_time_us.load(), 0u);
  EXPECT_EQ(proc_->GetScheduleState()->req_id_to_req_ptr.count(req->req_id), 0u);
}

TEST_F(EventDrivenScheduleProcessorWhiteBox, AddInferRequestAcceptsInputAtMaxLength) {
  auto req = MakeReq(1);
  req->input_tokens.resize(cfg_.max_token_len);

  std::vector<std::shared_ptr<InferRequest>> group{req};
  Status status = proc_->AddInferRequest(group);

  EXPECT_TRUE(status.OK());
  EXPECT_FALSE(req->finished);
  EXPECT_EQ(proc_->GetScheduleState()->req_id_to_req_ptr.count(req->req_id), 1u);
}

TEST_F(EventDrivenScheduleProcessorWhiteBox, AddInferRequestRejectNotifiesEachIndependentRequest) {
  auto batch_waiter = std::make_shared<Waiter>(3);
  std::vector<std::shared_ptr<InferRequest>> batch;

  for (int64_t req_id = 1; req_id <= 3; ++req_id) {
    auto req = MakeReq(req_id);
    req->input_tokens.resize(cfg_.max_token_len + 1);
    req->waiter = batch_waiter;

    // Match ModelPerfBackend: every independent Request is a singleton
    // InferRequest group, while all Requests share one batch waiter.
    std::vector<std::shared_ptr<InferRequest>> singleton_group{req};
    req->SetReqGroup(singleton_group);
    batch.push_back(req);
  }

  Status status = proc_->AddInferRequest(batch);

  EXPECT_EQ(status.GetCode(), RET_INPUT_LENGTH_EXCEEDED);
  EXPECT_EQ(batch_waiter->Cnt(), 0);
  for (const auto& req : batch) {
    EXPECT_TRUE(req->finished);
    EXPECT_EQ(req->finish_status.GetCode(), RET_INPUT_LENGTH_EXCEEDED);
    EXPECT_GT(req->finish_time_us.load(), 0u);
    EXPECT_EQ(proc_->GetScheduleState()->req_id_to_req_ptr.count(req->req_id), 0u);
  }
}

TEST_F(EventDrivenScheduleProcessorWhiteBox, AddInferRequestRejectNotifiesBeamGroupOnce) {
  constexpr size_t kBeamNum = 3;
  auto input = std::make_shared<KsanaPythonInput>();
  input->sampling_config.num_beams = kBeamNum;
  input->input_tokens.resize(cfg_.max_token_len + 1);
  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  auto request = std::make_shared<Request>(input, req_ctx);
  auto request_waiter = std::make_shared<Waiter>(1);
  request->waiter = request_waiter;
  g_alive_requests.push_back(request);

  std::vector<std::shared_ptr<InferRequest>> beam_group;
  for (size_t beam_idx = 0; beam_idx < request->output_group.size(); ++beam_idx) {
    beam_group.push_back(std::make_shared<InferRequest>(request, beam_idx));
  }
  ASSERT_EQ(beam_group.size(), kBeamNum);
  for (auto& req : beam_group) {
    req->SetReqGroup(beam_group);
  }

  Status status = proc_->AddInferRequest(beam_group);

  EXPECT_EQ(status.GetCode(), RET_INPUT_LENGTH_EXCEEDED);
  // A beam group represents one underlying Request, so its Waiter(1) must
  // reach exactly zero rather than being notified once per beam.
  EXPECT_EQ(request_waiter->Cnt(), 0);
  for (const auto& req : beam_group) {
    EXPECT_TRUE(req->finished);
    EXPECT_EQ(req->finish_status.GetCode(), RET_INPUT_LENGTH_EXCEEDED);
    EXPECT_GT(req->finish_time_us.load(), 0u);
    EXPECT_EQ(proc_->GetScheduleState()->req_id_to_req_ptr.count(req->req_id), 0u);
  }
}

}  // namespace ksana_llm
