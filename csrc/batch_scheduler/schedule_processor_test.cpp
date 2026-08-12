/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "batch_scheduler/schedule_processor.h"

#include <memory>
#include <vector>

#include "batch_scheduler/batch_scheduler_interface.h"
#include "cache_manager/cache_manager_interface.h"
#include "runtime/infer_request.h"
#include "runtime/schedule_output.h"
#include "utils/logger.h"
#include "utils/request.h"

#include "test.h"

namespace ksana_llm {

namespace {

// ---------- Minimal fake BatchScheduler: only what the two helpers touch. ----------
class FakeBatchScheduler : public BatchSchedulerInterface {
 public:
  bool launch_ok = false;
  std::vector<std::shared_ptr<InferRequest>> stop_on_launch;
  int launch_call_count = 0;

  bool TryToLaunchPlannedScheduleOutput(size_t, ScheduleOutput&,
                                        std::vector<std::shared_ptr<InferRequest>>& stopped_reqs) override {
    ++launch_call_count;
    if (!launch_ok) return false;
    for (auto& r : stop_on_launch) stopped_reqs.push_back(r);
    return true;
  }

  std::shared_ptr<ScheduleOutputGroup> Schedule(size_t) override { return nullptr; }
  void UpdateWithGenerationResult(size_t, const GenerationOutputGroup&) override {}
  Status AddInferRequest(std::vector<std::shared_ptr<InferRequest>>&) override { return Status(); }
  void SetCacheManager(std::shared_ptr<CacheManagerInterface>, int) override {}
  std::shared_ptr<CacheManagerInterface>& GetCacheManager(int) override { return null_cache_; }
  bool IsIdle(size_t) override { return true; }
  void WaitUntilHaveReqs(size_t) override {}
  void Stop() override {}

 private:
  std::shared_ptr<CacheManagerInterface> null_cache_;
};

class ScheduleProcessorTest : public testing::Test {
 protected:
  void SetUp() override {
    InitLoguru();
    fake_ = std::make_shared<FakeBatchScheduler>();
  }
  void TearDown() override { alive_requests_.clear(); }

  // Wire batch_scheduler_ and resize planning_sched_results_ without spawning
  // any scheduling thread.
  void SetupWithoutThreads(ScheduleProcessor& p) {
    p.batch_scheduler_ = fake_;
    p.planning_sched_results_.assign(p.max_pp_batch_num_, nullptr);
  }

  std::shared_ptr<InferRequest> MakeReq(int64_t req_id) {
    auto input = std::make_shared<KsanaPythonInput>();
    input->input_tokens = {1};
    auto ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
    auto request = std::make_shared<Request>(input, ctx);
    alive_requests_.push_back(request);
    auto req = std::make_shared<InferRequest>(request, 0);
    req->req_id = req_id;
    req->attn_dp_group_id = 0;
    return req;
  }

  // Oldest inflight is a chunk-mid prefill step: prefilling_tokens_ has `total`
  // tokens, planning workload covers only `chunk` (< total).
  std::shared_ptr<InferRequest> MakeChunkMidInflightReq(int64_t req_id, size_t total = 10, size_t chunk = 4) {
    auto req = MakeReq(req_id);
    req->output_tokens.clear();
    for (size_t i = 0; i < total; ++i) req->output_tokens.push_back(static_cast<int>(i));
    req->ResetPrefillingTokens();

    ScheduleTaskWorkload wl;
    wl.prefill_token_num = total;
    wl.prefill_start_offset = 0;
    wl.sampling_token_num = 1;
    req->SetRemainingWorkload(wl);

    ScheduleTaskWorkload planned;
    planned.prefill_token_num = chunk;
    planned.prefill_start_offset = 0;
    planned.sampling_token_num = 1;
    req->SetPlanningWorkload(planned);
    req->SetPlanningTask();
    req->LaunchPlanningTask();
    return req;
  }

  std::shared_ptr<ScheduleResult> MakeResult(std::vector<std::shared_ptr<InferRequest>> running) {
    auto r = std::make_shared<ScheduleResult>();
    r->schedule_output = std::make_shared<ScheduleOutput>();
    r->schedule_output->running_reqs = std::move(running);
    r->generation_output_group.reqs.resize(1);  // attn_dp_group_id=0 slot.
    return r;
  }

  std::shared_ptr<FakeBatchScheduler> fake_;
  std::vector<std::shared_ptr<Request>> alive_requests_;
};

// ---------- TryConsumeScheduleResult ----------

TEST_F(ScheduleProcessorTest, TryConsume_EmptySlot_ReturnsFalse) {
  ScheduleProcessor proc(/*enable_async=*/false, /*max_pp_batch_num=*/2);
  SetupWithoutThreads(proc);
  EXPECT_FALSE(proc.TryConsumeScheduleResult(0));
  EXPECT_EQ(fake_->launch_call_count, 0);
}

TEST_F(ScheduleProcessorTest, TryConsume_LaunchStillFails_KeepsParked) {
  ScheduleProcessor proc(/*enable_async=*/false, /*max_pp_batch_num=*/2);
  SetupWithoutThreads(proc);
  proc.planning_sched_results_[0] = MakeResult({MakeReq(1)});

  fake_->launch_ok = false;
  EXPECT_FALSE(proc.TryConsumeScheduleResult(0));
  EXPECT_NE(proc.planning_sched_results_[0], nullptr);  // still parked for next retry.
  EXPECT_EQ(fake_->launch_call_count, 1);
}

TEST_F(ScheduleProcessorTest, TryConsume_LaunchOk_ClearsSlotAndEnqueues) {
  ScheduleProcessor proc(/*enable_async=*/false, /*max_pp_batch_num=*/2);
  SetupWithoutThreads(proc);
  auto req = MakeReq(1);
  proc.planning_sched_results_[0] = MakeResult({req});

  fake_->launch_ok = true;
  EXPECT_TRUE(proc.TryConsumeScheduleResult(0));
  EXPECT_EQ(proc.planning_sched_results_[0], nullptr);

  auto delivered = proc.sched_result_queue_.NonBlockingGet();
  ASSERT_NE(delivered, nullptr);
  ASSERT_EQ(delivered->schedule_output->running_reqs.size(), 1u);
  EXPECT_EQ(delivered->schedule_output->running_reqs[0], req);
}

TEST_F(ScheduleProcessorTest, TryConsume_LaunchOk_RemovesStoppedReqsFromGenerationGroup) {
  ScheduleProcessor proc(/*enable_async=*/false, /*max_pp_batch_num=*/2);
  SetupWithoutThreads(proc);
  auto req = MakeReq(1);
  auto stopped = MakeReq(2);
  auto r = MakeResult({req});
  r->generation_output_group.reqs[0] = {req, stopped};  // both in generation group before launch.
  proc.planning_sched_results_[0] = r;

  fake_->launch_ok = true;
  fake_->stop_on_launch = {stopped};

  EXPECT_TRUE(proc.TryConsumeScheduleResult(0));
  // The stopped req must have been pulled out of the generation group so the
  // engine won't emit a spurious sample for it.
  const auto& reqs0 = r->generation_output_group.reqs[0];
  ASSERT_EQ(reqs0.size(), 1u);
  EXPECT_EQ(reqs0[0]->req_id, 1);
}

// ---------- NotifyNextPPMultiBatch ----------

TEST_F(ScheduleProcessorTest, NotifyNextPP_SinglePP_EarlyReturn) {
  ScheduleProcessor proc(/*enable_async=*/false, /*max_pp_batch_num=*/1);
  auto so = std::make_shared<ScheduleOutput>();
  so->running_reqs.push_back(MakeChunkMidInflightReq(1));
  const size_t before = so->running_reqs[0]->GetInflightCount();

  proc.NotifyNextPPMultiBatch(0, so);

  EXPECT_EQ(so->running_reqs[0]->GetInflightCount(), before);
}

TEST_F(ScheduleProcessorTest, NotifyNextPP_NoChunkMid_NoRetire) {
  ScheduleProcessor proc(/*enable_async=*/false, /*max_pp_batch_num=*/2);
  SetupWithoutThreads(proc);

  auto so = std::make_shared<ScheduleOutput>();
  auto req = MakeReq(1);
  so->running_reqs.push_back(req);
  ASSERT_FALSE(req->HasInflightTask());

  proc.NotifyNextPPMultiBatch(0, so);
  EXPECT_EQ(req->GetInflightCount(), 0u);
}

TEST_F(ScheduleProcessorTest, NotifyNextPP_ChunkMid_RetiresAndTriggersNextMb) {
  ScheduleProcessor proc(/*enable_async=*/false, /*max_pp_batch_num=*/2);
  SetupWithoutThreads(proc);

  auto so = std::make_shared<ScheduleOutput>();
  auto req = MakeChunkMidInflightReq(1);
  so->running_reqs.push_back(req);
  ASSERT_EQ(req->GetInflightCount(), 1u);
  ASSERT_TRUE(req->IsSplitPrefillStep());

  // Park a result on the *next* mb (0 -> next is 1). After NotifyNextPPMultiBatch,
  // it should attempt to consume that slot; make it fail-to-launch so we can
  // simply observe the call count.
  proc.planning_sched_results_[1] = MakeResult({MakeReq(99)});
  fake_->launch_ok = false;

  proc.NotifyNextPPMultiBatch(/*multi_batch_id=*/0, so);

  // Retire happened: oldest inflight was popped, chunk-mid counter bumped.
  EXPECT_EQ(req->GetInflightCount(), 0u);
  EXPECT_FALSE(req->UpdateAfterInflightTaskFinished()) << "chunk-mid counter must skip the next sample update";
  // Consume was attempted on next mb (id 1).
  EXPECT_EQ(fake_->launch_call_count, 1);
}

}  // namespace
}  // namespace ksana_llm
