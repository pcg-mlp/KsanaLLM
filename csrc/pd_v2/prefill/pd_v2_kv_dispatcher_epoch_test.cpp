/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
// Epoch-bound CaptureEpochReqs: BuildDispatchPlans must resolve against the
// BeginBatch-pinned set, never the live ExecutorRequestManager (force-fired
// callbacks under pipelining can otherwise hit a half-built next batch).

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "pd_v2/prefill/pd_v2_kv_dispatcher.h"
#include "runtime/executor_infer_request.h"
#include "runtime/infer_stage.h"

namespace ksana_llm {
namespace pd_v2 {

// Test peer: inspect private epoch state; CallBuild holds batch_reqs_mu_.
class PdV2KvDispatcherEpochTestPeer : public PdV2KvDispatcher {
 public:
  explicit PdV2KvDispatcherEpochTestPeer(Deps deps) : PdV2KvDispatcher(std::move(deps)) {}

  size_t CapturedEpochCount() {
    std::lock_guard<std::mutex> lk(batch_reqs_mu_);
    return reqs_by_epoch_.size();
  }
  bool HasEpoch(uint64_t epoch_id) {
    std::lock_guard<std::mutex> lk(batch_reqs_mu_);
    return reqs_by_epoch_.count(epoch_id) > 0;
  }
  size_t CapturedReqCount(uint64_t epoch_id) {
    std::lock_guard<std::mutex> lk(batch_reqs_mu_);
    auto it = reqs_by_epoch_.find(epoch_id);
    return it == reqs_by_epoch_.end() ? 0 : it->second.size();
  }
  std::vector<ReqDispatchPlan> CallBuild(int rank, uint64_t epoch_id) {
    std::lock_guard<std::mutex> lk(batch_reqs_mu_);
    return BuildDispatchPlans(rank, epoch_id);
  }
};

namespace {

PdV2KvDispatcher::Deps MakeDeps() {
  PdV2KvDispatcher::Deps deps;
  deps.connector = nullptr;
  deps.request_manager = nullptr;
  deps.barrier = nullptr;
  deps.num_layers = 4;
  deps.prefill_tp = 8;
  deps.prefill_attn_tp_per_dp = 8;
  deps.transfer_layer_chunk_size = 1;
  deps.use_mla = false;
  return deps;
}

// Minimal req routing to device_id; null block_allocator_manager on purpose
// (segfault if GetBlockPtrs is wrongly reached).
std::shared_ptr<ExecutorInferRequest> MakeReqRoutingTo(int64_t req_id, uint32_t device_id) {
  auto req = std::make_shared<ExecutorInferRequest>();
  req->req_id = req_id;
  req->infer_stage = InferStage::kContext;
  req->step = 0;
  req->block_allocator_manager = nullptr;
  req->pd_v2_decode_attn_dp_group_id = device_id;  // d_attn_tp==1 → target == device_id
  req->pd_v2_decode_engine_inference_addr = "127.0.0.1:9000";
  DecodeTargetLite t;
  t.decode_attn_tp_rank_in_group = 0;
  req->pd_v2_decode_targets = {t};
  return req;
}

}  // namespace

TEST(PdV2KvDispatcherEpoch, EpochZeroIsNoOp) {
  PdV2KvDispatcherEpochTestPeer disp(MakeDeps());
  disp.CaptureEpochReqs(0, {MakeReqRoutingTo(1, 0)});
  EXPECT_EQ(disp.CapturedEpochCount(), 0u);
  EXPECT_FALSE(disp.HasEpoch(0));
}

TEST(PdV2KvDispatcherEpoch, CapturePinsStrongRefs) {
  PdV2KvDispatcherEpochTestPeer disp(MakeDeps());
  auto req = MakeReqRoutingTo(1, 0);
  ASSERT_EQ(req.use_count(), 1);

  disp.CaptureEpochReqs(7, {req});
  EXPECT_EQ(req.use_count(), 2);
  EXPECT_TRUE(disp.HasEpoch(7));
  EXPECT_EQ(disp.CapturedReqCount(7), 1u);
}

TEST(PdV2KvDispatcherEpoch, EpochsAreIsolated) {
  PdV2KvDispatcherEpochTestPeer disp(MakeDeps());
  disp.CaptureEpochReqs(1, {MakeReqRoutingTo(10, 0)});
  disp.CaptureEpochReqs(2, {MakeReqRoutingTo(20, 0), MakeReqRoutingTo(21, 1)});

  EXPECT_EQ(disp.CapturedEpochCount(), 2u);
  EXPECT_EQ(disp.CapturedReqCount(1), 1u);
  EXPECT_EQ(disp.CapturedReqCount(2), 2u);
}

TEST(PdV2KvDispatcherEpoch, OnEpochEndReleasesCapturedRefsUnconditionally) {
  PdV2KvDispatcherEpochTestPeer disp(MakeDeps());
  auto req1 = MakeReqRoutingTo(10, 0);
  disp.CaptureEpochReqs(1, {req1});
  disp.CaptureEpochReqs(2, {MakeReqRoutingTo(20, 0)});
  ASSERT_EQ(req1.use_count(), 2);

  // Seal even when BuildDispatchPlans never ran for this epoch.
  disp.OnEpochEnd(/*rank=*/0, /*epoch_id=*/1);
  ASSERT_TRUE(disp.WaitUntilIdleForTest());

  EXPECT_FALSE(disp.HasEpoch(1));
  EXPECT_TRUE(disp.HasEpoch(2));
  EXPECT_EQ(req1.use_count(), 1);
}

TEST(PdV2KvDispatcherEpoch, BuildForUncapturedEpochReturnsEmptyAndTouchesNothing) {
  PdV2KvDispatcherEpochTestPeer disp(MakeDeps());
  disp.CaptureEpochReqs(1, {MakeReqRoutingTo(100, /*device_id=*/0)});

  auto plans = disp.CallBuild(/*rank=*/0, /*epoch_id=*/42);
  EXPECT_TRUE(plans.empty());
}

TEST(PdV2KvDispatcherEpoch, BuildForCapturedEpochIteratesCapturedSet) {
  PdV2KvDispatcherEpochTestPeer disp(MakeDeps());
  disp.CaptureEpochReqs(3, {MakeReqRoutingTo(200, /*device_id=*/5)});

  auto plans = disp.CallBuild(/*rank=*/0, /*epoch_id=*/3);  // rank 0 != device 5
  EXPECT_TRUE(plans.empty());

  disp.OnEpochEnd(/*rank=*/0, /*epoch_id=*/3);
  ASSERT_TRUE(disp.WaitUntilIdleForTest());
  EXPECT_FALSE(disp.HasEpoch(3));
  auto late = disp.CallBuild(/*rank=*/0, /*epoch_id=*/3);
  EXPECT_TRUE(late.empty());
}

}  // namespace pd_v2
}  // namespace ksana_llm
