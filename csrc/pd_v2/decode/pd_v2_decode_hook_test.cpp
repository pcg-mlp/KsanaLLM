/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
// Unit test for PdV2DecodeHook (Decode-side EDS schedule hook). Drives Process
// across its bypass branches (warmup / empty / kDecode / no-residual /
// threshold), the RouteRequest decline + kLocal paths, the empty-blocks
// guard, and the SubmitPrefillRequest success/failure paths — plus the
// OnPrefillComplete trampoline + UpdateAsyncResult drain — against a mock
// PdV2ConnectorInterface. No real connector, no engine.
#include "pd_v2/decode/pd_v2_decode_hook.h"

#include <gtest/gtest.h>

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "batch_scheduler/event_driven_scheduler/schedule_context.h"
#include "runtime/infer_request.h"
#include "utils/request.h"
#include "utils/ret_code.h"

namespace ksana_llm {
namespace pd_v2 {
namespace {

// Mock connector: only the Decode-side methods the hook calls are meaningful.
class MockDecodeConnector : public PdV2ConnectorInterface {
 public:
  std::optional<RouteArm> route_result = RouteArm::kRemotePeer;
  bool submit_ok = true;
  int submit_calls = 0;
  int release_calls = 0;
  PdRequestCompleteCb captured_cb = nullptr;
  void* captured_ctx = nullptr;
  uint64_t last_req_id = 0;

  bool CanAccept(uint64_t, uint32_t) override { return route_result.has_value(); }
  std::optional<RouteArm> RouteRequest(uint64_t, const LocalArmContext&) override { return route_result; }
  Status SubmitPrefillRequest(uint64_t req_id, const std::vector<int>&, const std::vector<int>&,
                              const SamplingConfigWire&, uint32_t, uint32_t, uint32_t, PdRequestCompleteCb on_complete,
                              void* ctx) override {
    ++submit_calls;
    captured_cb = on_complete;
    captured_ctx = ctx;
    last_req_id = req_id;
    return submit_ok ? Status() : Status(RET_RUNTIME_FAILED, "mock submit fail");
  }
  Status Release(uint64_t) override {
    ++release_calls;
    return Status();
  }

  // Unused pure virtuals.
  Status Initialize(const PdV2ConnectorConfig&) override { return Status(); }
  Status Shutdown() override { return Status(); }
  void RegisterIncomingHook(PdIncomingReqCb, void*) override {}
  Status SendPrefillComplete(const std::string&, uint64_t, PdCompleteStatus, const std::vector<int>&, const PeerHealth&,
                             const PdPrefillMetrics&) override {
    return Status();
  }
  Status RegisterKvPool(void*, size_t, size_t, size_t, size_t) override { return Status(); }
  Status GetKvBufferVa(int, uint64_t*, uint64_t*) override { return Status(); }
};

// Keep Requests alive for the InferRequest references.
std::vector<std::shared_ptr<Request>> g_alive;

std::shared_ptr<InferRequest> MakeReq(int64_t req_id, int prefill_token_num, bool with_blocks = true) {
  auto input = std::make_shared<KsanaPythonInput>();
  auto ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  auto request = std::make_shared<Request>(input, ctx);
  g_alive.push_back(request);
  auto req = std::make_shared<InferRequest>(request, 0);
  req->req_id = req_id;
  req->infer_stage = InferStage::kContext;
  req->block_token_num = 16;
  req->attn_dp_group_id = 0;
  req->prefix_cache_len = 0;
  for (int i = 0; i < prefill_token_num; ++i) req->output_tokens.push_back(i + 1);
  req->ResetPrefillingTokens();  // prefilling_tokens_ = output_tokens
  if (with_blocks) req->kv_cache_blocks = {0, 1, 2};
  return req;
}

std::shared_ptr<MockDecodeConnector> MakeConnector() { return std::make_shared<MockDecodeConnector>(); }

PdV2DecodeHook MakeHook(std::shared_ptr<MockDecodeConnector> conn, int threshold = 0) {
  return PdV2DecodeHook(conn, /*hook_idx=*/1, threshold, SelectorPolicy::kRoundRobin, {});
}

}  // namespace

TEST(PdV2DecodeHookTest, EmptyRequestsReturnEmpty) {
  auto conn = MakeConnector();
  auto hook = MakeHook(conn);
  ScheduleContext ctx;
  EXPECT_TRUE(hook.Process({}, ctx).empty());
}

TEST(PdV2DecodeHookTest, WarmupRequestBypassed) {
  auto conn = MakeConnector();
  auto hook = MakeHook(conn);
  ScheduleContext ctx;
  auto req = MakeReq(/*req_id=*/-1, 10);
  auto result = hook.Process({req}, ctx);
  EXPECT_TRUE(result.empty());  // no NextStep entry -> runs locally
  EXPECT_EQ(conn->submit_calls, 0);
}

TEST(PdV2DecodeHookTest, EmptyPrefillingBypassed) {
  auto conn = MakeConnector();
  auto hook = MakeHook(conn);
  ScheduleContext ctx;
  auto req = MakeReq(1, /*prefill_token_num=*/0);
  auto result = hook.Process({req}, ctx);
  EXPECT_TRUE(result.empty());
  EXPECT_EQ(conn->submit_calls, 0);
}

TEST(PdV2DecodeHookTest, DecodeStageBypassed) {
  auto conn = MakeConnector();
  auto hook = MakeHook(conn);
  ScheduleContext ctx;
  auto req = MakeReq(1, 10);
  req->infer_stage = InferStage::kDecode;  // global cache full hit
  auto result = hook.Process({req}, ctx);
  EXPECT_TRUE(result.empty());
  EXPECT_EQ(conn->submit_calls, 0);
}

TEST(PdV2DecodeHookTest, NoResidualBypassed) {
  auto conn = MakeConnector();
  auto hook = MakeHook(conn);
  ScheduleContext ctx;
  auto req = MakeReq(1, 8);
  req->prefix_cache_len = 8;  // remaining = 8 - 8 = 0 -> bypass
  auto result = hook.Process({req}, ctx);
  EXPECT_TRUE(result.empty());
  EXPECT_EQ(conn->submit_calls, 0);
}

TEST(PdV2DecodeHookTest, ThresholdBypassed) {
  auto conn = MakeConnector();
  auto hook = MakeHook(conn, /*threshold=*/100);
  ScheduleContext ctx;
  auto req = MakeReq(1, 10);  // remaining 10 <= 100 -> local
  auto result = hook.Process({req}, ctx);
  EXPECT_TRUE(result.empty());
  EXPECT_EQ(conn->submit_calls, 0);
}

TEST(PdV2DecodeHookTest, RouteDeclinedFailsRequest) {
  auto conn = MakeConnector();
  conn->route_result = std::nullopt;  // no healthy peer
  auto hook = MakeHook(conn);
  ScheduleContext ctx;
  auto req = MakeReq(42, 50);
  auto result = hook.Process({req}, ctx);
  ASSERT_EQ(result.count(42), 1u);
  EXPECT_EQ(result[42].action, NextStepAction::kFailAndEnd);
}

TEST(PdV2DecodeHookTest, RouteLocalRunsLocally) {
  auto conn = MakeConnector();
  conn->route_result = RouteArm::kLocal;
  auto hook = MakeHook(conn);
  ScheduleContext ctx;
  auto req = MakeReq(7, 50);
  auto result = hook.Process({req}, ctx);
  EXPECT_TRUE(result.empty());  // kLocal -> no entry, runs locally
  EXPECT_EQ(conn->submit_calls, 0);
}

TEST(PdV2DecodeHookTest, EmptyKvBlocksFails) {
  auto conn = MakeConnector();
  auto hook = MakeHook(conn);
  ScheduleContext ctx;
  auto req = MakeReq(9, 50, /*with_blocks=*/false);
  auto result = hook.Process({req}, ctx);
  ASSERT_EQ(result.count(9), 1u);
  EXPECT_EQ(result[9].action, NextStepAction::kFailAndEnd);
  EXPECT_EQ(conn->submit_calls, 0);
}

TEST(PdV2DecodeHookTest, SubmitSuccessWaitsAsync) {
  auto conn = MakeConnector();
  auto hook = MakeHook(conn);
  ScheduleContext ctx;
  auto req = MakeReq(11, 50);
  auto result = hook.Process({req}, ctx);
  ASSERT_EQ(result.count(11), 1u);
  EXPECT_EQ(result[11].action, NextStepAction::kWaitAsyncResult);
  EXPECT_EQ(conn->submit_calls, 1);
  EXPECT_TRUE(req->pd_v2.awaiting_prefill);
  ASSERT_NE(conn->captured_cb, nullptr);
}

TEST(PdV2DecodeHookTest, SubmitFailureFailsAndReleases) {
  auto conn = MakeConnector();
  conn->submit_ok = false;
  auto hook = MakeHook(conn);
  ScheduleContext ctx;
  auto req = MakeReq(13, 50);
  auto result = hook.Process({req}, ctx);
  ASSERT_EQ(result.count(13), 1u);
  EXPECT_EQ(result[13].action, NextStepAction::kFailAndEnd);
  EXPECT_EQ(conn->release_calls, 1);
}

namespace {
// Mirror the field resets EventDrivenScheduleProcessor::HandlePdV2DecodeRestart
// applies when a state-C (awaiting_prefill) Decode request is preempted and
// re-submitted through the hook chain. Keeps this unit test independent of the
// schedule processor while pinning the invariant the hook relies on: after a
// restart the pd_v2 flags are pristine.
void SimulatePdV2DecodeRestart(const std::shared_ptr<InferRequest>& req) {
  req->pd_v2.kv_from_remote = false;
  req->pd_v2.awaiting_prefill = false;
  req->pd_v2.pending_stop = false;
  req->infer_stage = InferStage::kContext;
  req->kv_cached_token_num = 0;
}
}  // namespace

// P1-6 regression: a request that reached state-C (submitted to a peer,
// awaiting_prefill=true), then got preempted + restarted (flags cleared), and
// on its re-run through the hook now qualifies for the short-request bypass
// (prefix cache covers the whole prompt). The bypass branch must NOT re-arm
// awaiting_prefill — otherwise the backend's RDMA-flight invariant check would
// fire on a request that has no in-flight peer Prefill.
TEST(PdV2DecodeHookTest, BypassAfterRestartKeepsAwaitingPrefillFalse) {
  auto conn = MakeConnector();
  auto hook = MakeHook(conn);
  ScheduleContext ctx;
  auto req = MakeReq(60, 50);

  // 1. First pass: forwarded to a peer, parked in state-C.
  auto first = hook.Process({req}, ctx);
  ASSERT_EQ(first.count(60), 1u);
  EXPECT_EQ(first[60].action, NextStepAction::kWaitAsyncResult);
  EXPECT_TRUE(req->pd_v2.awaiting_prefill);
  EXPECT_EQ(conn->submit_calls, 1);

  // 2. Preempt + restart clears the pd_v2 flags (as the schedule processor does).
  SimulatePdV2DecodeRestart(req);
  // 3. On re-run the Decode-side prefix cache now covers the whole prompt, so
  //    the no-residual fast path bypasses the peer round-trip.
  req->prefix_cache_len = 50;  // remaining = 50 - 50 = 0 -> bypass

  auto second = hook.Process({req}, ctx);
  EXPECT_TRUE(second.empty());                // bypass -> no NextStep entry, runs locally
  EXPECT_EQ(conn->submit_calls, 1);           // no second SubmitPrefillRequest
  EXPECT_FALSE(req->pd_v2.awaiting_prefill);  // bypass path must not re-arm the flag
}

// P1-6 companion: an abort/cancel arrived while the request sat in state-C
// (pending_stop=true), then the restart cleared it. A subsequent bypass must
// leave both pending_stop and awaiting_prefill false — the bypass path is a
// clean local run with no peer state to unwind.
TEST(PdV2DecodeHookTest, BypassAfterRestartWithPriorPendingStopStaysClean) {
  auto conn = MakeConnector();
  auto hook = MakeHook(conn);
  ScheduleContext ctx;
  auto req = MakeReq(61, 50);

  hook.Process({req}, ctx);
  EXPECT_TRUE(req->pd_v2.awaiting_prefill);
  req->pd_v2.pending_stop = true;  // user cancel / EDS timeout while waiting on the peer

  SimulatePdV2DecodeRestart(req);
  req->prefix_cache_len = 50;  // force the no-residual bypass on the re-run

  auto result = hook.Process({req}, ctx);
  EXPECT_TRUE(result.empty());
  EXPECT_EQ(conn->submit_calls, 1);
  EXPECT_FALSE(req->pd_v2.awaiting_prefill);
  EXPECT_FALSE(req->pd_v2.pending_stop);
}

TEST(PdV2DecodeHookTest, OnPrefillCompleteStashesResult) {
  auto conn = MakeConnector();
  auto hook = MakeHook(conn);
  hook.SetAsyncTaskFinishedCallback([](const EventAsyncTaskFinishedInfo&) {});
  ScheduleContext ctx;
  auto req = MakeReq(21, 50);
  hook.Process({req}, ctx);
  ASSERT_NE(conn->captured_cb, nullptr);
  EXPECT_EQ(hook.PendingCountForTest(), 0u);

  // Fire the completion trampoline (as the connector dispatch thread would).
  PdPrefillMetrics metrics;
  conn->captured_cb(21, PdCompleteStatus::kOk, {555, 556}, metrics, "peer:1", conn->captured_ctx);
  EXPECT_EQ(hook.PendingCountForTest(), 1u);

  // UpdateAsyncResult drains; with no ScheduleState the req isn't resolvable,
  // but the drain path is exercised.
  hook.UpdateAsyncResult({}, ctx);
}

// ---- UpdateAsyncResult splice path: req IS resolvable via ctx.id_to_req ----

namespace {
// Submit a req through Process, register it in the schedule state so
// UpdateAsyncResult can resolve it via GetReqById, then stash a completion via
// the captured trampoline. `state` must outlive the subsequent UpdateAsyncResult
// call (GetReqById reads schedule_state_->req_id_to_req_ptr).
void SubmitAndComplete(PdV2DecodeHook& hook, std::shared_ptr<MockDecodeConnector> conn, ScheduleContext& ctx,
                       ScheduleState& state, const std::shared_ptr<InferRequest>& req, PdCompleteStatus status,
                       const std::vector<int>& tokens, const std::string& peer = "peer:1") {
  ctx.SetScheduleState(&state);
  state.req_id_to_req_ptr[req->req_id] = req;
  hook.Process({req}, ctx);
  ASSERT_NE(conn->captured_cb, nullptr);
  PdPrefillMetrics metrics;
  conn->captured_cb(static_cast<uint64_t>(req->req_id), status, tokens, metrics, peer, conn->captured_ctx);
}
}  // namespace

TEST(PdV2DecodeHookTest, UpdateAsyncResultSplicesSingleToken) {
  auto conn = MakeConnector();
  auto hook = MakeHook(conn);
  ScheduleContext ctx;
  ScheduleState state;
  auto req = MakeReq(50, 10);
  SubmitAndComplete(hook, conn, ctx, state, req, PdCompleteStatus::kOk, {999});

  auto result = hook.UpdateAsyncResult({}, ctx);
  EXPECT_FALSE(req->pd_v2.awaiting_prefill);
  EXPECT_EQ(req->infer_stage, InferStage::kDecode);
  EXPECT_TRUE(req->pd_v2.kv_from_remote);
  ASSERT_FALSE(req->output_tokens.empty());
  EXPECT_EQ(req->output_tokens.back(), 999);
  EXPECT_EQ(req->forwarding_tokens, req->GetPrefillingTokens());
  ASSERT_EQ(result.count(50), 1u);
  EXPECT_EQ(result[50].action, NextStepAction::kContinue);
}

TEST(PdV2DecodeHookTest, UpdateAsyncResultSplicesMtpDrafts) {
  auto conn = MakeConnector();
  auto hook = MakeHook(conn);
  ScheduleContext ctx;
  ScheduleState state;
  auto req = MakeReq(51, 10);
  SubmitAndComplete(hook, conn, ctx, state, req, PdCompleteStatus::kOk, {999, 1001, 1002});

  hook.UpdateAsyncResult({}, ctx);
  EXPECT_EQ(req->output_tokens.back(), 999);  // only the main token is appended to output
  EXPECT_EQ(req->draft_tokens.mtp, (std::vector<int>{1001, 1002}));
}

TEST(PdV2DecodeHookTest, UpdateAsyncResultRecordsPeerInReqCtx) {
  auto conn = MakeConnector();
  auto hook = MakeHook(conn);
  ScheduleContext ctx;
  ScheduleState state;
  auto req = MakeReq(52, 10);
  SubmitAndComplete(hook, conn, ctx, state, req, PdCompleteStatus::kOk, {999}, "prefill-node-7");

  hook.UpdateAsyncResult({}, ctx);
  ASSERT_NE(req->req_ctx, nullptr);
  EXPECT_EQ((*req->req_ctx)["pd_v2_prefill_peer"], "prefill-node-7");
}

TEST(PdV2DecodeHookTest, UpdateAsyncResultFailedStatusFails) {
  auto conn = MakeConnector();
  auto hook = MakeHook(conn);
  ScheduleContext ctx;
  ScheduleState state;
  auto req = MakeReq(53, 10);
  SubmitAndComplete(hook, conn, ctx, state, req, PdCompleteStatus::kFailed, {999});

  auto result = hook.UpdateAsyncResult({}, ctx);
  ASSERT_EQ(result.count(53), 1u);
  EXPECT_EQ(result[53].action, NextStepAction::kFailAndEnd);
  EXPECT_FALSE(req->pd_v2.awaiting_prefill);
}

TEST(PdV2DecodeHookTest, UpdateAsyncResultEmptyTokensFails) {
  auto conn = MakeConnector();
  auto hook = MakeHook(conn);
  ScheduleContext ctx;
  ScheduleState state;
  auto req = MakeReq(54, 10);
  SubmitAndComplete(hook, conn, ctx, state, req, PdCompleteStatus::kOk, {});  // OK but no tokens

  auto result = hook.UpdateAsyncResult({}, ctx);
  ASSERT_EQ(result.count(54), 1u);
  EXPECT_EQ(result[54].action, NextStepAction::kFailAndEnd);
}

TEST(PdV2DecodeHookTest, UpdateAsyncResultPendingStopFails) {
  auto conn = MakeConnector();
  auto hook = MakeHook(conn);
  ScheduleContext ctx;
  ScheduleState state;
  auto req = MakeReq(55, 10);
  SubmitAndComplete(hook, conn, ctx, state, req, PdCompleteStatus::kOk, {999});
  req->pd_v2.pending_stop = true;  // user cancel / EDS timeout while waiting

  auto result = hook.UpdateAsyncResult({}, ctx);
  ASSERT_EQ(result.count(55), 1u);
  EXPECT_EQ(result[55].action, NextStepAction::kFailAndEnd);
  EXPECT_FALSE(req->pd_v2.pending_stop);  // reset by the terminal path
}

// ---- OnPrefillComplete trampoline edge cases ----

TEST(PdV2DecodeHookTest, OnPrefillCompleteNullCtxIsSafe) {
  PdPrefillMetrics metrics;
  // ctx == nullptr: logs an error and returns without dereferencing.
  PdV2DecodeHook::OnPrefillComplete(1, PdCompleteStatus::kOk, {1}, metrics, "peer", nullptr);
  SUCCEED();
}

TEST(PdV2DecodeHookTest, OnPrefillCompleteWithoutCallbackStillStashes) {
  auto conn = MakeConnector();
  auto hook = MakeHook(conn);  // no SetAsyncTaskFinishedCallback -> async_callback_ is null
  ScheduleContext ctx;
  auto req = MakeReq(56, 10);
  hook.Process({req}, ctx);
  ASSERT_NE(conn->captured_cb, nullptr);
  PdPrefillMetrics metrics;
  conn->captured_cb(56, PdCompleteStatus::kOk, {999}, metrics, "peer:1", conn->captured_ctx);
  EXPECT_EQ(hook.PendingCountForTest(), 1u);  // stashed even with no wake callback
}

// ---- MakeWireSamplingConfig (private static; reached via -fno-access-control) ----

TEST(PdV2DecodeHookTest, MakeWireSamplingConfigPtpDrafts) {
  SamplingConfig sc;
  sc.topk = 5;
  sc.topp = 0.9f;
  sc.temperature = 0.7f;
  sc.repetition_penalty = 1.1f;
  sc.ptp_token_num = 3;
  sc.stop_token_ids = {2, 3};
  auto w = PdV2DecodeHook::MakeWireSamplingConfig(sc);
  EXPECT_EQ(w.max_new_tokens, 4);  // 1 main + 3 drafts
  EXPECT_EQ(w.topk, 5);
  EXPECT_FLOAT_EQ(w.topp, 0.9f);
  EXPECT_EQ(w.ptp_token_num, 3);
  EXPECT_EQ(w.stop_token_ids, (std::vector<int32_t>{2, 3}));
}

TEST(PdV2DecodeHookTest, MakeWireSamplingConfigPtpUnsetCollapsesToOne) {
  SamplingConfig sc;
  sc.ptp_token_num = -1;  // "use global default" -> collapse to 0 drafts on the wire
  auto w = PdV2DecodeHook::MakeWireSamplingConfig(sc);
  EXPECT_EQ(w.max_new_tokens, 1);  // 1 + max(0, -1)
  EXPECT_EQ(w.ptp_token_num, -1);
}

// ---- Release wiring on request destruction ----

TEST(PdV2DecodeHookTest, SubmitRegistersReleaseOnDestroy) {
  auto conn = MakeConnector();
  auto hook = MakeHook(conn);
  ScheduleContext ctx;
  auto req = MakeReq(60, 50);
  hook.Process({req}, ctx);
  EXPECT_EQ(conn->submit_calls, 1);
  EXPECT_EQ(conn->release_calls, 0);
  // Dropping the InferRequest fires the AddOnDestroy hook -> connector->Release.
  req.reset();
  EXPECT_EQ(conn->release_calls, 1);
}

}  // namespace pd_v2
}  // namespace ksana_llm
