/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
// Unit test for PdV2PrefillCompletion. Drives the two completion gates
// (barrier report + sampler/token ready) against a fake PdV2ConnectorInterface
// and locally constructed Request objects. Relies on -fno-access-control
// (set by the cpp_test cmake macro) to reach the private static helpers
// DoSendPrefillComplete / TryCompleteUnlocked / OnRequestFinished directly.
#include "endpoints/pd_v2/pd_v2_prefill_completion.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "pd_v2/public/pd_v2_connector.h"
#include "pd_v2/public/pd_v2_types.h"
#include "utils/request.h"
#include "utils/waiter.h"

namespace ksana_llm {
namespace pd_v2 {
namespace {

// Records every SendPrefillComplete so tests can assert the completion
// chokepoint fired with the expected status/tokens. All other interface
// methods are no-op (engine/executor surfaces unused by the completion path).
class FakeConnector : public PdV2ConnectorInterface {
 public:
  Status Initialize(const PdV2ConnectorConfig&) override { return Status(); }
  Status Shutdown() override { return Status(); }
  Status RegisterKvPool(void*, size_t, size_t, size_t, size_t) override { return Status(); }
  Status GetKvBufferVa(int, uint64_t*, uint64_t*) override { return Status(); }
  bool CanAccept(uint64_t, uint32_t) override { return true; }
  Status SubmitPrefillRequest(uint64_t, const std::vector<int>&, const std::vector<int>&, const SamplingConfigWire&,
                              uint32_t, uint32_t, uint32_t, PdRequestCompleteCb, void*) override {
    return Status();
  }
  Status Release(uint64_t req_id) override {
    ++release_calls;
    last_release_req_id = req_id;
    return Status();
  }
  void RegisterIncomingHook(PdIncomingReqCb, void*) override {}
  Status SendPrefillComplete(const std::string& peer_addr, uint64_t req_id, PdCompleteStatus status,
                             const std::vector<int>& generated_tokens, const PeerHealth& snapshot,
                             const PdPrefillMetrics& prefill_metrics) override {
    ++send_calls;
    last_peer_addr = peer_addr;
    last_req_id = req_id;
    last_status = status;
    last_tokens = generated_tokens;
    last_health = snapshot;
    last_metrics = prefill_metrics;
    return send_status;
  }

  int send_calls = 0;
  int release_calls = 0;
  uint64_t last_release_req_id = 0;
  std::string last_peer_addr;
  uint64_t last_req_id = 0;
  PdCompleteStatus last_status = PdCompleteStatus::kOk;
  std::vector<int> last_tokens;
  PeerHealth last_health;
  PdPrefillMetrics last_metrics;
  Status send_status = Status();
};

std::shared_ptr<Request> MakeRequest(int64_t req_id, std::vector<int> input_tokens) {
  return std::make_shared<Request>(req_id, std::move(input_tokens), SamplingConfig{});
}

// Append generated tokens past the prompt so OnRequestFinished extracts a
// non-empty generated-token suffix (output_tokens references output_group[0],
// initialized to the input tokens by the Request ctor).
void AppendGenerated(const std::shared_ptr<Request>& req, const std::vector<int>& gen) {
  for (int t : gen) req->output_tokens.push_back(t);
}

PdV2PrefillCompletion::Deps MakeDeps(std::shared_ptr<FakeConnector> conn,
                                     std::function<PeerHealth()> snapshot_fn = {}) {
  PdV2PrefillCompletion::Deps deps;
  deps.connector = std::move(conn);
  deps.snapshot_fn = std::move(snapshot_fn);
  return deps;
}

}  // namespace

class PdV2PrefillCompletionTest : public ::testing::Test {
 protected:
  std::shared_ptr<FakeConnector> conn_ = std::make_shared<FakeConnector>();
};

// Admit bookkeeping: alive + admitted counters move, completed stays 0.
TEST_F(PdV2PrefillCompletionTest, AdmitTracksState) {
  PdV2PrefillCompletion c(MakeDeps(conn_));
  auto req = MakeRequest(1, {10, 11, 12});
  c.Admit(1, req, "decode:1", /*expected_barrier_reports=*/1);
  EXPECT_EQ(c.NumAdmittedForTest(), 1u);
  EXPECT_EQ(c.NumAliveForTest(), 1u);
  EXPECT_EQ(c.NumCompletedForTest(), 0u);
}

// OnBarrierDone for a req_id never admitted: warns and returns, no state change.
TEST_F(PdV2PrefillCompletionTest, OnBarrierDoneUnknownReqIsNoop) {
  PdV2PrefillCompletion c(MakeDeps(conn_));
  c.OnBarrierDone(999, 0);
  EXPECT_EQ(c.NumAliveForTest(), 0u);
  EXPECT_EQ(conn_->send_calls, 0);
}

// expected_barrier_reports=2: first report does not complete the barrier gate;
// tokens gate still open so no send.
TEST_F(PdV2PrefillCompletionTest, BarrierPartialDoesNotComplete) {
  PdV2PrefillCompletion c(MakeDeps(conn_));
  auto req = MakeRequest(2, {1, 2});
  c.Admit(2, req, "decode:2", /*expected_barrier_reports=*/2);
  c.OnBarrierDone(2, 0);
  EXPECT_EQ(conn_->send_calls, 0);
  EXPECT_EQ(c.NumAliveForTest(), 1u);
  // Second report closes the barrier gate, but tokens gate still open.
  c.OnBarrierDone(2, 0);
  EXPECT_EQ(conn_->send_calls, 0);
  EXPECT_EQ(c.NumAliveForTest(), 1u);
}

// Non-zero barrier status marks the aggregated barrier_status as failed.
TEST_F(PdV2PrefillCompletionTest, BarrierFailStatusAggregates) {
  PdV2PrefillCompletion c(MakeDeps(conn_));
  auto req = MakeRequest(3, {1, 2, 3});
  c.Admit(3, req, "decode:3", 1);
  c.OnBarrierDone(3, /*status=*/1);  // barrier_done, but tokens not ready -> no send
  EXPECT_EQ(conn_->send_calls, 0);
  // Now fire tokens gate (OK output) -> completes, but final status is kFailed
  // because barrier_status aggregated a failure.
  AppendGenerated(req, {77});
  auto cb = c.MakeRequestFinishedCallback(3);
  cb();
  ASSERT_EQ(conn_->send_calls, 1);
  EXPECT_EQ(conn_->last_status, PdCompleteStatus::kFailed);
  EXPECT_EQ(c.NumCompletedForTest(), 1u);
  EXPECT_EQ(c.NumAliveForTest(), 0u);
}

// Barrier-first then tokens: send fires on the tokens-ready callback.
TEST_F(PdV2PrefillCompletionTest, BarrierFirstThenTokensCompletes) {
  PdV2PrefillCompletion c(MakeDeps(conn_));
  auto req = MakeRequest(4, {5, 6, 7});
  AppendGenerated(req, {100, 101});
  c.Admit(4, req, "decode:4", 1);
  c.OnBarrierDone(4, 0);
  EXPECT_EQ(conn_->send_calls, 0);  // tokens gate still open
  auto cb = c.MakeRequestFinishedCallback(4);
  cb();
  ASSERT_EQ(conn_->send_calls, 1);
  EXPECT_EQ(conn_->last_status, PdCompleteStatus::kOk);
  EXPECT_EQ(conn_->last_tokens, (std::vector<int>{100, 101}));
  EXPECT_EQ(conn_->last_peer_addr, "decode:4");
  EXPECT_EQ(conn_->last_req_id, 4u);
}

// Tokens-first then barrier: send fires on the barrier-done call.
TEST_F(PdV2PrefillCompletionTest, TokensFirstThenBarrierCompletes) {
  PdV2PrefillCompletion c(MakeDeps(conn_));
  auto req = MakeRequest(5, {9});
  AppendGenerated(req, {200});
  c.Admit(5, req, "decode:5", 1);
  auto cb = c.MakeRequestFinishedCallback(5);
  cb();
  EXPECT_EQ(conn_->send_calls, 0);  // barrier gate still open
  c.OnBarrierDone(5, 0);
  ASSERT_EQ(conn_->send_calls, 1);
  EXPECT_EQ(conn_->last_status, PdCompleteStatus::kOk);
  EXPECT_EQ(conn_->last_tokens, (std::vector<int>{200}));
}

// finish_status not OK: tokens gate AND barrier gate are both force-set, so a
// single callback completes with kFailed.
TEST_F(PdV2PrefillCompletionTest, RequestFinishedWithFailedStatus) {
  PdV2PrefillCompletion c(MakeDeps(conn_));
  auto req = MakeRequest(6, {1, 2, 3});
  req->finish_status = Status(RET_RUNTIME_FAILED, "boom");
  c.Admit(6, req, "decode:6", /*expected_barrier_reports=*/3);  // barrier never reported
  auto cb = c.MakeRequestFinishedCallback(6);
  cb();
  ASSERT_EQ(conn_->send_calls, 1);
  EXPECT_EQ(conn_->last_status, PdCompleteStatus::kFailed);
  EXPECT_EQ(c.NumCompletedForTest(), 1u);
}

// finish_status OK but output_tokens == input (no generated suffix): token gate
// flags kFailed (empty tokens), completes kFailed once barrier closes.
TEST_F(PdV2PrefillCompletionTest, RequestFinishedEmptyOutputIsFailed) {
  PdV2PrefillCompletion c(MakeDeps(conn_));
  auto req = MakeRequest(7, {1, 2, 3});  // no AppendGenerated
  c.Admit(7, req, "decode:7", 1);
  c.OnBarrierDone(7, 0);
  auto cb = c.MakeRequestFinishedCallback(7);
  cb();
  ASSERT_EQ(conn_->send_calls, 1);
  EXPECT_EQ(conn_->last_status, PdCompleteStatus::kFailed);
  EXPECT_TRUE(conn_->last_tokens.empty());
}

// Callback after the completion object is destroyed: weak_ptr lock fails, no
// crash, no send.
TEST_F(PdV2PrefillCompletionTest, CallbackAfterDestroyIsSafe) {
  std::function<void()> cb;
  {
    auto c = std::make_shared<PdV2PrefillCompletion>(MakeDeps(conn_));
    auto req = MakeRequest(8, {1});
    AppendGenerated(req, {2});
    c->Admit(8, req, "decode:8", 1);
    cb = c->MakeRequestFinishedCallback(8);
  }  // c destroyed -> SharedState weak_ptr expires
  cb();  // must not crash
  EXPECT_EQ(conn_->send_calls, 0);
}

// Callback for a req_id not present in the alive table: warns and drops.
TEST_F(PdV2PrefillCompletionTest, CallbackForUnknownReqDropped) {
  PdV2PrefillCompletion c(MakeDeps(conn_));
  auto req = MakeRequest(9, {1, 2});
  c.Admit(9, req, "decode:9", 1);
  auto cb = c.MakeRequestFinishedCallback(/*req_id=*/4242);  // different id
  cb();
  EXPECT_EQ(conn_->send_calls, 0);
  EXPECT_EQ(c.NumAliveForTest(), 1u);  // req 9 still alive
}

// snapshot_fn provided: DoSendPrefillComplete fills health from it.
TEST_F(PdV2PrefillCompletionTest, SnapshotFnSuppliesHealth) {
  bool called = false;
  auto snapshot = [&called]() {
    called = true;
    PeerHealth h;
    h.healthy = true;
    h.free_kv_blocks = 123;
    return h;
  };
  PdV2PrefillCompletion c(MakeDeps(conn_, snapshot));
  auto req = MakeRequest(10, {1});
  AppendGenerated(req, {2});
  c.Admit(10, req, "decode:10", 1);
  c.OnBarrierDone(10, 0);
  c.MakeRequestFinishedCallback(10)();
  ASSERT_EQ(conn_->send_calls, 1);
  EXPECT_TRUE(called);
  EXPECT_EQ(conn_->last_health.free_kv_blocks, 123u);
}

// SendPrefillComplete returning an error hits the warning branch (no throw).
TEST_F(PdV2PrefillCompletionTest, SendFailureIsLogged) {
  conn_->send_status = Status(RET_RUNTIME_FAILED, "send failed");
  PdV2PrefillCompletion c(MakeDeps(conn_));
  auto req = MakeRequest(11, {1});
  AppendGenerated(req, {2});
  c.Admit(11, req, "decode:11", 1);
  c.OnBarrierDone(11, 0);
  c.MakeRequestFinishedCallback(11)();
  EXPECT_EQ(conn_->send_calls, 1);
  EXPECT_EQ(c.NumCompletedForTest(), 1u);
}

// ---- Direct calls into private static helpers (via -fno-access-control) ----

// DoSendPrefillComplete fallback: status kOk + empty tokens + non-null request
// pulls tokens from request->output_tokens.
TEST_F(PdV2PrefillCompletionTest, DoSendFallbackPullsOutputTokens) {
  auto req = MakeRequest(12, {1, 2});
  req->output_tokens.push_back(3);  // output_tokens = {1,2,3}
  PdV2PrefillCompletion::DoSendPrefillComplete(conn_, /*snapshot_fn=*/{}, req, "decode:12", PdCompleteStatus::kOk,
                                               /*tokens=*/{}, 12);
  ASSERT_EQ(conn_->send_calls, 1);
  EXPECT_EQ(conn_->last_tokens, (std::vector<int>{1, 2, 3}));
  // No snapshot_fn + kOk -> health.healthy derived true.
  EXPECT_TRUE(conn_->last_health.healthy);
}

// DoSendPrefillComplete with a null connector must not crash and not send.
TEST_F(PdV2PrefillCompletionTest, DoSendNullConnectorIsSafe) {
  auto req = MakeRequest(13, {1});
  PdV2PrefillCompletion::DoSendPrefillComplete(/*connector=*/nullptr, /*snapshot_fn=*/{}, req, "decode:13",
                                               PdCompleteStatus::kOk, {9}, 13);
  EXPECT_EQ(conn_->send_calls, 0);
}

// DoSendPrefillComplete with a null request + failed status: no fallback, health
// derived from status (unhealthy), still sends.
TEST_F(PdV2PrefillCompletionTest, DoSendNullRequestFailedStatus) {
  PdV2PrefillCompletion::DoSendPrefillComplete(conn_, /*snapshot_fn=*/{}, /*request=*/nullptr, "decode:14",
                                               PdCompleteStatus::kFailed, {}, 14);
  ASSERT_EQ(conn_->send_calls, 1);
  EXPECT_EQ(conn_->last_status, PdCompleteStatus::kFailed);
  EXPECT_FALSE(conn_->last_health.healthy);
}

}  // namespace pd_v2
}  // namespace ksana_llm
