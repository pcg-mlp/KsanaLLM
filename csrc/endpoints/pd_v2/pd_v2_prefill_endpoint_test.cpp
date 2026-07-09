/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
// Unit test for PdV2PrefillEndpoint. Drives OnIncoming / Register /
// OnBarrierDone against a fake PdV2ConnectorInterface and a real (CPU-only)
// LocalEndpoint whose SubmitAsync merely enqueues onto a Channel the test
// drains. -fno-access-control (cpp_test cmake macro) exposes deps_ for setup.
#include "endpoints/pd_v2/pd_v2_prefill_endpoint.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "configure/environment.h"
#include "endpoints/local/local_endpoint.h"
#include "pd_v2/public/pd_v2_connector.h"
#include "pd_v2/public/pd_v2_types.h"
#include "pd_v2/wire/prefill_request_wire.h"
#include "pd_v2/wire/sampling_config_wire.h"
#include "utils/channel.h"
#include "utils/request.h"
#include "utils/status.h"

namespace ksana_llm {
namespace pd_v2 {
namespace {

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
  void RegisterIncomingHook(PdIncomingReqCb on_incoming, void* ctx) override {
    ++register_hook_calls;
    hook = on_incoming;
    hook_ctx = ctx;
  }
  Status SendPrefillComplete(const std::string& peer_addr, uint64_t req_id, PdCompleteStatus status,
                             const std::vector<int>& generated_tokens, const PeerHealth& snapshot,
                             const PdPrefillMetrics&) override {
    ++send_calls;
    last_peer_addr = peer_addr;
    last_req_id = req_id;
    last_status = status;
    last_tokens = generated_tokens;
    last_health = snapshot;
    return Status();
  }

  int register_hook_calls = 0;
  PdIncomingReqCb hook = nullptr;
  void* hook_ctx = nullptr;
  int send_calls = 0;
  int release_calls = 0;
  uint64_t last_release_req_id = 0;
  std::string last_peer_addr;
  uint64_t last_req_id = 0;
  PdCompleteStatus last_status = PdCompleteStatus::kOk;
  std::vector<int> last_tokens;
  PeerHealth last_health;
};

using RequestChannel = Channel<std::pair<Status, std::shared_ptr<Request>>>;

SamplingConfigWire MakeWire(int32_t max_new_tokens = 4) {
  SamplingConfigWire w;
  w.topk = 5;
  w.topp = 0.9f;
  w.temperature = 0.7f;
  w.max_new_tokens = max_new_tokens;
  w.stop_token_ids = {2, 3};
  w.ptp_token_num = 0;
  return w;
}

}  // namespace

class PdV2PrefillEndpointTest : public ::testing::Test {
 protected:
  void SetUp() override {
    conn_ = std::make_shared<FakeConnector>();
    EndpointConfig cfg;
    local_endpoint_ = std::make_shared<LocalEndpoint>(cfg, channel_);
  }

  // Build an endpoint with real local_endpoint + fake connector.
  std::unique_ptr<PdV2PrefillEndpoint> MakeEndpoint(uint32_t prefill_attn_tp = 0,
                                                    std::function<PeerHealth()> snapshot_fn = {}) {
    PdV2PrefillEndpoint::Deps deps;
    deps.local_endpoint = local_endpoint_;
    deps.connector = conn_;
    deps.snapshot_fn = std::move(snapshot_fn);
    deps.prefill_attn_tp = prefill_attn_tp;
    return std::make_unique<PdV2PrefillEndpoint>(std::move(deps));
  }

  // Non-blocking drain of one submitted request, or nullptr if channel empty.
  // Channel::Read blocks until n elements are available, so guard on Size()
  // first (safe: the test is single-threaded and OnIncoming enqueues inline).
  std::shared_ptr<Request> TryReadOne() {
    if (channel_.Size() == 0) return nullptr;
    std::pair<Status, std::shared_ptr<Request>> item;
    if (channel_.Read(&item, 1) != 1) return nullptr;
    return item.second;
  }

  RequestChannel channel_;
  std::shared_ptr<FakeConnector> conn_;
  std::shared_ptr<LocalEndpoint> local_endpoint_;
};

// Register forwards to the connector's RegisterIncomingHook with `this` as ctx.
TEST_F(PdV2PrefillEndpointTest, RegisterInstallsHook) {
  auto ep = MakeEndpoint();
  ep->Register();
  EXPECT_EQ(conn_->register_hook_calls, 1);
  EXPECT_EQ(conn_->hook_ctx, ep.get());
  EXPECT_EQ(conn_->hook, &PdV2PrefillEndpoint::OnIncomingTrampoline);
}

// Register with a null connector returns early (no crash).
TEST_F(PdV2PrefillEndpointTest, RegisterWithNullConnectorIsNoop) {
  PdV2PrefillEndpoint::Deps deps;
  deps.local_endpoint = local_endpoint_;
  deps.connector = nullptr;  // logs error in ctor
  PdV2PrefillEndpoint ep(std::move(deps));
  ep.Register();  // returns immediately
  SUCCEED();
}

// Trampoline with null ctx is a no-op; with a valid ctx it dispatches OnIncoming.
TEST_F(PdV2PrefillEndpointTest, TrampolineNullCtxIsNoop) {
  IncomingPrefillRequest req;
  req.req_id = 1;
  PdV2PrefillEndpoint::OnIncomingTrampoline(req, nullptr);  // must not crash
  SUCCEED();
}

TEST_F(PdV2PrefillEndpointTest, TrampolineDispatchesToOnIncoming) {
  auto ep = MakeEndpoint();
  auto wire = MakeWire();
  IncomingPrefillRequest req;
  req.req_id = 42;
  req.decode_engine_inference_addr = "decode:42";
  req.input_tokens = {1, 2, 3};
  req.sampling_config_wire = &wire;
  PdV2PrefillEndpoint::OnIncomingTrampoline(req, ep.get());
  EXPECT_EQ(ep->NumIncomingAdmittedForTest(), 1u);
  auto submitted = TryReadOne();
  ASSERT_NE(submitted, nullptr);
  EXPECT_EQ(submitted->req_id, 42);
}

// OnIncoming on an endpoint missing deps returns without admitting.
TEST_F(PdV2PrefillEndpointTest, OnIncomingMissingDepsDrops) {
  PdV2PrefillEndpoint::Deps deps;
  deps.local_endpoint = nullptr;  // missing
  deps.connector = conn_;
  PdV2PrefillEndpoint ep(std::move(deps));
  auto wire = MakeWire();
  IncomingPrefillRequest req;
  req.req_id = 7;
  req.sampling_config_wire = &wire;
  ep.OnIncoming(req);
  EXPECT_EQ(ep.NumIncomingAdmittedForTest(), 0u);
}

// Unsupported topology (num_decode_targets > prefill_attn_tp): reject with
// kFailed via SendPrefillComplete; snapshot_fn absent -> health defaulted true.
TEST_F(PdV2PrefillEndpointTest, OnIncomingTopologyRejectNoSnapshot) {
  auto ep = MakeEndpoint(/*prefill_attn_tp=*/1);
  auto wire = MakeWire();
  IncomingPrefillRequest req;
  req.req_id = 8;
  req.decode_engine_inference_addr = "decode:8";
  req.input_tokens = {1};
  req.sampling_config_wire = &wire;
  req.num_decode_targets = 2;  // > prefill_attn_tp(1)
  ep->OnIncoming(req);
  ASSERT_EQ(conn_->send_calls, 1);
  EXPECT_EQ(conn_->last_status, PdCompleteStatus::kFailed);
  EXPECT_EQ(conn_->last_req_id, 8u);
  EXPECT_TRUE(conn_->last_health.healthy);  // defaulted
  EXPECT_EQ(ep->NumIncomingAdmittedForTest(), 0u);
}

// Same reject path, but snapshot_fn supplies the health snapshot.
TEST_F(PdV2PrefillEndpointTest, OnIncomingTopologyRejectWithSnapshot) {
  bool called = false;
  auto snapshot = [&called]() {
    called = true;
    PeerHealth h;
    h.healthy = false;
    h.free_kv_blocks = 9;
    return h;
  };
  auto ep = MakeEndpoint(/*prefill_attn_tp=*/2, snapshot);
  auto wire = MakeWire();
  IncomingPrefillRequest req;
  req.req_id = 9;
  req.sampling_config_wire = &wire;
  req.num_decode_targets = 5;  // > 2
  ep->OnIncoming(req);
  ASSERT_EQ(conn_->send_calls, 1);
  EXPECT_TRUE(called);
  EXPECT_EQ(conn_->last_health.free_kv_blocks, 9u);
}

// Null sampling_config_wire: drop without admitting.
TEST_F(PdV2PrefillEndpointTest, OnIncomingNullWireDrops) {
  auto ep = MakeEndpoint();
  IncomingPrefillRequest req;
  req.req_id = 10;
  req.sampling_config_wire = nullptr;
  ep->OnIncoming(req);
  EXPECT_EQ(ep->NumIncomingAdmittedForTest(), 0u);
  EXPECT_EQ(TryReadOne(), nullptr);
}

// Happy path with no decode targets: request admitted + submitted to channel,
// sampling config translated, pd_v2 fields populated.
TEST_F(PdV2PrefillEndpointTest, OnIncomingAdmitsAndSubmits) {
  auto ep = MakeEndpoint();
  auto wire = MakeWire(/*max_new_tokens=*/4);
  IncomingPrefillRequest req;
  req.req_id = 11;
  req.decode_engine_inference_addr = "decode:11";
  req.input_tokens = {100, 101, 102};
  req.sampling_config_wire = &wire;
  req.num_decode_targets = 0;
  req.decode_attn_dp_group_id = 3;
  ep->OnIncoming(req);

  EXPECT_EQ(ep->NumIncomingAdmittedForTest(), 1u);
  EXPECT_EQ(ep->NumAliveForTest(), 1u);
  auto submitted = TryReadOne();
  ASSERT_NE(submitted, nullptr);
  EXPECT_EQ(submitted->req_id, 11);
  EXPECT_EQ(submitted->input_tokens, (std::vector<int>{100, 101, 102}));
  EXPECT_TRUE(submitted->is_pd_v2_prefill);
  EXPECT_EQ(submitted->pd_v2_decode_peer_addr, "decode:11");
  EXPECT_EQ(submitted->pd_v2_decode_attn_dp_group_id, 3u);
  EXPECT_EQ(submitted->sampling_config.topk, 5);
  EXPECT_EQ(submitted->sampling_config.max_new_tokens, 4);
  EXPECT_EQ(submitted->sampling_config.stop_token_ids, (std::vector<int>{2, 3}));
  EXPECT_TRUE(static_cast<bool>(submitted->waiter));
}

// Happy path WITH decode targets: targets parsed into pd_v2_decode_targets, and
// the request's pd_v2_on_destroy invokes connector->Release when destroyed.
TEST_F(PdV2PrefillEndpointTest, OnIncomingParsesDecodeTargetsAndReleaseOnDestroy) {
  auto ep = MakeEndpoint();
  auto wire = MakeWire();

  std::vector<DecodeTarget> targets(2);
  targets[0].decode_attn_tp_rank_in_group = 0;
  targets[0].kv_pool_base_va = 0x1000;
  targets[0].kv_block_ids = {1, 2};
  targets[0].cached_block_count = 1;
  targets[1].decode_attn_tp_rank_in_group = 1;
  targets[1].kv_pool_base_va = 0x2000;
  targets[1].kv_block_ids = {3};

  IncomingPrefillRequest req;
  req.req_id = 12;
  req.decode_engine_inference_addr = "decode:12";
  req.input_tokens = {7, 8};
  req.sampling_config_wire = &wire;
  req.decode_targets_wire = targets.data();
  req.num_decode_targets = 2;
  ep->OnIncoming(req);

  auto submitted = TryReadOne();
  ASSERT_NE(submitted, nullptr);
  ASSERT_EQ(submitted->pd_v2_decode_targets.size(), 2u);
  EXPECT_EQ(submitted->pd_v2_decode_targets[0].kv_pool_base_va, 0x1000u);
  EXPECT_EQ(submitted->pd_v2_decode_targets[0].kv_block_ids, (std::vector<int>{1, 2}));
  EXPECT_EQ(submitted->pd_v2_decode_targets[0].cached_block_count, 1u);
  EXPECT_EQ(submitted->pd_v2_decode_targets[1].kv_pool_base_va, 0x2000u);

  // pd_v2_on_destroy is wired to fire from InferRequest's destroy chain
  // (model_driver AddOnDestroy), NOT from ~Request, so invoke it directly to
  // verify the lambda routes to connector->Release with the captured req_id.
  ASSERT_TRUE(static_cast<bool>(submitted->pd_v2_on_destroy));
  submitted->pd_v2_on_destroy();
  EXPECT_EQ(conn_->release_calls, 1);
  EXPECT_EQ(conn_->last_release_req_id, req.req_id);
}

// OnBarrierDone forwards into the embedded completion (drives a full complete).
TEST_F(PdV2PrefillEndpointTest, OnBarrierDoneForwardsToCompletion) {
  auto ep = MakeEndpoint();
  auto wire = MakeWire();
  IncomingPrefillRequest req;
  req.req_id = 13;
  req.decode_engine_inference_addr = "decode:13";
  req.input_tokens = {1, 2};
  req.sampling_config_wire = &wire;
  req.num_decode_targets = 1;  // expects 1 barrier report
  ep->OnIncoming(req);
  auto submitted = TryReadOne();
  ASSERT_NE(submitted, nullptr);

  // Produce a generated token, then fire the request-finished waiter callback
  // (tokens gate) and the barrier gate; completion should send.
  submitted->output_tokens.push_back(999);
  ASSERT_TRUE(static_cast<bool>(submitted->waiter));
  submitted->waiter->Notify();  // triggers MakeRequestFinishedCallback -> tokens gate
  ep->OnBarrierDone(13, 0);     // barrier gate -> complete
  EXPECT_EQ(conn_->send_calls, 1);
  EXPECT_EQ(conn_->last_status, PdCompleteStatus::kOk);
  EXPECT_EQ(conn_->last_tokens, (std::vector<int>{999}));
  EXPECT_EQ(ep->NumCompletedForTest(), 1u);
}

}  // namespace pd_v2
}  // namespace ksana_llm
