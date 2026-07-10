/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
// Coverage test for MooncakeEngineConnector against the FAKE in-process
// mooncake::TransferEngine + FAKE IpcFuncWrapper. The reliable cases use
// P2PHANDSHAKE metadata (no Redis) and exercise Initialize (Decode + Prefill
// roles), the boot write-side/stage pools, GetKvBufferVa RPC + caching, RouteRequest /
// CanAccept with no peer, and the engine-only-invalid methods. A final
// Redis-gated case brings up a Decode + Prefill connector pair and waits for
// service discovery to make a peer selectable (covering the keepalive /
// heartbeat / snapshot paths); it GTEST_SKIPs when Redis is unavailable.
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include "ipc/ipc_func_wrapper.h"  // fake
#include "pd_v2/mooncake/mooncake_engine_connector.h"
#include "pd_v2/public/pd_v2_config.h"
#include "pd_v2/public/pd_v2_types.h"
#include "transfer_engine.h"  // fake: Reset()

namespace ksana_llm {
namespace pd_v2 {
namespace mooncake_backend {

namespace {

PdV2ConnectorConfig EngineConfig(NodeRole role, const std::string& inference_addr, const std::string& metadata_addr) {
  PdV2ConnectorConfig c;
  c.deployment_role = DeploymentRole::kEngine;
  c.backend_type = "mooncake";
  c.cluster_name = "dev_cluster";
  c.node_role = role;
  c.inference_addr = inference_addr;
  c.metadata_addr = metadata_addr;
  return c;
}

constexpr const char* kRedisAddr = "redis://29.226.50.29:6379";

bool RedisAvailable() {
  // RedisAliveStore reads MC_REDIS_PASSWORD from env; set it for the probe.
  setenv("MC_REDIS_PASSWORD", "Redis@123456", /*overwrite=*/1);
  RedisAliveStore s;
  Status rs = s.Connect(kRedisAddr, /*password=*/"");
  bool ok = rs.OK();
  if (ok) s.Disconnect();
  return ok;
}

}  // namespace

class EngineConnectorFakeTest : public ::testing::Test {
 protected:
  void SetUp() override { mooncake::TransferEngine::Reset(); }
  void TearDown() override { mooncake::TransferEngine::Reset(); }
};

TEST_F(EngineConnectorFakeTest, InitializeRejectsExecutorRole) {
  MooncakeEngineConnector conn;
  PdV2ConnectorConfig c = EngineConfig(NodeRole::kDecode, "node:1", "P2PHANDSHAKE");
  c.deployment_role = DeploymentRole::kExecutor;
  EXPECT_FALSE(conn.Initialize(c).OK());
}

TEST_F(EngineConnectorFakeTest, DecodeRoleInitializeAndShutdown) {
  MooncakeEngineConnector conn;
  ASSERT_TRUE(conn.Initialize(EngineConfig(NodeRole::kDecode, "dec:1", "P2PHANDSHAKE")).OK());
  // RegisterKvPool is executor-only on the engine connector.
  EXPECT_FALSE(conn.RegisterKvPool(nullptr, 0, 0, 0, 0).OK());
  ASSERT_TRUE(conn.Shutdown().OK());
}

TEST_F(EngineConnectorFakeTest, PrefillRoleInitializeAndShutdown) {
  MooncakeEngineConnector conn;
  ASSERT_TRUE(conn.Initialize(EngineConfig(NodeRole::kPrefill, "pre:1", "P2PHANDSHAKE")).OK());
  // Prefill side: registering an incoming hook + snapshot provider must not crash.
  conn.RegisterIncomingHook([](const IncomingPrefillRequest&, void*) {}, nullptr);
  conn.SetSnapshotProvider([]() { return PeerHealth{}; });
  ASSERT_TRUE(conn.Shutdown().OK());
}

TEST_F(EngineConnectorFakeTest, GetKvBufferVaThroughIpcAndCaches) {
  MooncakeEngineConnector conn;
  ASSERT_TRUE(conn.Initialize(EngineConfig(NodeRole::kDecode, "dec:1", "P2PHANDSHAKE")).OK());

  IpcFuncWrapper ipc;
  std::atomic<int> query_calls{0};
  ipc.kv_query_fn_ = [&](uint64_t* base, uint64_t* total, uint64_t* bs, uint64_t* nb, uint64_t* nl) -> bool {
    query_calls.fetch_add(1);
    if (base) *base = 0xCAFE0000;
    if (total) *total = 4096;
    if (bs) *bs = 256;
    if (nb) *nb = 16;
    if (nl) *nl = 4;
    return true;
  };
  conn.SetIpcFuncWrapper(&ipc);

  uint64_t base = 0, total = 0;
  ASSERT_TRUE(conn.GetKvBufferVa(/*rank=*/0, &base, &total).OK());
  EXPECT_EQ(base, 0xCAFE0000u);
  EXPECT_EQ(total, 4096u);
  EXPECT_EQ(query_calls.load(), 1);

  // Second call for the same rank is served from cache (no extra RPC).
  uint64_t base2 = 0, total2 = 0;
  ASSERT_TRUE(conn.GetKvBufferVa(0, &base2, &total2).OK());
  EXPECT_EQ(base2, 0xCAFE0000u);
  EXPECT_EQ(query_calls.load(), 1);  // still cached

  ASSERT_TRUE(conn.Shutdown().OK());
}

TEST_F(EngineConnectorFakeTest, GetKvBufferVaFailsWhenKvNotReady) {
  MooncakeEngineConnector conn;
  ASSERT_TRUE(conn.Initialize(EngineConfig(NodeRole::kDecode, "dec:1", "P2PHANDSHAKE")).OK());
  IpcFuncWrapper ipc;
  ipc.kv_query_fn_ = [](uint64_t*, uint64_t*, uint64_t*, uint64_t*, uint64_t*) -> bool { return false; };
  conn.SetIpcFuncWrapper(&ipc);
  uint64_t base = 0, total = 0;
  EXPECT_FALSE(conn.GetKvBufferVa(0, &base, &total).OK());
  ASSERT_TRUE(conn.Shutdown().OK());
}

TEST_F(EngineConnectorFakeTest, NoPeerCanAcceptAndRouteRequest) {
  MooncakeEngineConnector conn;
  ASSERT_TRUE(conn.Initialize(EngineConfig(NodeRole::kDecode, "dec:1", "P2PHANDSHAKE")).OK());
  // No PrefillSnapshot ever arrived -> PeerSelector has no peer.
  EXPECT_FALSE(conn.CanAccept(/*req_id=*/1, /*input_token_count=*/128));
  // Default (non-LinUCB) RouteRequest forwards to CanAccept -> nullopt.
  LocalArmContext ctx;
  ctx.remaining_prefill_tokens = 128;
  EXPECT_FALSE(conn.RouteRequest(1, ctx).has_value());
  ASSERT_TRUE(conn.Shutdown().OK());
}

TEST_F(EngineConnectorFakeTest, HasHealthyPrefillPeerFalseForPrefillRole) {
  // Prefill-role connectors never build a PeerSelector, so the Decode-only
  // /health gate must report no healthy Prefill peer.
  MooncakeEngineConnector conn;
  ASSERT_TRUE(conn.Initialize(EngineConfig(NodeRole::kPrefill, "pre:1", "P2PHANDSHAKE")).OK());
  EXPECT_FALSE(conn.HasHealthyPrefillPeer());
  ASSERT_TRUE(conn.Shutdown().OK());
}

TEST_F(EngineConnectorFakeTest, HasHealthyPrefillPeerFalseWithNoSnapshot) {
  // Decode role owns a PeerSelector, but until a PrefillSnapshot arrives no
  // peer is healthy -> the gate stays closed.
  MooncakeEngineConnector conn;
  ASSERT_TRUE(conn.Initialize(EngineConfig(NodeRole::kDecode, "dec:1", "P2PHANDSHAKE")).OK());
  EXPECT_FALSE(conn.HasHealthyPrefillPeer());
  ASSERT_TRUE(conn.Shutdown().OK());
}

TEST_F(EngineConnectorFakeTest, MiscDecodeAccessors) {
  MooncakeEngineConnector conn;
  ASSERT_TRUE(conn.Initialize(EngineConfig(NodeRole::kDecode, "dec:1", "P2PHANDSHAKE")).OK());
  // Release of an unknown req is idempotent.
  EXPECT_TRUE(conn.Release(999).OK());
  // No latency sample yet for any peer.
  EXPECT_FALSE(conn.GetPeerLatencyUs("ghost").has_value());
  ASSERT_TRUE(conn.Shutdown().OK());
}

TEST_F(EngineConnectorFakeTest, SendPrefillCompleteUnknownRoute) {
  MooncakeEngineConnector conn;
  ASSERT_TRUE(conn.Initialize(EngineConfig(NodeRole::kPrefill, "pre:1", "P2PHANDSHAKE")).OK());
  PeerHealth h;
  PdPrefillMetrics m;
  // No incoming route recorded for this addr/req — connector handles it
  // gracefully (best-effort send to an unknown peer).
  conn.SendPrefillComplete("unknown_decode:1", /*req_id=*/7, PdCompleteStatus::kOk, {1, 2, 3}, h, m);
  ASSERT_TRUE(conn.Shutdown().OK());
}

// Redis-backed service discovery: a Prefill + Decode connector pair over the
// fake tent should discover each other through Redis keepalive/heartbeat, so
// the Decode side eventually has a selectable peer. Exercises the keepalive
// loop, heartbeat loop, BroadcastSnapshot and OnPrefillSnapshot paths.
TEST_F(EngineConnectorFakeTest, RedisDiscoveryMakesPeerSelectable) {
  if (!RedisAvailable()) GTEST_SKIP() << "Redis unavailable";
  setenv("MC_REDIS_PASSWORD", "Redis@123456", /*overwrite=*/1);

  MooncakeEngineConnector prefill;
  PdV2ConnectorConfig pc = EngineConfig(NodeRole::kPrefill, "29.226.50.29:18090", kRedisAddr);
  pc.engine.snapshot_push_interval_ms = 100;  // fast discovery for the test
  ASSERT_TRUE(prefill.Initialize(pc).OK());
  prefill.RegisterIncomingHook([](const IncomingPrefillRequest&, void*) {}, nullptr);
  prefill.SetSnapshotProvider([]() { return PeerHealth{}; });

  MooncakeEngineConnector decode;
  PdV2ConnectorConfig dc = EngineConfig(NodeRole::kDecode, "29.226.50.29:18089", kRedisAddr);
  dc.engine.snapshot_push_interval_ms = 100;
  ASSERT_TRUE(decode.Initialize(dc).OK());

  // Wait for keepalive (Decode -> Redis) + heartbeat (Prefill polls Redis,
  // pushes PrefillSnapshot) + Decode poller to register the peer.
  bool accepted = false;
  for (int i = 0; i < 100 && !accepted; ++i) {
    accepted = decode.CanAccept(/*req_id=*/1, /*input_token_count=*/16);
    if (!accepted) std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  EXPECT_TRUE(accepted) << "Decode never discovered the Prefill peer via Redis";
  if (accepted) decode.Release(1);

  // A selectable peer is a healthy peer: the Decode-side /health gate must now
  // report a healthy Prefill peer. The Prefill side has no PeerSelector and so
  // never reports one.
  EXPECT_EQ(decode.HasHealthyPrefillPeer(), accepted);
  EXPECT_FALSE(prefill.HasHealthyPrefillPeer());

  ASSERT_TRUE(decode.Shutdown().OK());
  ASSERT_TRUE(prefill.Shutdown().OK());
}

// File-scope rendezvous for the C-function-pointer callbacks (PdIncomingReqCb
// and PdRequestCompleteCb can't capture).
namespace {
MooncakeEngineConnector* g_prefill_for_reply = nullptr;
std::mutex g_rt_mu;
std::atomic<bool> g_incoming_seen{false};
std::atomic<bool> g_complete_seen{false};
std::vector<int> g_complete_tokens;
PdCompleteStatus g_complete_status = PdCompleteStatus::kFailed;

void RoundTripIncomingHook(const IncomingPrefillRequest& req, void* /*ctx*/) {
  g_incoming_seen.store(true);
  // Reply with a deterministic token set so the Decode side can verify.
  if (g_prefill_for_reply != nullptr) {
    PeerHealth h;
    PdPrefillMetrics m;
    g_prefill_for_reply->SendPrefillComplete(req.decode_engine_inference_addr, req.req_id, PdCompleteStatus::kOk,
                                             {101, 102, 103}, h, m);
  }
}

void RoundTripOnComplete(uint64_t /*req_id*/, PdCompleteStatus status, const std::vector<int>& generated_tokens,
                         const PdPrefillMetrics& /*metrics*/, const std::string& /*prefill_peer*/, void* /*ctx*/) {
  std::lock_guard<std::mutex> lk(g_rt_mu);
  g_complete_status = status;
  g_complete_tokens = generated_tokens;
  g_complete_seen.store(true);
}
}  // namespace

// Full Decode→Prefill→Decode round trip over Redis discovery + fake tent:
// SubmitPrefillRequest stages the wire payload (write-side pool → fake RDMA WRITE into
// the Prefill stage pool), notifies the peer; the Prefill poller deserializes,
// fires the incoming hook (which replies via SendPrefillComplete), and the
// Decode poller delivers the generated tokens to on_complete. Exercises
// BuildPrefillRequest / AcquirePeerStageSlots / StageWriteAndWait /
// OnPrefillReq / OnStageAck / OnPrefillComplete end-to-end.
TEST_F(EngineConnectorFakeTest, RedisFullPrefillRoundTrip) {
  if (!RedisAvailable()) GTEST_SKIP() << "Redis unavailable";
  setenv("MC_REDIS_PASSWORD", "Redis@123456", /*overwrite=*/1);

  g_incoming_seen.store(false);
  g_complete_seen.store(false);
  {
    std::lock_guard<std::mutex> lk(g_rt_mu);
    g_complete_tokens.clear();
  }

  MooncakeEngineConnector prefill;
  PdV2ConnectorConfig pc = EngineConfig(NodeRole::kPrefill, "29.226.50.29:18092", kRedisAddr);
  pc.engine.snapshot_push_interval_ms = 100;
  ASSERT_TRUE(prefill.Initialize(pc).OK());
  g_prefill_for_reply = &prefill;
  prefill.RegisterIncomingHook(RoundTripIncomingHook, nullptr);
  prefill.SetSnapshotProvider([]() { return PeerHealth{}; });

  MooncakeEngineConnector decode;
  PdV2ConnectorConfig dc = EngineConfig(NodeRole::kDecode, "29.226.50.29:18091", kRedisAddr);
  dc.engine.snapshot_push_interval_ms = 100;
  ASSERT_TRUE(decode.Initialize(dc).OK());

  // Decode-side KV buffer VA lookup (BuildPrefillRequest queries it per rank).
  IpcFuncWrapper ipc;
  ipc.kv_query_fn_ = [](uint64_t* base, uint64_t* total, uint64_t* bs, uint64_t* nb, uint64_t* nl) -> bool {
    if (base) *base = 0x10000000;
    if (total) *total = 1u << 20;
    if (bs) *bs = 4096;
    if (nb) *nb = 256;
    if (nl) *nl = 4;
    return true;
  };
  decode.SetIpcFuncWrapper(&ipc);

  // Wait for discovery so the Decode has a peer + a stage assignment.
  bool accepted = false;
  for (int i = 0; i < 100 && !accepted; ++i) {
    accepted = decode.CanAccept(/*req_id=*/77, /*input_token_count=*/8);
    if (!accepted) std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  ASSERT_TRUE(accepted) << "discovery did not complete";

  SamplingConfigWire sc;
  sc.max_new_tokens = 4;
  std::vector<int> input_tokens = {10, 11, 12, 13, 14, 15};
  std::vector<int> block_ids = {0, 1};
  Status s = decode.SubmitPrefillRequest(/*req_id=*/77, input_tokens, block_ids, sc, /*attn_dp_group_id=*/0,
                                         /*cached_block_count=*/0, /*decode_prefix_cache_offset=*/0,
                                         RoundTripOnComplete, /*ctx=*/nullptr);
  ASSERT_TRUE(s.OK()) << "SubmitPrefillRequest failed: " << s.GetMessage();

  for (int i = 0; i < 200 && !g_complete_seen.load(); ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
  }
  EXPECT_TRUE(g_incoming_seen.load()) << "Prefill never received the request";
  ASSERT_TRUE(g_complete_seen.load()) << "Decode never received the completion";
  {
    std::lock_guard<std::mutex> lk(g_rt_mu);
    EXPECT_EQ(g_complete_status, PdCompleteStatus::kOk);
    EXPECT_EQ(g_complete_tokens, (std::vector<int>{101, 102, 103}));
  }

  g_prefill_for_reply = nullptr;
  ASSERT_TRUE(decode.Shutdown().OK());
  ASSERT_TRUE(prefill.Shutdown().OK());
}

}  // namespace mooncake_backend
}  // namespace pd_v2
}  // namespace ksana_llm
