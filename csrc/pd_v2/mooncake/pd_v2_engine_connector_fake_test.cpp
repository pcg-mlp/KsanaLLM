/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <string>
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

}  // namespace mooncake_backend
}  // namespace pd_v2
}  // namespace ksana_llm
