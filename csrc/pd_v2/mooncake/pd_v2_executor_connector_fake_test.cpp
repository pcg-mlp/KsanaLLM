/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
// Coverage test for MooncakeExecutorConnector, driven against the FAKE
// in-process mooncake::TransferEngine + a FAKE IpcFuncWrapper. Exercises the
// executor-side PdV2ConnectorInterface surface (RegisterKvPool, the
// KV-write/batch-tracking/poll path consumed by PdV2KvDispatcher +
// PdV2KvBarrier, and the engine-only methods that must return errors)
// without RDMA, a metadata server, or a second process.
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "pd_v2/mooncake/mooncake_executor_connector.h"
#include "pd_v2/mooncake/redis_alive_store.h"
#include "pd_v2/public/pd_v2_config.h"
#include "ipc/ipc_func_wrapper.h"  // fake
#include "transfer_engine.h"  // fake: Reset()


namespace ksana_llm {
namespace pd_v2 {
namespace mooncake_backend {

namespace {

PdV2ConnectorConfig ExecConfig(const std::string& inference_addr, int rank, uint64_t rpc_port) {
  PdV2ConnectorConfig c;
  c.deployment_role = DeploymentRole::kExecutor;
  c.backend_type = "mooncake";
  c.cluster_name = "dev_cluster";
  c.inference_addr = inference_addr;
  c.metadata_addr = "P2PHANDSHAKE";  // not redis:// -> epoch poll thread skipped
  c.executor.executor_rank = rank;
  c.executor.rpc_port = rpc_port;
  return c;
}

constexpr const char* kRedisAddr = "redis://29.226.50.29:6379";

bool RedisAvailable() {
  setenv("MC_REDIS_PASSWORD", "Redis@123456", /*overwrite=*/1);
  RedisAliveStore s;
  Status rs = s.Connect(kRedisAddr, /*password=*/"");
  bool ok = rs.OK();
  if (ok) s.Disconnect();
  return ok;
}

}  // namespace

class ExecutorConnectorFakeTest : public ::testing::Test {
 protected:
  void SetUp() override { mooncake::TransferEngine::Reset(); }
  void TearDown() override { mooncake::TransferEngine::Reset(); }
};

TEST_F(ExecutorConnectorFakeTest, InitializeRejectsEngineRole) {
  MooncakeExecutorConnector conn;
  PdV2ConnectorConfig c = ExecConfig("node", 0, 21000);
  c.deployment_role = DeploymentRole::kEngine;
  EXPECT_FALSE(conn.Initialize(c).OK());
}

TEST_F(ExecutorConnectorFakeTest, RegisterBeforeInitializeRejected) {
  MooncakeExecutorConnector conn;
  std::vector<uint8_t> buf(64, 0);
  EXPECT_FALSE(conn.RegisterKvPool(buf.data(), buf.size(), 32, 2, 4).OK());
}

TEST_F(ExecutorConnectorFakeTest, RegisterKvPoolAndQuery) {
  MooncakeExecutorConnector conn;
  ASSERT_TRUE(conn.Initialize(ExecConfig("decode0", /*rank=*/0, 21000)).OK());

  std::vector<uint8_t> kv(1024, 0);
  ASSERT_TRUE(conn.RegisterKvPool(kv.data(), kv.size(), /*block_size=*/256, /*num_blocks=*/4, /*num_layers=*/8).OK());

  uint64_t base = 0, total = 0;
  size_t bs = 0, nb = 0, nl = 0;
  ASSERT_TRUE(conn.GetLocalKvPool(&base, &total, &bs, &nb, &nl));
  EXPECT_EQ(base, reinterpret_cast<uint64_t>(kv.data()));
  EXPECT_EQ(total, kv.size());
  EXPECT_EQ(bs, 256u);
  EXPECT_EQ(nb, 4u);
  EXPECT_EQ(nl, 8u);

  // Shape override delegates to the same state.
  uint64_t b2 = 0, t2 = 0;
  size_t s2 = 0, n2 = 0, l2 = 0;
  EXPECT_TRUE(conn.GetLocalKvPoolShape(&b2, &t2, &s2, &n2, &l2));
  EXPECT_EQ(b2, base);

  ASSERT_TRUE(conn.Shutdown().OK());
}

TEST_F(ExecutorConnectorFakeTest, GetLocalKvPoolFalseBeforeRegister) {
  MooncakeExecutorConnector conn;
  ASSERT_TRUE(conn.Initialize(ExecConfig("decode0", 0, 21000)).OK());
  uint64_t base = 0, total = 0;
  size_t bs = 0, nb = 0, nl = 0;
  EXPECT_FALSE(conn.GetLocalKvPool(&base, &total, &bs, &nb, &nl));
}

TEST_F(ExecutorConnectorFakeTest, SetIpcFuncWrapperWiresQueryFn) {
  MooncakeExecutorConnector conn;
  ASSERT_TRUE(conn.Initialize(ExecConfig("decode0", 0, 21000)).OK());
  std::vector<uint8_t> kv(512, 0);
  ASSERT_TRUE(conn.RegisterKvPool(kv.data(), kv.size(), 128, 4, 2).OK());

  IpcFuncWrapper ipc;
  conn.SetIpcFuncWrapper(&ipc);
  ASSERT_TRUE(static_cast<bool>(ipc.kv_query_fn_));

  // Engine-side RPC resolves through the registered query fn.
  uint64_t base = 0, total = 0, bs = 0, nb = 0, nl = 0;
  ASSERT_TRUE(ipc.PdV2GetKvPoolBaseVa(/*rank=*/0, &base, &total, &bs, &nb, &nl).OK());
  EXPECT_EQ(base, reinterpret_cast<uint64_t>(kv.data()));
  EXPECT_EQ(bs, 128u);
  EXPECT_EQ(nl, 2u);

  // Null wrapper is a no-op (no crash).
  conn.SetIpcFuncWrapper(nullptr);
}

TEST_F(ExecutorConnectorFakeTest, EngineOnlyMethodsReturnInvalid) {
  MooncakeExecutorConnector conn;
  ASSERT_TRUE(conn.Initialize(ExecConfig("decode0", 0, 21000)).OK());
  EXPECT_FALSE(conn.CanAccept(1, 10));
  uint64_t a = 0, b = 0;
  EXPECT_FALSE(conn.GetKvBufferVa(0, &a, &b).OK());
  EXPECT_FALSE(conn.Release(1).OK());
  SamplingConfigWire sc;
  EXPECT_FALSE(conn.SubmitPrefillRequest(1, {}, {}, sc, 0, 0, 0, nullptr, nullptr).OK());
  PeerHealth h;
  PdPrefillMetrics m;
  EXPECT_FALSE(conn.SendPrefillComplete("addr", 1, PdCompleteStatus::kOk, {}, h, m).OK());
}

TEST_F(ExecutorConnectorFakeTest, SubmitWriteAsyncForRequestTracksAndWrites) {
  // Remote (receiver) connector registers a KV pool under "remote.exec.r0".
  MooncakeExecutorConnector remote;
  ASSERT_TRUE(remote.Initialize(ExecConfig("remote", /*rank=*/0, 22000)).OK());
  std::vector<uint8_t> remote_kv(256, 0);
  ASSERT_TRUE(remote.RegisterKvPool(remote_kv.data(), remote_kv.size(), 64, 4, 4).OK());

  // Sender (Prefill executor) writes into the remote KV pool.
  MooncakeExecutorConnector sender;
  ASSERT_TRUE(sender.Initialize(ExecConfig("sender", /*rank=*/0, 22001)).OK());

  std::vector<uint8_t> src(64);
  for (size_t i = 0; i < src.size(); ++i) src[i] = static_cast<uint8_t>(i ^ 0x3c);

  uint64_t batch = 0;
  ASSERT_TRUE(sender
                  .SubmitWriteAsyncForRequest("remote.exec.r0", reinterpret_cast<uint64_t>(remote_kv.data()),
                                              src.data(), src.size(), /*req_id=*/42, &batch)
                  .OK());
  EXPECT_NE(batch, 0u);
  EXPECT_EQ(std::memcmp(remote_kv.data(), src.data(), src.size()), 0);

  // Barrier surface: in-flight ids, poll, erase.
  auto ids = sender.GetInflightBatchIds(42);
  ASSERT_EQ(ids.size(), 1u);
  EXPECT_EQ(ids[0], batch);
  bool completed = false, failed = false;
  ASSERT_TRUE(sender.PollBatchStatus(batch, &completed, &failed).OK());
  EXPECT_TRUE(completed);
  EXPECT_FALSE(failed);
  sender.EraseInflightBatch(42, batch);
  EXPECT_TRUE(sender.GetInflightBatchIds(42).empty());
}

TEST_F(ExecutorConnectorFakeTest, SubmitWriteBatchAsyncForRequest) {
  MooncakeExecutorConnector remote, sender;
  ASSERT_TRUE(remote.Initialize(ExecConfig("remote", 0, 22000)).OK());
  std::vector<uint8_t> remote_kv(256, 0);
  ASSERT_TRUE(remote.RegisterKvPool(remote_kv.data(), remote_kv.size(), 64, 4, 4).OK());
  ASSERT_TRUE(sender.Initialize(ExecConfig("sender", 0, 22001)).OK());

  std::vector<uint8_t> b0(32, 0xAB), b1(32, 0xCD);
  std::vector<PdV2ConnectorInterface::KvWriteEntry> entries = {
      {reinterpret_cast<uint64_t>(remote_kv.data()), b0.data(), b0.size()},
      {reinterpret_cast<uint64_t>(remote_kv.data() + 32), b1.data(), b1.size()},
  };
  uint64_t batch = 0;
  ASSERT_TRUE(sender.SubmitWriteBatchAsyncForRequest("remote.exec.r0", entries, /*req_id=*/7, &batch).OK());
  EXPECT_NE(batch, 0u);
  EXPECT_EQ(std::memcmp(remote_kv.data(), b0.data(), 32), 0);
  EXPECT_EQ(std::memcmp(remote_kv.data() + 32, b1.data(), 32), 0);
  ASSERT_EQ(sender.GetInflightBatchIds(7).size(), 1u);

  // Empty entries: OK, nothing tracked.
  uint64_t empty_batch = 5;
  EXPECT_TRUE(sender.SubmitWriteBatchAsyncForRequest("remote.exec.r0", {}, /*req_id=*/8, &empty_batch).OK());
  EXPECT_EQ(empty_batch, 0u);
  EXPECT_TRUE(sender.GetInflightBatchIds(8).empty());
}

TEST_F(ExecutorConnectorFakeTest, SubmitToUnknownPeerFailsAndTracksNothing) {
  MooncakeExecutorConnector sender;
  ASSERT_TRUE(sender.Initialize(ExecConfig("sender", 0, 22001)).OK());
  std::vector<uint8_t> src(16, 1);
  uint64_t batch = 9;
  EXPECT_FALSE(sender.SubmitWriteAsyncForRequest("ghost.exec.r0", 0x1000, src.data(), src.size(), 1, &batch).OK());
  EXPECT_EQ(batch, 0u);
  EXPECT_TRUE(sender.GetInflightBatchIds(1).empty());
}

TEST_F(ExecutorConnectorFakeTest, BarrierSurfaceBeforeInitialize) {
  MooncakeExecutorConnector conn;  // never initialized
  EXPECT_TRUE(conn.GetInflightBatchIds(1).empty());
  bool c = false, f = false;
  EXPECT_FALSE(conn.PollBatchStatus(1, &c, &f).OK());
  uint64_t batch = 3;
  std::vector<uint8_t> src(8, 0);
  EXPECT_FALSE(conn.SubmitWriteAsyncForRequest("x", 0, src.data(), src.size(), 1, &batch).OK());
}

// Redis-backed executor epoch tracker: seed a decode_alive key, let the
// EpochPollLoop discover it, then change its epoch (invalidate path) and
// delete it (disappeared path). Exercises EpochPollLoop / ParseEpochFromJson
// / InvalidatePeerSegments. Skips when Redis is unreachable.
TEST_F(ExecutorConnectorFakeTest, RedisEpochPollDiscoverInvalidateDisappear) {
  if (!RedisAvailable()) GTEST_SKIP() << "Redis unavailable";
  setenv("MC_REDIS_PASSWORD", "Redis@123456", /*overwrite=*/1);

  const std::string key = "pd_v2/decode_alive/dev_cluster/29.226.50.29:18077";
  RedisAliveStore seeder;
  ASSERT_TRUE(seeder.Connect(kRedisAddr, "").OK());
  ASSERT_TRUE(seeder.Set(key, "{\"epoch\":100}", /*ttl_s=*/30).OK());

  MooncakeExecutorConnector conn;
  PdV2ConnectorConfig c = ExecConfig("execdecode", /*rank=*/0, 23000);
  c.metadata_addr = kRedisAddr;  // enables the epoch poll thread
  ASSERT_TRUE(conn.Initialize(c).OK());

  // Poll interval is ~1s; wait through discover -> epoch-change -> disappear.
  std::this_thread::sleep_for(std::chrono::milliseconds(1300));  // discover
  ASSERT_TRUE(seeder.Set(key, "{\"epoch\":200}", 30).OK());
  std::this_thread::sleep_for(std::chrono::milliseconds(1200));  // epoch mismatch -> invalidate
  ASSERT_TRUE(seeder.Del(key).OK());
  std::this_thread::sleep_for(std::chrono::milliseconds(1200));  // disappeared -> invalidate

  ASSERT_TRUE(conn.Shutdown().OK());
  seeder.Del(key);
  seeder.Disconnect();
}

}  // namespace mooncake_backend
}  // namespace pd_v2
}  // namespace ksana_llm
