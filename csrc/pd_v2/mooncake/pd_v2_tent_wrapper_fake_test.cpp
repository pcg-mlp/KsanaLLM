/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
// Coverage test for the pd_v2 Mooncake tent wrappers
// (engine_tent_wrapper.cpp / executor_tent_wrapper.cpp), driven against the
// FAKE in-process mooncake::TransferEngine (csrc/pd_v2/mooncake/fake_tent/).
// No RDMA, no metadata server, no second process — two wrapper instances in
// one address space exchange notifies and perform real in-process memcpy.
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "pd_v2/mooncake/engine_tent_wrapper.h"
#include "pd_v2/mooncake/executor_tent_wrapper.h"
#include "transfer_engine.h"  // fake: for Reset()/FailNextSubmitTransfer()

namespace ksana_llm {
namespace pd_v2 {
namespace mooncake_backend {

class TentWrapperFakeTest : public ::testing::Test {
 protected:
  void SetUp() override { mooncake::TransferEngine::Reset(); }
  void TearDown() override { mooncake::TransferEngine::Reset(); }
};

// ---- EngineTentWrapper ----

TEST_F(TentWrapperFakeTest, EngineInitializeTwiceRejected) {
  EngineTentWrapper w;
  EXPECT_TRUE(w.Initialize("redis://x", "engineA").OK());
  EXPECT_FALSE(w.Initialize("redis://x", "engineA").OK());  // already initialized
  EXPECT_TRUE(w.Shutdown().OK());
}

TEST_F(TentWrapperFakeTest, EngineMethodsBeforeInitRejected) {
  EngineTentWrapper w;
  Notification n{"pd_v2/x", "payload"};
  EXPECT_FALSE(w.SendNotification("peer", n).OK());
  std::vector<Notification> out;
  EXPECT_TRUE(w.ReceiveNotifications(&out).OK());  // returns OK, empty, when not init
  EXPECT_TRUE(out.empty());
  uint64_t bid = 7;
  EXPECT_FALSE(w.SubmitStageWriteAsync("peer", 0, nullptr, 0, &bid).OK());
  EXPECT_EQ(bid, 0u);
  EXPECT_FALSE(w.RegisterStageMemory(nullptr, 0).OK());
}

TEST_F(TentWrapperFakeTest, EngineNotifyRoundTripBetweenTwoEngines) {
  EngineTentWrapper a, b;
  ASSERT_TRUE(a.Initialize("redis://x", "engineA").OK());
  ASSERT_TRUE(b.Initialize("redis://x", "engineB").OK());

  // Binary (non-UTF-8) payload to exercise the base64 wrap/unwrap path.
  std::string payload;
  for (int i = 0; i < 256; ++i) payload.push_back(static_cast<char>(i));
  Notification sent{"pd_v2/prefill_req", payload};
  ASSERT_TRUE(a.SendNotification("engineB", sent).OK());

  std::vector<Notification> got;
  ASSERT_TRUE(b.ReceiveNotifications(&got).OK());
  ASSERT_EQ(got.size(), 1u);
  EXPECT_EQ(got[0].name, "pd_v2/prefill_req");
  EXPECT_EQ(got[0].msg, payload);  // base64 round-trip preserved binary bytes

  // A second send reuses the cached SegmentHandle path.
  ASSERT_TRUE(a.SendNotification("engineB", {"pd_v2/again", "x"}).OK());
  got.clear();
  ASSERT_TRUE(b.ReceiveNotifications(&got).OK());
  ASSERT_EQ(got.size(), 1u);
  EXPECT_EQ(got[0].name, "pd_v2/again");
}

TEST_F(TentWrapperFakeTest, EngineSendToUnknownPeerFails) {
  EngineTentWrapper a;
  ASSERT_TRUE(a.Initialize("redis://x", "engineA").OK());
  EXPECT_FALSE(a.SendNotification("nonexistent_peer", {"c", "m"}).OK());
}

TEST_F(TentWrapperFakeTest, EngineStageWriteMemcpiesIntoPeerBuffer) {
  EngineTentWrapper sender, receiver;
  ASSERT_TRUE(sender.Initialize("redis://x", "engineSender").OK());
  ASSERT_TRUE(receiver.Initialize("redis://x", "engineRecv").OK());

  std::vector<uint8_t> dst(64, 0);
  ASSERT_TRUE(receiver.RegisterStageMemory(dst.data(), dst.size()).OK());

  std::vector<uint8_t> src(64);
  for (size_t i = 0; i < src.size(); ++i) src[i] = static_cast<uint8_t>(i + 1);

  uint64_t batch = 0;
  ASSERT_TRUE(sender
                  .SubmitStageWriteAsync("engineRecv", reinterpret_cast<uint64_t>(dst.data()), src.data(), src.size(),
                                         &batch)
                  .OK());
  EXPECT_NE(batch, 0u);
  EXPECT_EQ(std::memcmp(dst.data(), src.data(), src.size()), 0);  // payload landed

  bool completed = false, failed = false;
  ASSERT_TRUE(sender.GetBatchStatus(batch, &completed, &failed).OK());
  EXPECT_TRUE(completed);
  EXPECT_FALSE(failed);
  EXPECT_TRUE(sender.FreeBatchID(batch).OK());
}

TEST_F(TentWrapperFakeTest, EngineStageWriteSubmitFailureSurfaces) {
  EngineTentWrapper sender, receiver;
  ASSERT_TRUE(sender.Initialize("redis://x", "engineSender").OK());
  ASSERT_TRUE(receiver.Initialize("redis://x", "engineRecv").OK());
  std::vector<uint8_t> dst(8, 0), src(8, 9);
  ASSERT_TRUE(receiver.RegisterStageMemory(dst.data(), dst.size()).OK());

  mooncake::TransferEngine::FailNextSubmitTransfer();
  uint64_t batch = 123;
  Status s = sender.SubmitStageWriteAsync("engineRecv", reinterpret_cast<uint64_t>(dst.data()), src.data(), src.size(),
                                          &batch);
  EXPECT_FALSE(s.OK());
  EXPECT_EQ(batch, 0u);
}

// ---- ExecutorTentWrapper ----

TEST_F(TentWrapperFakeTest, ExecutorRegisterAndWrite) {
  ExecutorTentWrapper recv;  // acts as the remote KV-pool owner
  ASSERT_TRUE(recv.Initialize("redis://x", "rank0", /*rpc_port=*/21000).OK());
  std::vector<uint8_t> kv(128, 0);
  ASSERT_TRUE(recv.RegisterKvPool(kv.data(), kv.size(), "cpu:0").OK());

  ExecutorTentWrapper sender;
  ASSERT_TRUE(sender.Initialize("redis://x", "rank1", /*rpc_port=*/21001).OK());

  std::vector<uint8_t> src(32);
  for (size_t i = 0; i < src.size(); ++i) src[i] = static_cast<uint8_t>(0xA0 + (i & 0x0f));

  // Synchronous write.
  ASSERT_TRUE(sender.SubmitWrite("rank0", reinterpret_cast<uint64_t>(kv.data()), src.data(), src.size()).OK());
  EXPECT_EQ(std::memcmp(kv.data(), src.data(), src.size()), 0);
}

TEST_F(TentWrapperFakeTest, ExecutorSubmitWriteAsyncAndPoll) {
  ExecutorTentWrapper recv, sender;
  ASSERT_TRUE(recv.Initialize("redis://x", "rank0", 21000).OK());
  ASSERT_TRUE(sender.Initialize("redis://x", "rank1", 21001).OK());
  std::vector<uint8_t> kv(64, 0), src(64, 0x5a);
  ASSERT_TRUE(recv.RegisterKvPool(kv.data(), kv.size(), "cpu:0").OK());

  uint64_t batch = 0;
  ASSERT_TRUE(
      sender.SubmitWriteAsync("rank0", reinterpret_cast<uint64_t>(kv.data()), src.data(), src.size(), &batch).OK());
  EXPECT_NE(batch, 0u);
  bool completed = false, failed = false;
  ASSERT_TRUE(sender.GetBatchStatus(batch, &completed, &failed).OK());
  EXPECT_TRUE(completed);
  EXPECT_FALSE(failed);
  EXPECT_TRUE(sender.FreeBatchID(batch).OK());
}

TEST_F(TentWrapperFakeTest, ExecutorSubmitWriteBatchAsync) {
  ExecutorTentWrapper recv, sender;
  ASSERT_TRUE(recv.Initialize("redis://x", "rank0", 21000).OK());
  ASSERT_TRUE(sender.Initialize("redis://x", "rank1", 21001).OK());
  std::vector<uint8_t> kv(256, 0);
  ASSERT_TRUE(recv.RegisterKvPool(kv.data(), kv.size(), "cpu:0").OK());

  std::vector<uint8_t> b0(16, 0x11), b1(16, 0x22);
  std::vector<ExecutorTentWrapper::BatchWriteEntry> entries = {
      {reinterpret_cast<uint64_t>(kv.data()), b0.data(), b0.size()},
      {reinterpret_cast<uint64_t>(kv.data() + 16), b1.data(), b1.size()},
  };
  uint64_t batch = 0;
  ASSERT_TRUE(sender.SubmitWriteBatchAsync("rank0", entries, &batch).OK());
  EXPECT_NE(batch, 0u);
  EXPECT_EQ(std::memcmp(kv.data(), b0.data(), 16), 0);
  EXPECT_EQ(std::memcmp(kv.data() + 16, b1.data(), 16), 0);

  // Empty entries -> batch id 0, OK (no-op).
  uint64_t empty_batch = 99;
  sender.SubmitWriteBatchAsync("rank0", {}, &empty_batch);
  EXPECT_EQ(empty_batch, 0u);
}

TEST_F(TentWrapperFakeTest, ExecutorWriteToUnknownPeerFails) {
  ExecutorTentWrapper sender;
  ASSERT_TRUE(sender.Initialize("redis://x", "rank1", 21001).OK());
  std::vector<uint8_t> src(8, 1);
  uint64_t batch = 7;
  EXPECT_FALSE(sender.SubmitWriteAsync("ghost", 0x1000, src.data(), src.size(), &batch).OK());
  EXPECT_EQ(batch, 0u);
}

TEST_F(TentWrapperFakeTest, ExecutorMethodsBeforeInitRejected) {
  ExecutorTentWrapper w;
  std::vector<uint8_t> buf(8, 0);
  EXPECT_FALSE(w.RegisterKvPool(buf.data(), buf.size(), "cpu:0").OK());
  uint64_t batch = 5;
  EXPECT_FALSE(w.SubmitWriteAsync("rank0", 0x10, buf.data(), buf.size(), &batch).OK());
}

}  // namespace mooncake_backend
}  // namespace pd_v2
}  // namespace ksana_llm
