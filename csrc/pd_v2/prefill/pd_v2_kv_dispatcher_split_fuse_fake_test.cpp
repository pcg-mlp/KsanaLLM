/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
//
// End-to-end split-fuse trigger test for PdV2KvDispatcher, driven against the
// REAL MooncakeExecutorConnector over the FAKE in-process mooncake tent
// (fake_tent/). No RDMA, no metadata server, no GPU block allocator: a fake
// BlockAllocatorManager hands out host pointers into a local source buffer so
// the dispatcher's SubmitWriteBatchAsyncForRequest lands bytes in the "remote"
// KV pool in-process.
//
// What it locks in (the split-fuse invariant this feature adds):
//   * is_last_prefill_step == false  → dispatcher DEFERS: no RDMA write, no
//     in-flight batch tracked on the connector. (Non-final chunk: full prompt
//     KV not yet resident.)
//   * is_last_prefill_step == true   → dispatcher TRANSFERS: bytes land in the
//     remote KV pool and the connector tracks the in-flight batch.
//
#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "cache_manager/block_allocator/block_allocator_manager.h"
#include "executor/executor_request_manager.h"
#include "ipc/ipc_func_wrapper.h"  // fake: records ReportPdV2BarrierDone
#include "pd_v2/mooncake/mooncake_executor_connector.h"
#include "pd_v2/prefill/pd_v2_kv_barrier.h"
#include "pd_v2/prefill/pd_v2_kv_dispatcher.h"
#include "pd_v2/public/pd_v2_config.h"
#include "runtime/executor_infer_request.h"
#include "runtime/infer_stage.h"
#include "transfer_engine.h"  // fake: Reset()

namespace ksana_llm {
namespace pd_v2 {
namespace {

using mooncake_backend::MooncakeExecutorConnector;

// ---- Fake block allocator stack: returns preset host pointers ---------------

class FakeBlockAllocator : public BlockAllocatorInterface {
 public:
  explicit FakeBlockAllocator(std::vector<void*> ptrs) : ptrs_(std::move(ptrs)) {}

  Status GetBlockPtrs(const std::vector<int>& blocks, std::vector<void*>& addrs) override {
    addrs.resize(blocks.size(), nullptr);
    for (size_t i = 0; i < blocks.size(); ++i) {
      const int b = blocks[i];
      if (b >= 0 && static_cast<size_t>(b) < ptrs_.size()) addrs[i] = ptrs_[b];
    }
    return Status();
  }

  // Unused stubs.
  void PreAllocateBlocks() override {}
  void Clear() override {}
  Status AllocateBlocks(size_t, std::vector<int>&) override { return Status(); }
  Status FreeBlocks(const std::vector<int>&) override { return Status(); }
  void* GetBlocksBasePtr() override { return nullptr; }
  int GetBlocksBaseId() override { return 0; }
  size_t GetFreeBlockNumber() override { return 0; }
  size_t GetUsedBlockNumber() override { return 0; }

 private:
  std::vector<void*> ptrs_;
};

class FakeBlockGroup : public BlockAllocatorGroupInterface {
 public:
  explicit FakeBlockGroup(std::shared_ptr<BlockAllocatorInterface> dev) : dev_(std::move(dev)) {}
  std::vector<int> GetBlockAllocatorDevices() const override { return {0}; }
  std::shared_ptr<BlockAllocatorInterface> GetHostBlockAllocator() const override { return nullptr; }
  std::shared_ptr<BlockAllocatorInterface> GetDeviceBlockAllocator(int) const override { return dev_; }
  const BlockAllocatorGroupConfig& GetBlockAllocatorGroupConfig() const override { return cfg_; }

 private:
  std::shared_ptr<BlockAllocatorInterface> dev_;
  BlockAllocatorGroupConfig cfg_{};
};

// Subclasses the concrete BlockAllocatorManager (the field type on
// ExecutorInferRequest). Base ctor only stores its args, so nullptr deps are
// safe; GetBlockAllocatorGroup is virtual and overridden to return the fake.
class FakeBlockManager : public BlockAllocatorManager {
 public:
  explicit FakeBlockManager(std::shared_ptr<BlockAllocatorGroupInterface> group)
      : BlockAllocatorManager(/*is_engine=*/false, /*memory_allocator=*/nullptr, /*context=*/nullptr),
        group_(std::move(group)) {}
  std::shared_ptr<BlockAllocatorGroupInterface> GetBlockAllocatorGroup(int) override { return group_; }

 private:
  std::shared_ptr<BlockAllocatorGroupInterface> group_;
};

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

// Layout shared by both sides.
constexpr size_t kBlockSize = 64;
constexpr size_t kNumBlocks = 2;
constexpr size_t kNumLayers = 2;

// Build a context-stage pd_v2 ExecutorInferRequest routed to prefill rank 0 →
// decode rank 0, writing both blocks into `remote_base`. `src_ptrs` are the
// per-block source host pointers (from the fake allocator).
std::shared_ptr<ExecutorInferRequest> BuildReq(int64_t req_id, bool is_last_prefill_step, uint64_t remote_base,
                                               const std::vector<void*>& src_ptrs) {
  auto dev = std::make_shared<FakeBlockAllocator>(src_ptrs);
  auto group = std::make_shared<FakeBlockGroup>(dev);
  auto req = std::make_shared<ExecutorInferRequest>();
  req->req_id = req_id;
  req->infer_stage = InferStage::kContext;
  req->step = 0;
  req->is_last_prefill_step = is_last_prefill_step;
  req->attn_dp_group_id = 0;
  req->kv_cache_blocks = {0, 1};
  req->block_allocator_manager = std::make_shared<FakeBlockManager>(group);

  DecodeTargetLite t;
  t.decode_attn_tp_rank_in_group = 0;
  t.kv_pool_base_va = remote_base;
  t.kv_block_ids = {0, 1};
  t.cached_block_count = 0;
  req->pd_v2_decode_targets = {t};
  req->pd_v2_decode_engine_inference_addr = "remote";
  req->pd_v2_decode_attn_dp_group_id = 0;
  return req;
}

PdV2KvDispatcher::Deps MakeDeps(PdV2ConnectorInterface* conn, ExecutorRequestManager* rm,
                                uint32_t transfer_layer_chunk_size = 0) {
  PdV2KvDispatcher::Deps d;
  d.connector = conn;
  d.request_manager = rm;
  // 0 → whole model in one shot at the final layer (no intermediate chunk-end
  // layers). >0 → every N layers is a chunk-end (enables mid-forward abort).
  d.transfer_layer_chunk_size = transfer_layer_chunk_size;
  d.num_layers = kNumLayers;
  d.prefill_tp = 1;
  d.prefill_attn_tp_per_dp = 1;
  d.use_mla = false;
  d.barrier = nullptr;  // barrier wiring not needed to observe the write trigger
  return d;
}

// Spin until the barrier has reported `total` outcomes (completed + failed) or
// the timeout elapses. Returns true if the count was reached.
bool WaitForBarrierOutcomes(const PdV2KvBarrier& barrier, size_t total, int max_ms = 3000) {
  for (int i = 0; i < max_ms; ++i) {
    if (barrier.CompletedCountForTest() + barrier.FailedCountForTest() >= total) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return false;
}

}  // namespace

class SplitFuseTransferFakeTest : public ::testing::Test {
 protected:
  void SetUp() override { mooncake::TransferEngine::Reset(); }
  void TearDown() override { mooncake::TransferEngine::Reset(); }
};

// Non-final chunk: dispatcher must NOT write and must NOT track an in-flight
// batch. Remote pool stays untouched.
TEST_F(SplitFuseTransferFakeTest, NonFinalChunkDoesNotTransfer) {
  MooncakeExecutorConnector remote, sender;
  ASSERT_TRUE(remote.Initialize(ExecConfig("remote", 0, 24000)).OK());
  std::vector<uint8_t> remote_kv(kNumBlocks * kBlockSize, 0);
  ASSERT_TRUE(remote.RegisterKvPool(remote_kv.data(), remote_kv.size(), kBlockSize, kNumBlocks, kNumLayers).OK());

  ASSERT_TRUE(sender.Initialize(ExecConfig("sender", 0, 24001)).OK());
  std::vector<uint8_t> sender_kv(kNumBlocks * kBlockSize, 0);
  ASSERT_TRUE(sender.RegisterKvPool(sender_kv.data(), sender_kv.size(), kBlockSize, kNumBlocks, kNumLayers).OK());

  std::vector<uint8_t> src(kNumBlocks * kBlockSize);
  for (size_t i = 0; i < src.size(); ++i) src[i] = static_cast<uint8_t>(i + 1);
  std::vector<void*> src_ptrs = {src.data(), src.data() + kBlockSize};

  ExecutorRequestManager rm;
  auto req =
      BuildReq(/*req_id=*/101, /*is_last_prefill_step=*/false, reinterpret_cast<uint64_t>(remote_kv.data()), src_ptrs);
  rm.AddInferRequest(req->req_id, req);

  PdV2KvDispatcher dispatcher(MakeDeps(&sender, &rm));
  dispatcher.CaptureEpochReqs(/*epoch_id=*/1, {req});
  dispatcher.OnLayerComplete(/*rank=*/0, /*epoch_id=*/1, /*layer_index=*/kNumLayers - 1);
  ASSERT_TRUE(dispatcher.WaitUntilIdleForTest());

  // Deferred: nothing written, nothing tracked.
  std::vector<uint8_t> zero(remote_kv.size(), 0);
  EXPECT_EQ(std::memcmp(remote_kv.data(), zero.data(), remote_kv.size()), 0);
  EXPECT_TRUE(sender.GetInflightBatchIds(101).empty());
}

// Final chunk: dispatcher writes every block into the remote KV pool and tracks
// the in-flight batch (which the barrier would later poll).
TEST_F(SplitFuseTransferFakeTest, FinalChunkTransfersAllBlocks) {
  MooncakeExecutorConnector remote, sender;
  ASSERT_TRUE(remote.Initialize(ExecConfig("remote", 0, 24000)).OK());
  std::vector<uint8_t> remote_kv(kNumBlocks * kBlockSize, 0);
  ASSERT_TRUE(remote.RegisterKvPool(remote_kv.data(), remote_kv.size(), kBlockSize, kNumBlocks, kNumLayers).OK());

  ASSERT_TRUE(sender.Initialize(ExecConfig("sender", 0, 24001)).OK());
  std::vector<uint8_t> sender_kv(kNumBlocks * kBlockSize, 0);
  ASSERT_TRUE(sender.RegisterKvPool(sender_kv.data(), sender_kv.size(), kBlockSize, kNumBlocks, kNumLayers).OK());

  std::vector<uint8_t> src(kNumBlocks * kBlockSize);
  for (size_t i = 0; i < src.size(); ++i) src[i] = static_cast<uint8_t>((i * 7 + 3) & 0xff);
  std::vector<void*> src_ptrs = {src.data(), src.data() + kBlockSize};

  ExecutorRequestManager rm;
  auto req =
      BuildReq(/*req_id=*/202, /*is_last_prefill_step=*/true, reinterpret_cast<uint64_t>(remote_kv.data()), src_ptrs);
  rm.AddInferRequest(req->req_id, req);

  PdV2KvDispatcher dispatcher(MakeDeps(&sender, &rm));
  dispatcher.CaptureEpochReqs(/*epoch_id=*/1, {req});
  dispatcher.OnLayerComplete(/*rank=*/0, /*epoch_id=*/1, /*layer_index=*/kNumLayers - 1);
  ASSERT_TRUE(dispatcher.WaitUntilIdleForTest());

  // Transferred: both blocks landed byte-for-byte, in-flight batch tracked.
  EXPECT_EQ(std::memcmp(remote_kv.data(), src.data(), src.size()), 0);
  EXPECT_FALSE(sender.GetInflightBatchIds(202).empty());

  // Barrier surface: the tracked batch polls as completed (fake tent writes are
  // synchronous), which is what PdV2KvBarrier would report to the engine.
  auto ids = sender.GetInflightBatchIds(202);
  ASSERT_FALSE(ids.empty());
  bool completed = false, failed = false;
  ASSERT_TRUE(sender.PollBatchStatus(ids[0], &completed, &failed).OK());
  EXPECT_TRUE(completed);
  EXPECT_FALSE(failed);
}

// Cross-epoch split-fuse lifecycle on ONE request: a non-final chunk epoch
// defers (empty plans, no write), then the next forward step — same req, now
// the final chunk — rebuilds fresh plans and transfers every block. Guards
// against state leaking between the two epochs (stale dispatch_plans_by_epoch_
// entry, epoch-key collision, dangling req refs) that the two isolated
// single-epoch tests above cannot surface.
TEST_F(SplitFuseTransferFakeTest, NonFinalEpochFollowedByFinalEpochTransfers) {
  MooncakeExecutorConnector remote, sender;
  ASSERT_TRUE(remote.Initialize(ExecConfig("remote", 0, 24000)).OK());
  std::vector<uint8_t> remote_kv(kNumBlocks * kBlockSize, 0);
  ASSERT_TRUE(remote.RegisterKvPool(remote_kv.data(), remote_kv.size(), kBlockSize, kNumBlocks, kNumLayers).OK());

  ASSERT_TRUE(sender.Initialize(ExecConfig("sender", 0, 24001)).OK());
  std::vector<uint8_t> sender_kv(kNumBlocks * kBlockSize, 0);
  ASSERT_TRUE(sender.RegisterKvPool(sender_kv.data(), sender_kv.size(), kBlockSize, kNumBlocks, kNumLayers).OK());

  std::vector<uint8_t> src(kNumBlocks * kBlockSize);
  for (size_t i = 0; i < src.size(); ++i) src[i] = static_cast<uint8_t>((i * 11 + 1) & 0xff);
  std::vector<void*> src_ptrs = {src.data(), src.data() + kBlockSize};

  ExecutorRequestManager rm;
  // Same request object drives both forward steps; only is_last_prefill_step flips.
  auto req =
      BuildReq(/*req_id=*/505, /*is_last_prefill_step=*/false, reinterpret_cast<uint64_t>(remote_kv.data()), src_ptrs);
  rm.AddInferRequest(req->req_id, req);

  PdV2KvDispatcher dispatcher(MakeDeps(&sender, &rm));

  // Epoch 1 — non-final chunk: deferred. Nothing written or tracked, and the
  // final-layer callback seals the (empty) epoch entry so none linger.
  dispatcher.CaptureEpochReqs(/*epoch_id=*/1, {req});
  dispatcher.OnLayerComplete(/*rank=*/0, /*epoch_id=*/1, /*layer_index=*/kNumLayers - 1);
  ASSERT_TRUE(dispatcher.WaitUntilIdleForTest());
  EXPECT_TRUE(sender.GetInflightBatchIds(505).empty());
  std::vector<uint8_t> zero(remote_kv.size(), 0);
  EXPECT_EQ(std::memcmp(remote_kv.data(), zero.data(), remote_kv.size()), 0);
  EXPECT_EQ(dispatcher.PendingEpochCountForTest(), 0u);  // epoch 1 sealed

  // Epoch 2 — same req, now the final chunk: fresh plans, full transfer.
  req->is_last_prefill_step = true;
  dispatcher.CaptureEpochReqs(/*epoch_id=*/2, {req});
  dispatcher.OnLayerComplete(/*rank=*/0, /*epoch_id=*/2, /*layer_index=*/kNumLayers - 1);
  ASSERT_TRUE(dispatcher.WaitUntilIdleForTest());
  EXPECT_FALSE(sender.GetInflightBatchIds(505).empty());
  EXPECT_EQ(std::memcmp(remote_kv.data(), src.data(), src.size()), 0);
  EXPECT_EQ(dispatcher.PendingEpochCountForTest(), 0u);  // epoch 2 sealed too
}

// OnEpochEnd empty-plans reclaim (split-fuse mid-forward abort path). With
// transfer_layer_chunk_size=1 there are intermediate chunk-end layers, so a
// non-final chunk whose forward ABORTS before the final layer leaves an
// EMPTY-plans epoch entry in dispatch_plans_by_epoch_. OnEpochEnd (the
// tracker's guaranteed cleanup) must reclaim it as a benign no-op — no RDMA,
// no crash, no barrier needed — and leave nothing behind, without being
// mistaken for a genuine epoch leak.
TEST_F(SplitFuseTransferFakeTest, OnEpochEndReclaimsAbortedNonFinalEpoch) {
  MooncakeExecutorConnector sender;
  ASSERT_TRUE(sender.Initialize(ExecConfig("sender", 0, 24001)).OK());
  std::vector<uint8_t> sender_kv(kNumBlocks * kBlockSize, 0);
  ASSERT_TRUE(sender.RegisterKvPool(sender_kv.data(), sender_kv.size(), kBlockSize, kNumBlocks, kNumLayers).OK());

  std::vector<uint8_t> src(kNumBlocks * kBlockSize, 0);
  std::vector<void*> src_ptrs = {src.data(), src.data() + kBlockSize};

  ExecutorRequestManager rm;
  auto req = BuildReq(/*req_id=*/606, /*is_last_prefill_step=*/false, /*remote_base=*/0, src_ptrs);
  rm.AddInferRequest(req->req_id, req);

  // chunk_size=1 → layer 0 is an intermediate chunk-end (not the final layer).
  PdV2KvDispatcher dispatcher(MakeDeps(&sender, &rm, /*transfer_layer_chunk_size=*/1));

  // Forward reaches the first chunk-end (layer 0) then aborts: the epoch entry
  // is built (empty plans, non-final chunk) but the final-layer erase never runs.
  dispatcher.CaptureEpochReqs(/*epoch_id=*/7, {req});
  dispatcher.OnLayerComplete(/*rank=*/0, /*epoch_id=*/7, /*layer_index=*/0);
  ASSERT_TRUE(dispatcher.WaitUntilIdleForTest());
  EXPECT_EQ(dispatcher.PendingEpochCountForTest(), 1u);  // residual empty-plans epoch
  EXPECT_TRUE(sender.GetInflightBatchIds(606).empty());  // non-final: no write

  // OnEpochEnd reclaims the empty-plans epoch (benign no-op, barrier==nullptr).
  dispatcher.OnEpochEnd(/*rank=*/0, /*epoch_id=*/7);
  ASSERT_TRUE(dispatcher.WaitUntilIdleForTest());
  EXPECT_EQ(dispatcher.PendingEpochCountForTest(), 0u);
  EXPECT_TRUE(sender.GetInflightBatchIds(606).empty());

  // Idempotent: a second OnEpochEnd for the already-reclaimed epoch is a no-op.
  dispatcher.OnEpochEnd(/*rank=*/0, /*epoch_id=*/7);
  ASSERT_TRUE(dispatcher.WaitUntilIdleForTest());
  EXPECT_EQ(dispatcher.PendingEpochCountForTest(), 0u);
}

// Barrier → IpcFuncWrapper::ReportPdV2BarrierDone coverage (kOk path). Drives a
// real PdV2KvBarrier over the same fake-tent connector: after the final chunk
// transfers, the tracked batch polls as completed, the barrier drains it and
// reports kOk (status=0) exactly once for the request. Locks in the fake sink's
// recording of req_id / status / count added for split-fuse barrier wiring.
TEST_F(SplitFuseTransferFakeTest, BarrierReportsOkAfterFinalChunkTransfer) {
  MooncakeExecutorConnector remote, sender;
  ASSERT_TRUE(remote.Initialize(ExecConfig("remote", 0, 24000)).OK());
  std::vector<uint8_t> remote_kv(kNumBlocks * kBlockSize, 0);
  ASSERT_TRUE(remote.RegisterKvPool(remote_kv.data(), remote_kv.size(), kBlockSize, kNumBlocks, kNumLayers).OK());

  ASSERT_TRUE(sender.Initialize(ExecConfig("sender", 0, 24001)).OK());
  std::vector<uint8_t> sender_kv(kNumBlocks * kBlockSize, 0);
  ASSERT_TRUE(sender.RegisterKvPool(sender_kv.data(), sender_kv.size(), kBlockSize, kNumBlocks, kNumLayers).OK());

  std::vector<uint8_t> src(kNumBlocks * kBlockSize);
  for (size_t i = 0; i < src.size(); ++i) src[i] = static_cast<uint8_t>((i * 13 + 5) & 0xff);
  std::vector<void*> src_ptrs = {src.data(), src.data() + kBlockSize};

  ExecutorRequestManager rm;
  auto req =
      BuildReq(/*req_id=*/303, /*is_last_prefill_step=*/true, reinterpret_cast<uint64_t>(remote_kv.data()), src_ptrs);
  rm.AddInferRequest(req->req_id, req);

  PdV2KvDispatcher dispatcher(MakeDeps(&sender, &rm));
  dispatcher.CaptureEpochReqs(/*epoch_id=*/1, {req});
  dispatcher.OnLayerComplete(/*rank=*/0, /*epoch_id=*/1, /*layer_index=*/kNumLayers - 1);
  ASSERT_TRUE(dispatcher.WaitUntilIdleForTest());
  ASSERT_FALSE(sender.GetInflightBatchIds(303).empty());

  // Wire a real barrier at the fake IpcFuncWrapper sink and drain the request.
  IpcFuncWrapper ipc;
  PdV2KvBarrier::Deps bd;
  bd.connector = &sender;
  bd.ipc_func_wrapper = &ipc;
  bd.poll_interval = std::chrono::microseconds(50);
  bd.per_req_timeout = std::chrono::milliseconds(2000);
  PdV2KvBarrier barrier(bd);
  barrier.WaitAndNotify(303);

  ASSERT_TRUE(WaitForBarrierOutcomes(barrier, 1));
  EXPECT_EQ(barrier.CompletedCountForTest(), 1u);
  EXPECT_EQ(barrier.FailedCountForTest(), 0u);
  // Fake sink recorded exactly one kOk report for this req.
  EXPECT_EQ(ipc.barrier_report_count_, 1);
  EXPECT_EQ(ipc.last_barrier_req_id_, 303);
  EXPECT_EQ(ipc.last_barrier_status_, 0);  // kBarrierStatusOk
}

// Barrier → ReportPdV2BarrierDone coverage (kFailed path). A non-final chunk
// defers (no in-flight batch), but dispatch_failed=true seeds a failed barrier
// status so the fake sink records status=1 for the request — the partial-submit
// failure surface split-fuse relies on.
TEST_F(SplitFuseTransferFakeTest, BarrierReportsFailedOnDispatchFailure) {
  MooncakeExecutorConnector sender;
  ASSERT_TRUE(sender.Initialize(ExecConfig("sender", 0, 24001)).OK());
  std::vector<uint8_t> sender_kv(kNumBlocks * kBlockSize, 0);
  ASSERT_TRUE(sender.RegisterKvPool(sender_kv.data(), sender_kv.size(), kBlockSize, kNumBlocks, kNumLayers).OK());

  // No in-flight batch for this req (nothing transferred): the barrier drains
  // immediately and only the dispatch_failed seed decides the reported status.
  ASSERT_TRUE(sender.GetInflightBatchIds(404).empty());

  IpcFuncWrapper ipc;
  PdV2KvBarrier::Deps bd;
  bd.connector = &sender;
  bd.ipc_func_wrapper = &ipc;
  bd.poll_interval = std::chrono::microseconds(50);
  bd.per_req_timeout = std::chrono::milliseconds(2000);
  PdV2KvBarrier barrier(bd);
  barrier.WaitAndNotify(404, /*dispatch_failed=*/true);

  ASSERT_TRUE(WaitForBarrierOutcomes(barrier, 1));
  EXPECT_EQ(barrier.FailedCountForTest(), 1u);
  EXPECT_EQ(barrier.CompletedCountForTest(), 0u);
  EXPECT_EQ(ipc.barrier_report_count_, 1);
  EXPECT_EQ(ipc.last_barrier_req_id_, 404);
  EXPECT_EQ(ipc.last_barrier_status_, 1);  // kBarrierStatusFailed
}

}  // namespace pd_v2
}  // namespace ksana_llm
