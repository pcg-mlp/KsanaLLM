/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "pd_v2/prefill/pd_v2_kv_barrier.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "pd_v2/public/pd_v2_connector.h"
#include "utils/ret_code.h"

namespace ksana_llm {

// Stub so the test links without pulling the heavy `ipc` library (and its
// transitive executor/distributed/cache_manager deps). PdV2KvBarrier only
// ever calls ReportPdV2BarrierDone, and only when ipc_func_wrapper != nullptr
// — every test below passes nullptr, so this stub is never invoked. It exists
// purely to satisfy the linker for the one symbol kv_barrier.cpp references.
class IpcFuncWrapper {
 public:
  Status ReportPdV2BarrierDone(int64_t /*req_id*/, uint8_t /*status*/) { return Status(); }
};

namespace pd_v2 {
namespace {

// Connector double driving the barrier's poll loop. Only the four barrier
// surface methods are meaningful; everything else is a no-op default (the
// interface defaults most of them) or a trivial stub for the pure virtuals.
class FakeBarrierConnector : public PdV2ConnectorInterface {
 public:
  // ---- barrier surface ----
  std::vector<uint64_t> GetInflightBatchIds(uint64_t req_id) const override {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = inflight_.find(req_id);
    if (it == inflight_.end()) return {};
    return it->second;
  }

  Status PollBatchStatus(uint64_t batch_id, bool* out_completed, bool* out_failed) override {
    std::lock_guard<std::mutex> lk(mu_);
    *out_completed = false;
    *out_failed = false;
    if (poll_returns_error_) return Status(RET_RUNTIME_FAILED, "poll boom");
    if (failing_batches_.count(batch_id)) {
      *out_failed = true;
      return Status();
    }
    if (completed_batches_.count(batch_id) || complete_all_) {
      *out_completed = true;
    }
    // else: still in flight (both false)
    return Status();
  }

  void EraseInflightBatch(uint64_t req_id, uint64_t batch_id) override {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = inflight_.find(req_id);
    if (it == inflight_.end()) return;
    auto& v = it->second;
    v.erase(std::remove(v.begin(), v.end(), batch_id), v.end());
    if (v.empty()) inflight_.erase(it);
  }

  // ---- test setup helpers ----
  void SetInflight(uint64_t req_id, std::vector<uint64_t> batches) {
    std::lock_guard<std::mutex> lk(mu_);
    inflight_[req_id] = std::move(batches);
  }
  void MarkCompleted(uint64_t batch_id) {
    std::lock_guard<std::mutex> lk(mu_);
    completed_batches_.insert(batch_id);
  }
  void MarkFailing(uint64_t batch_id) {
    std::lock_guard<std::mutex> lk(mu_);
    failing_batches_.insert(batch_id);
  }
  void CompleteAll() {
    std::lock_guard<std::mutex> lk(mu_);
    complete_all_ = true;
  }
  void SetPollError() {
    std::lock_guard<std::mutex> lk(mu_);
    poll_returns_error_ = true;
  }

  // ---- unused pure virtuals ----
  Status Initialize(const PdV2ConnectorConfig&) override { return Status(); }
  Status Shutdown() override { return Status(); }
  bool CanAccept(uint64_t, uint32_t) override { return true; }
  Status SubmitPrefillRequest(uint64_t, const std::vector<int>&, const std::vector<int>&, const SamplingConfigWire&,
                              uint32_t, uint32_t, uint32_t, PdRequestCompleteCb, void*) override {
    return Status();
  }
  Status Release(uint64_t) override { return Status(); }
  void RegisterIncomingHook(PdIncomingReqCb, void*) override {}
  Status SendPrefillComplete(const std::string&, uint64_t, PdCompleteStatus, const std::vector<int>&, const PeerHealth&,
                             const PdPrefillMetrics&) override {
    return Status();
  }
  Status RegisterKvPool(void*, size_t, size_t, size_t, size_t) override { return Status(); }
  Status GetKvBufferVa(int, uint64_t*, uint64_t*) override { return Status(); }

 private:
  mutable std::mutex mu_;
  std::map<uint64_t, std::vector<uint64_t>> inflight_;
  std::set<uint64_t> completed_batches_;
  std::set<uint64_t> failing_batches_;
  bool complete_all_ = false;
  bool poll_returns_error_ = false;
};

PdV2KvBarrier::Deps MakeDeps(FakeBarrierConnector* c) {
  PdV2KvBarrier::Deps d;
  d.connector = c;
  d.ipc_func_wrapper = nullptr;  // stub never called
  d.poll_interval = std::chrono::microseconds(50);
  d.per_req_timeout = std::chrono::milliseconds(2000);
  return d;
}

// Spin until completed+failed counts reach `total`, or timeout.
bool WaitForOutcomes(const PdV2KvBarrier& b, size_t total, int max_ms = 3000) {
  for (int i = 0; i < max_ms; ++i) {
    if (b.CompletedCountForTest() + b.FailedCountForTest() >= total) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return false;
}

}  // namespace

// All batches already completed -> req drains to kOk (completed_count++).
TEST(PdV2KvBarrierTest, AllCompletedReportsOk) {
  FakeBarrierConnector c;
  c.SetInflight(/*req=*/1, {10, 11, 12});
  c.CompleteAll();
  PdV2KvBarrier barrier(MakeDeps(&c));
  barrier.WaitAndNotify(1);
  ASSERT_TRUE(WaitForOutcomes(barrier, 1));
  EXPECT_EQ(barrier.CompletedCountForTest(), 1u);
  EXPECT_EQ(barrier.FailedCountForTest(), 0u);
}

// No in-flight batches (e.g. full prefix-cache hit on Decode) -> immediate kOk.
// Must still report, not silently skip.
TEST(PdV2KvBarrierTest, EmptyInflightReportsOk) {
  FakeBarrierConnector c;
  PdV2KvBarrier barrier(MakeDeps(&c));
  barrier.WaitAndNotify(/*req=*/2);
  ASSERT_TRUE(WaitForOutcomes(barrier, 1));
  EXPECT_EQ(barrier.CompletedCountForTest(), 1u);
  EXPECT_EQ(barrier.FailedCountForTest(), 0u);
}

// A batch reporting failed -> kFailed.
TEST(PdV2KvBarrierTest, FailedBatchReportsFailed) {
  FakeBarrierConnector c;
  c.SetInflight(/*req=*/3, {20, 21});
  c.MarkCompleted(20);
  c.MarkFailing(21);
  PdV2KvBarrier barrier(MakeDeps(&c));
  barrier.WaitAndNotify(3);
  ASSERT_TRUE(WaitForOutcomes(barrier, 1));
  EXPECT_EQ(barrier.FailedCountForTest(), 1u);
  EXPECT_EQ(barrier.CompletedCountForTest(), 0u);
}

// PollBatchStatus itself returning non-OK -> conservative kFailed.
TEST(PdV2KvBarrierTest, PollErrorReportsFailed) {
  FakeBarrierConnector c;
  c.SetInflight(/*req=*/4, {30});
  c.SetPollError();
  PdV2KvBarrier barrier(MakeDeps(&c));
  barrier.WaitAndNotify(4);
  ASSERT_TRUE(WaitForOutcomes(barrier, 1));
  EXPECT_EQ(barrier.FailedCountForTest(), 1u);
}

// dispatch_failed=true seeds kFailed even though every (submit-succeeded)
// batch eventually completes. This surfaces partial-submit failures.
TEST(PdV2KvBarrierTest, DispatchFailedSeedsFailedEvenIfBatchesComplete) {
  FakeBarrierConnector c;
  c.SetInflight(/*req=*/5, {40});
  c.CompleteAll();
  PdV2KvBarrier barrier(MakeDeps(&c));
  barrier.WaitAndNotify(5, /*dispatch_failed=*/true);
  ASSERT_TRUE(WaitForOutcomes(barrier, 1));
  EXPECT_EQ(barrier.FailedCountForTest(), 1u);
  EXPECT_EQ(barrier.CompletedCountForTest(), 0u);
}

// Batches that never complete -> per_req_timeout fires kFailed.
TEST(PdV2KvBarrierTest, TimeoutReportsFailed) {
  FakeBarrierConnector c;
  c.SetInflight(/*req=*/6, {50});  // never marked complete/failing -> stuck in flight
  PdV2KvBarrier::Deps d = MakeDeps(&c);
  d.per_req_timeout = std::chrono::milliseconds(30);
  PdV2KvBarrier barrier(d);
  barrier.WaitAndNotify(6);
  ASSERT_TRUE(WaitForOutcomes(barrier, 1));
  EXPECT_EQ(barrier.FailedCountForTest(), 1u);
}

// Null connector: WaitAndNotify is a no-op, no outcome ever recorded.
TEST(PdV2KvBarrierTest, NullConnectorIsNoOp) {
  PdV2KvBarrier::Deps d;
  d.connector = nullptr;
  d.ipc_func_wrapper = nullptr;
  PdV2KvBarrier barrier(d);
  barrier.WaitAndNotify(7);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_EQ(barrier.CompletedCountForTest(), 0u);
  EXPECT_EQ(barrier.FailedCountForTest(), 0u);
}

// Multiple requests are each processed independently and FIFO-drained.
TEST(PdV2KvBarrierTest, MultipleRequestsAllReported) {
  FakeBarrierConnector c;
  c.CompleteAll();
  c.SetInflight(1, {1});
  c.SetInflight(2, {2});
  c.SetInflight(3, {3});
  PdV2KvBarrier barrier(MakeDeps(&c));
  barrier.WaitAndNotify(1);
  barrier.WaitAndNotify(2);
  barrier.WaitAndNotify(3);
  ASSERT_TRUE(WaitForOutcomes(barrier, 3));
  EXPECT_EQ(barrier.CompletedCountForTest(), 3u);
  EXPECT_EQ(barrier.FailedCountForTest(), 0u);
}

}  // namespace pd_v2
}  // namespace ksana_llm
