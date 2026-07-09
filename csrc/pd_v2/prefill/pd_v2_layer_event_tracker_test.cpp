/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "pd_v2/prefill/pd_v2_layer_event_tracker.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <mutex>
#include <set>
#include <thread>
#include <utility>
#include <vector>

#include "device/device_utils.h"

namespace ksana_llm {
namespace pd_v2 {

class PdV2LayerEventTrackerTest : public ::testing::Test {
 protected:
  void SetUp() override { SetDevice(0); }
  void TearDown() override { DeviceSynchronize(); }
};

// Collects callback fires (epoch_id, layer_index) thread-safely. Both the
// monitor thread and ResetState's force-fire path push here.
struct FireSink {
  std::mutex mu;
  std::vector<std::pair<uint64_t, int>> fires;
  int rank_seen = -1;

  PdV2LayerEventTracker::LayerCallback Cb() {
    return [this](int rank, uint64_t epoch, int layer) {
      std::lock_guard<std::mutex> lk(mu);
      rank_seen = rank;
      fires.emplace_back(epoch, layer);
    };
  }
  std::set<int> LayersForEpoch(uint64_t epoch) {
    std::lock_guard<std::mutex> lk(mu);
    std::set<int> s;
    for (auto& f : fires)
      if (f.first == epoch) s.insert(f.second);
    return s;
  }
  // Poll until epoch's fired layers equal `want` (or timeout). ResetState
  // guarantees only *eventual* delivery: the monitor thread may have claimed
  // a layer (processed=true) but not yet run its callback out-of-lock when
  // ResetState returns, so reading the sink synchronously races that fire.
  // This is benign in production (decode just waits µs longer); tests must
  // poll. Returns true if the set matched before the deadline.
  bool WaitLayersForEpoch(uint64_t epoch, const std::set<int>& want,
                          std::chrono::milliseconds timeout = std::chrono::milliseconds(2000)) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (true) {
      if (LayersForEpoch(epoch) == want) return true;
      if (std::chrono::steady_clock::now() >= deadline) return false;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
  size_t CountForEpochLayer(uint64_t epoch, int layer) {
    std::lock_guard<std::mutex> lk(mu);
    return std::count(fires.begin(), fires.end(), std::make_pair(epoch, layer));
  }
};

// Not active before Initialize: BeginBatch returns 0, Record/Reset are no-ops.
TEST_F(PdV2LayerEventTrackerTest, InactiveBeforeInitialize) {
  PdV2LayerEventTracker tracker;
  EXPECT_FALSE(tracker.IsActive());
  EXPECT_EQ(tracker.BeginBatch(), 0u);
  Stream stream;
  tracker.RecordLayerProgress(0, 0, stream);  // no crash
  tracker.ResetState(0);                       // epoch 0 -> no-op
  stream.Destroy();
}

// Initialize is idempotent; double-init is ignored, IsActive flips true.
TEST_F(PdV2LayerEventTrackerTest, InitializeIdempotentAndCleanup) {
  PdV2LayerEventTracker tracker;
  tracker.Initialize(/*rank=*/0, /*max_layers=*/4);
  EXPECT_TRUE(tracker.IsActive());
  tracker.Initialize(0, 99);  // second call ignored
  EXPECT_TRUE(tracker.IsActive());
  tracker.Cleanup();
  EXPECT_FALSE(tracker.IsActive());
  // Re-initialize after cleanup works.
  tracker.Initialize(0, 4);
  EXPECT_TRUE(tracker.IsActive());
  tracker.Cleanup();
}

// BeginBatch hands out monotonically increasing, non-zero epoch ids.
TEST_F(PdV2LayerEventTrackerTest, BeginBatchEpochIdsMonotonic) {
  PdV2LayerEventTracker tracker;
  tracker.Initialize(0, 4);
  uint64_t e1 = tracker.BeginBatch();
  uint64_t e2 = tracker.BeginBatch();
  uint64_t e3 = tracker.BeginBatch();
  EXPECT_GT(e1, 0u);
  EXPECT_GT(e2, e1);
  EXPECT_GT(e3, e2);
  tracker.ResetState(e1);
  tracker.ResetState(e2);
  tracker.ResetState(e3);
  tracker.Cleanup();
}

// Each recorded layer fires its callback exactly once (monitor or
// ResetState force-fire), with the right epoch_id and rank.
TEST_F(PdV2LayerEventTrackerTest, RecordAllLayersFireOncePerLayer) {
  const int kRank = 0;
  const int kLayers = 6;
  PdV2LayerEventTracker tracker;
  tracker.Initialize(kRank, kLayers);
  FireSink sink;
  tracker.RegisterCallback(sink.Cb());

  uint64_t epoch = tracker.BeginBatch();
  Stream stream;
  for (int i = 0; i < kLayers; ++i) {
    tracker.RecordLayerProgress(kRank, i, stream);
  }
  StreamSynchronize(stream);
  // ResetState force-fires anything the monitor missed; combined with the
  // monitor this guarantees every recorded layer fired.
  tracker.ResetState(epoch);

  EXPECT_TRUE(sink.WaitLayersForEpoch(epoch, (std::set<int>{0, 1, 2, 3, 4, 5})));
  for (int i = 0; i < kLayers; ++i) {
    EXPECT_EQ(sink.CountForEpochLayer(epoch, i), 1u) << "layer " << i << " fired wrong number of times";
  }
  EXPECT_EQ(sink.rank_seen, kRank);

  stream.Destroy();
  tracker.Cleanup();
}

// Multiple callbacks all fan out.
TEST_F(PdV2LayerEventTrackerTest, MultipleCallbacksAllFire) {
  PdV2LayerEventTracker tracker;
  tracker.Initialize(0, 2);
  FireSink a, b;
  tracker.RegisterCallback(a.Cb());
  tracker.RegisterCallback(b.Cb());
  uint64_t epoch = tracker.BeginBatch();
  Stream stream;
  tracker.RecordLayerProgress(0, 0, stream);
  tracker.RecordLayerProgress(0, 1, stream);
  StreamSynchronize(stream);
  tracker.ResetState(epoch);
  EXPECT_TRUE(a.WaitLayersForEpoch(epoch, (std::set<int>{0, 1})));
  EXPECT_TRUE(b.WaitLayersForEpoch(epoch, (std::set<int>{0, 1})));
  stream.Destroy();
  tracker.Cleanup();
}

// RecordLayerProgress with rank mismatch is ignored (logged error, no fire).
TEST_F(PdV2LayerEventTrackerTest, RankMismatchIgnored) {
  PdV2LayerEventTracker tracker;
  tracker.Initialize(/*rank=*/0, 4);
  FireSink sink;
  tracker.RegisterCallback(sink.Cb());
  uint64_t epoch = tracker.BeginBatch();
  Stream stream;
  tracker.RecordLayerProgress(/*rank=*/7, 0, stream);  // wrong rank
  StreamSynchronize(stream);
  tracker.ResetState(epoch);
  EXPECT_TRUE(sink.LayersForEpoch(epoch).empty());
  stream.Destroy();
  tracker.Cleanup();
}

// Out-of-range layer index is silently dropped, no crash.
TEST_F(PdV2LayerEventTrackerTest, OutOfRangeLayerDropped) {
  PdV2LayerEventTracker tracker;
  tracker.Initialize(0, 4);
  FireSink sink;
  tracker.RegisterCallback(sink.Cb());
  uint64_t epoch = tracker.BeginBatch();
  Stream stream;
  tracker.RecordLayerProgress(0, -1, stream);
  tracker.RecordLayerProgress(0, 99, stream);
  tracker.RecordLayerProgress(0, 2, stream);  // valid
  StreamSynchronize(stream);
  tracker.ResetState(epoch);
  EXPECT_TRUE(sink.WaitLayersForEpoch(epoch, (std::set<int>{2})));
  stream.Destroy();
  tracker.Cleanup();
}

// RecordLayerProgress before any BeginBatch is rejected (no current slot).
TEST_F(PdV2LayerEventTrackerTest, RecordWithoutBeginBatchRejected) {
  PdV2LayerEventTracker tracker;
  tracker.Initialize(0, 4);
  FireSink sink;
  tracker.RegisterCallback(sink.Cb());
  Stream stream;
  // No BeginBatch -> current_write_epoch_slot_ is null.
  tracker.RecordLayerProgress(0, 0, stream);
  StreamSynchronize(stream);
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  std::lock_guard<std::mutex> lk(sink.mu);
  EXPECT_TRUE(sink.fires.empty());
  stream.Destroy();
  tracker.Cleanup();
}

// ResetState for an unknown epoch is a no-op (logged error), no crash.
TEST_F(PdV2LayerEventTrackerTest, ResetUnknownEpochNoOp) {
  PdV2LayerEventTracker tracker;
  tracker.Initialize(0, 4);
  tracker.ResetState(/*epoch_id=*/424242);  // never begun
  EXPECT_TRUE(tracker.IsActive());
  tracker.Cleanup();
}

// Slot pool exhaustion path: open more concurrent batches than the initial
// pool (8) to force AcquireFreeSlotLocked's grow branch.
TEST_F(PdV2LayerEventTrackerTest, SlotPoolGrowsWhenExhausted) {
  PdV2LayerEventTracker tracker;
  tracker.Initialize(0, 2);
  std::vector<uint64_t> epochs;
  for (int i = 0; i < 12; ++i) {  // > initial pool of 8
    uint64_t e = tracker.BeginBatch();
    EXPECT_GT(e, 0u);
    epochs.push_back(e);
  }
  for (uint64_t e : epochs) tracker.ResetState(e);
  tracker.Cleanup();
}

// Two concurrently-active batches keep independent slots; each epoch's
// callbacks carry its own epoch_id.
TEST_F(PdV2LayerEventTrackerTest, ConcurrentBatchesIndependent) {
  const int kLayers = 3;
  PdV2LayerEventTracker tracker;
  tracker.Initialize(0, kLayers);
  FireSink sink;
  tracker.RegisterCallback(sink.Cb());

  Stream stream;
  uint64_t e1 = tracker.BeginBatch();
  tracker.RecordLayerProgress(0, 0, stream);
  tracker.RecordLayerProgress(0, 1, stream);
  // BeginBatch again switches the write slot to e2 (mirrors T2(C) opening a
  // new batch); e1's already-recorded events stay in e1's slot.
  uint64_t e2 = tracker.BeginBatch();
  tracker.RecordLayerProgress(0, 0, stream);
  StreamSynchronize(stream);
  tracker.ResetState(e1);
  tracker.ResetState(e2);

  EXPECT_TRUE(sink.WaitLayersForEpoch(e1, (std::set<int>{0, 1})));
  EXPECT_TRUE(sink.WaitLayersForEpoch(e2, (std::set<int>{0})));
  stream.Destroy();
  tracker.Cleanup();
}

}  // namespace pd_v2
}  // namespace ksana_llm
