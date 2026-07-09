/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "device/device_utils.h"

namespace ksana_llm {
namespace pd_v2 {

// pd_v2-only per-layer cudaEvent tracker for the Prefill executor.
//
// Why this is separate from `csrc/executor/layer_progress_tracker.{h,cpp}`
// (the V1 tracker):
//   - V1 tracker is gated by env ENABLE_LAYER_TRACKER=1 (off by default
//     to dodge a deadlock); flipping that on for pd_v2 would change V1
//     scheduler timing.
//   - V1 tracker has exactly one consumer (IPC ReportLayerProgress);
//     adding a callback list to it would invade V1 hot path code.
//
// Lifecycle (driven by InferenceExecutor):
//   1. Initialize(rank, num_layers) — allocates a pool of EpochSlots
//      (each owning per-layer cudaEvents) + starts monitor thread.
//   2. RegisterCallback(cb) — multiple callbacks fan out on the monitor
//      thread; signature is (rank, epoch_id, layer_index) so the consumer
//      can group per-batch state.
//   3. BeginBatch() — call from HandleExecute (T2) before the first
//      RecordLayerProgress of the batch. Returns a fresh epoch_id and
//      atomically sets `current_write_epoch_` so subsequent
//      RecordLayerProgress calls land in the right slot.
//   4. RecordLayerProgress(rank, layer_index, stream) — called by attention
//      layer post K/V cache write. Writes into current_write_epoch_'s
//      events. Signature unchanged so attention kernels don't need to
//      know about epochs.
//   5. ResetState(epoch_id) — called from HandlePostprocess (T3) with the
//      epoch_id threaded through NewModelOutput. Force-fires any pending
//      callbacks for that epoch (mirrors LayerProgressTracker's force-
//      report pattern) then returns the slot to the free pool.
//   6. Cleanup() — torn down by InferenceExecutor on shutdown.
//
// Why per-batch slots: the previous single-events_ design had T3(B) and
// T2(C) racing on a shared events_ vector — T2(C) RecordLayerProgress
// could overwrite events_[i] before T4 monitor had fired the callback for
// batch B's layer i, silently dropping that callback (`unprocessed_at_wipe`
// >0 in diag) and hanging the decode peer at pd_v2_awaiting_prefill until
// 120s timeout. With per-batch EpochSlot, T2(C) and T3(B) operate on
// different slots and cannot interfere. ResetState's force-fire fallback
// closes the residual case where the monitor lagged within a batch.
class PdV2LayerEventTracker {
 public:
  using LayerCallback = std::function<void(int rank, uint64_t epoch_id, int layer_index)>;
  // Fired exactly once per epoch when its slot is sealed in ResetState,
  // AFTER all layer callbacks (including force-fired ones) have run. Unlike
  // the layer callback it fires unconditionally — even when Forward was
  // aborted mid-pass and the final-layer event never fired — so consumers
  // can reclaim any per-epoch state keyed by epoch_id (e.g. the dispatcher's
  // dispatch_plans_by_epoch_ strong refs) instead of leaking it.
  using EpochEndCallback = std::function<void(int rank, uint64_t epoch_id)>;

  PdV2LayerEventTracker();
  ~PdV2LayerEventTracker();

  PdV2LayerEventTracker(const PdV2LayerEventTracker&) = delete;
  PdV2LayerEventTracker& operator=(const PdV2LayerEventTracker&) = delete;

  // Boot-time. Pre-allocates a pool of EpochSlots (each with
  // num_layers cudaEvents) + spawns monitor thread. Idempotent.
  void Initialize(int rank, int max_layers);

  // Tear-down. Stops monitor thread + destroys all events.
  void Cleanup();

  bool IsActive() const { return initialized_.load(std::memory_order_acquire); }

  // Register a per-layer callback. Callbacks fire on the monitor thread.
  void RegisterCallback(LayerCallback cb);

  // Register an epoch-end callback. Fires once per epoch from ResetState
  // (the guaranteed seal point), after all layer callbacks for that epoch.
  void RegisterEpochEndCallback(EpochEndCallback cb);

  // T2: open a fresh write slot and return its epoch_id. The same epoch_id
  // must be passed to ResetState in the matching HandlePostprocess call.
  // No-op (returns 0) if tracker not active.
  uint64_t BeginBatch();

  // T2: record a layer event into the current write slot (set by the
  // most recent BeginBatch). Signature kept stable so attention kernels
  // don't need to know about epochs.
  void RecordLayerProgress(int rank, int layer_index, Stream& stream);

  // T3: force-fire pending callbacks for `epoch_id`, then seal that slot
  // back to the free pool. Force-fire is safe because ResetState is called
  // post-Forward, so all GPU events of this epoch are reached on device.
  void ResetState(uint64_t epoch_id);

 private:
  void MonitorLoop();

  struct EventInfo {
    Event event;
    bool processed = true;  // initial: nothing to monitor
  };

  // One slot per concurrently-active batch. T2 writes its current batch's
  // RecordLayerProgress events into the slot pointed at by `current_write_epoch_slot_`.
  // T3 force-fires + seals the slot for the epoch it was given. Different
  // batches always map to different slots, so T2(C) and T3(B) never touch
  // the same slot.
  struct EpochSlot {
    std::vector<EventInfo> events;  // size = num_layers
    int current_progress = -1;      // last layer fired by monitor (per-slot)
    int max_recorded = -1;          // highest layer recorded (per-slot)
    uint64_t epoch_id = 0;          // 0 means free
    bool active = false;            // true between BeginBatch and ResetState
  };

  EpochSlot* AcquireFreeSlotLocked();                // pool pop or grow; mu_ held by caller
  void DestroyEventsForSlotLocked(EpochSlot& slot);  // mu_ held by caller

  std::atomic<bool> initialized_{false};
  std::atomic<bool> monitor_running_{false};

  mutable std::mutex mu_;
  int rank_ = -1;
  int num_layers_ = 0;
  uint64_t next_epoch_id_ = 1;  // monotonic; 0 reserved for "no epoch"

  std::vector<std::unique_ptr<EpochSlot>> slot_pool_;         // owns all slots
  std::vector<EpochSlot*> free_slots_;                        // sealed, available
  std::unordered_map<uint64_t, EpochSlot*> active_by_epoch_;  // active, by epoch_id
  EpochSlot* current_write_epoch_slot_ = nullptr;             // T2's write target

  std::vector<LayerCallback> callbacks_;
  std::vector<EpochEndCallback> epoch_end_callbacks_;
  std::thread monitor_thread_;
};

}  // namespace pd_v2
}  // namespace ksana_llm
