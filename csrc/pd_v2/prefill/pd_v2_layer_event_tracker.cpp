/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "pd_v2/prefill/pd_v2_layer_event_tracker.h"

#include "utils/logger.h"

namespace ksana_llm {
namespace pd_v2 {

PdV2LayerEventTracker::PdV2LayerEventTracker() = default;

PdV2LayerEventTracker::~PdV2LayerEventTracker() { Cleanup(); }

void PdV2LayerEventTracker::Initialize(int rank, int max_layers) {
  if (initialized_.load(std::memory_order_acquire)) {
    KLLM_LOG_INFO << "PdV2LayerEventTracker::Initialize called twice; second call ignored";
    return;
  }
  std::lock_guard<std::mutex> lk(mu_);
  rank_ = rank;
  num_layers_ = max_layers;
  next_epoch_id_ = 1;

  // Pre-allocate a small pool. Pipeline depth in the executor is bounded
  // (preprocess / execute / postprocess / monitor — at most ~2 batches
  // concurrently in tracker land: T2 writing one + T3 resetting the
  // previous one). 8 leaves headroom for transient stalls; the pool grows
  // dynamically if exhausted, but pre-allocation keeps the steady-state
  // free of cudaEventCreate calls in the hot path.
  SetDevice(rank);
  constexpr int kInitialPoolSize = 8;
  slot_pool_.reserve(kInitialPoolSize);
  free_slots_.reserve(kInitialPoolSize);
  for (int s = 0; s < kInitialPoolSize; ++s) {
    auto slot = std::make_unique<EpochSlot>();
    slot->events.resize(max_layers);
    for (int i = 0; i < max_layers; ++i) {
      EventCreateWithFlags(&slot->events[i].event, EVENT_DISABLE_TIMING);
      slot->events[i].processed = true;
    }
    free_slots_.push_back(slot.get());
    slot_pool_.push_back(std::move(slot));
  }

  monitor_running_.store(true, std::memory_order_release);
  monitor_thread_ = std::thread(&PdV2LayerEventTracker::MonitorLoop, this);

  initialized_.store(true, std::memory_order_release);
  KLLM_LOG_INFO << "PdV2LayerEventTracker initialized rank=" << rank << " max_layers=" << max_layers
                << " initial_pool_size=" << kInitialPoolSize;
}

void PdV2LayerEventTracker::Cleanup() {
  if (!initialized_.load(std::memory_order_acquire)) return;

  monitor_running_.store(false, std::memory_order_release);
  if (monitor_thread_.joinable()) monitor_thread_.join();

  std::lock_guard<std::mutex> lk(mu_);
  for (auto& slot_uptr : slot_pool_) {
    DestroyEventsForSlotLocked(*slot_uptr);
  }
  slot_pool_.clear();
  free_slots_.clear();
  active_by_epoch_.clear();
  current_write_epoch_slot_ = nullptr;
  callbacks_.clear();
  epoch_end_callbacks_.clear();
  initialized_.store(false, std::memory_order_release);
  KLLM_LOG_INFO << "PdV2LayerEventTracker cleaned up rank=" << rank_;
}

void PdV2LayerEventTracker::DestroyEventsForSlotLocked(EpochSlot& slot) {
  for (auto& info : slot.events) {
    EventDestroy(info.event);
  }
  slot.events.clear();
}

void PdV2LayerEventTracker::RegisterCallback(LayerCallback cb) {
  std::lock_guard<std::mutex> lk(mu_);
  callbacks_.push_back(std::move(cb));
}

void PdV2LayerEventTracker::RegisterEpochEndCallback(EpochEndCallback cb) {
  std::lock_guard<std::mutex> lk(mu_);
  epoch_end_callbacks_.push_back(std::move(cb));
}

PdV2LayerEventTracker::EpochSlot* PdV2LayerEventTracker::AcquireFreeSlotLocked() {
  if (!free_slots_.empty()) {
    EpochSlot* slot = free_slots_.back();
    free_slots_.pop_back();
    return slot;
  }
  // Pool exhausted — grow. Rare path (happens only if pipeline depth
  // exceeded the initial pool size). cudaEventCreate is fast but allocates
  // on the device, hence the warning so the initial size can be tuned.
  KLLM_LOG_WARNING << "PdV2LayerEventTracker: slot pool exhausted (size=" << slot_pool_.size()
                   << "), allocating one more on demand";
  auto slot = std::make_unique<EpochSlot>();
  slot->events.resize(num_layers_);
  for (int i = 0; i < num_layers_; ++i) {
    EventCreateWithFlags(&slot->events[i].event, EVENT_DISABLE_TIMING);
    slot->events[i].processed = true;
  }
  EpochSlot* raw = slot.get();
  slot_pool_.push_back(std::move(slot));
  return raw;
}

uint64_t PdV2LayerEventTracker::BeginBatch() {
  if (!initialized_.load(std::memory_order_acquire)) return 0;
  std::lock_guard<std::mutex> lk(mu_);
  EpochSlot* slot = AcquireFreeSlotLocked();
  slot->current_progress = -1;
  slot->max_recorded = -1;
  slot->epoch_id = next_epoch_id_++;
  slot->active = true;
  for (auto& info : slot->events) info.processed = true;  // start clean
  current_write_epoch_slot_ = slot;
  active_by_epoch_[slot->epoch_id] = slot;
  return slot->epoch_id;
}

void PdV2LayerEventTracker::RecordLayerProgress(int rank, int layer_index, Stream& stream) {
  if (!initialized_.load(std::memory_order_acquire)) return;
  std::lock_guard<std::mutex> lk(mu_);
  if (rank != rank_) {
    KLLM_LOG_ERROR << "PdV2LayerEventTracker rank mismatch: expected=" << rank_ << " got=" << rank;
    return;
  }
  if (current_write_epoch_slot_ == nullptr) {
    KLLM_LOG_ERROR << "PdV2LayerEventTracker::RecordLayerProgress called without preceding BeginBatch (rank=" << rank
                   << " layer=" << layer_index << ")";
    return;
  }
  EpochSlot* slot = current_write_epoch_slot_;
  if (layer_index < 0 || layer_index >= static_cast<int>(slot->events.size())) {
    return;
  }
  EventRecord(slot->events[layer_index].event, stream);
  slot->events[layer_index].processed = false;
  if (layer_index > slot->max_recorded) slot->max_recorded = layer_index;
}

void PdV2LayerEventTracker::ResetState(uint64_t epoch_id) {
  if (!initialized_.load(std::memory_order_acquire)) return;
  if (epoch_id == 0) return;  // no epoch to reset (BeginBatch was a no-op)

  std::vector<int> force_fire_layers;
  std::vector<LayerCallback> cbs_snapshot;
  std::vector<EpochEndCallback> epoch_end_cbs_snapshot;
  int rank_snapshot = -1;
  int diag_progress = -1;
  int diag_recorded = -1;
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = active_by_epoch_.find(epoch_id);
    if (it == active_by_epoch_.end()) {
      KLLM_LOG_ERROR << "PdV2LayerEventTracker::ResetState: epoch_id=" << epoch_id << " not found in active_by_epoch_";
      return;
    }
    EpochSlot* slot = it->second;
    diag_progress = slot->current_progress;
    diag_recorded = slot->max_recorded;
    // Force-fire any layer the monitor never observed. ResetState runs in
    // HandlePostprocess after Forward done, so the GPU has reached every
    // recorded layer — a 补 fire is semantically safe. Mark processed=true
    // atomically so a concurrent MonitorLoop iteration (which checks
    // `if (info.processed) continue`) won't double-fire.
    for (int li = slot->current_progress + 1; li <= slot->max_recorded; ++li) {
      if (li < 0 || li >= static_cast<int>(slot->events.size())) continue;
      if (slot->events[li].processed) continue;
      slot->events[li].processed = true;
      force_fire_layers.push_back(li);
    }
    cbs_snapshot = callbacks_;
    epoch_end_cbs_snapshot = epoch_end_callbacks_;
    rank_snapshot = rank_;
    // Seal: return slot to free pool. Reset state so next BeginBatch finds
    // it clean.
    slot->active = false;
    slot->current_progress = -1;
    slot->max_recorded = -1;
    slot->epoch_id = 0;
    for (auto& info : slot->events) info.processed = true;
    if (current_write_epoch_slot_ == slot) {
      // Defensive: next BeginBatch will overwrite, but unsetting prevents
      // a misuse window where RecordLayerProgress runs between ResetState
      // and BeginBatch.
      current_write_epoch_slot_ = nullptr;
    }
    active_by_epoch_.erase(it);
    free_slots_.push_back(slot);
  }
  // Fire outside the lock (matches MonitorLoop pattern; dispatcher
  // callback may take ms+ on Mooncake submitTransfer).
  for (int li : force_fire_layers) {
    for (auto& cb : cbs_snapshot) cb(rank_snapshot, epoch_id, li);
  }
  // Epoch-end fires unconditionally AFTER every layer callback so consumers
  // reclaim per-epoch state even when Forward aborted before the final layer
  // (force_fire_layers empty) and the layer callback's own cleanup never ran.
  for (auto& cb : epoch_end_cbs_snapshot) cb(rank_snapshot, epoch_id);
  if (!force_fire_layers.empty()) {
    KLLM_LOG_WARNING << "PdV2LayerEventTracker::ResetState force-fired pending callbacks rank=" << rank_snapshot
                     << " epoch=" << epoch_id << " progress=" << diag_progress << " recorded=" << diag_recorded
                     << " force_fired=" << force_fire_layers.size();
  }
}

void PdV2LayerEventTracker::MonitorLoop() {
  // Monitor 线程新建后默认在 logical 0；下游 callback 会访问本 rank 的 GPU/KV pool，
  // 因此进入循环前需要切到对应设备。
  SetDevice(rank_);
  while (monitor_running_.load(std::memory_order_acquire)) {
    struct FireEntry {
      uint64_t epoch_id;
      int layer_index;
    };
    std::vector<FireEntry> just_completed;
    std::vector<LayerCallback> cbs_snapshot;
    int rank_snapshot = -1;
    {
      std::lock_guard<std::mutex> lk(mu_);
      // Walk all active slots. Each slot is independent — an event ready
      // in one batch doesn't depend on another batch. Per-slot iteration
      // remains in-order (compute is in-order on the primary stream within
      // a batch).
      for (auto& kv : active_by_epoch_) {
        EpochSlot* slot = kv.second;
        const int start = slot->current_progress + 1;
        const int end = slot->max_recorded;
        for (int i = start; i <= end; ++i) {
          if (i < 0 || i >= static_cast<int>(slot->events.size())) break;
          auto& info = slot->events[i];
          if (info.processed) continue;
          if (!EventQuery(info.event)) {
            break;
          }
          info.processed = true;
          slot->current_progress = i;
          just_completed.push_back({kv.first, i});
        }
      }
      cbs_snapshot = callbacks_;
      rank_snapshot = rank_;
    }
    for (const auto& fe : just_completed) {
      for (auto& cb : cbs_snapshot) {
        cb(rank_snapshot, fe.epoch_id, fe.layer_index);
      }
    }
    if (just_completed.empty()) {
      std::this_thread::yield();
    }
  }
}

}  // namespace pd_v2
}  // namespace ksana_llm
