/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

#include "utils/status.h"

namespace ksana_llm {
namespace pd_v2 {
namespace mooncake_backend {

// Per-peer slot ring sitting in front of a fixed HostStagePool partition.
// Decode-side credit accounting for the pd_v2 control-stage path: every
// pending PrefillRequest reserves a contiguous run of slots in this peer's
// partition until the matching StageAck releases them. See
// docs/technology/design/pd_v2_control_stage.md §5.2.
//
// Allocation policy: next-fit + bitmap. Each AcquireSlots scans for a
// contiguous run of `n_slots` free bits starting from a hint pointer that
// advances past the most-recently-allocated run. Out-of-order ReleaseSlots
// just clears bits — the bitmap doesn't care which order acks come back
// in. That keeps the algorithm simple at the cost of O(partition_n_slots)
// per allocation, which is fine for the design's 256-slot/peer default.
//
// Threading: AcquireSlots blocks on a CV when the request can't fit;
// ReleaseSlots / Shutdown wake all waiters. Shutdown causes pending
// AcquireSlots to fail immediately so peer churn doesn't deadlock the
// hook thread.
class PeerCredits {
 public:
  PeerCredits();
  ~PeerCredits();

  PeerCredits(const PeerCredits&) = delete;
  PeerCredits& operator=(const PeerCredits&) = delete;

  // Bind to one absolute slot range in the HostStagePool. `partition_base_slot`
  // and `partition_n_slots` come straight from
  // HostStagePool::PeerPartition. Re-Init after Shutdown is allowed.
  Status Init(uint32_t partition_base_slot, uint32_t partition_n_slots);

  // Reserve a contiguous run of `n_slots`. Returns the *absolute* start
  // slot index in the pool (already partition-base-added) so callers
  // can plug it directly into HostStagePool::SlotPtr or the
  // PrefillRequest stage_start_slot field. Blocks for at most
  // `timeout_ms` (negative = wait forever, 0 = non-blocking try).
  // Errors with RET_INVALID_ARGUMENT if n_slots == 0 or > partition,
  // RET_REQUEST_TIMEOUT on timeout, RET_RUNTIME_FAILED if Shutdown
  // unblocks the wait.
  Status AcquireSlots(uint32_t n_slots, int64_t timeout_ms, uint32_t* out_start_slot);

  // Release the slot range previously reserved by AcquireSlots.
  // start_slot is absolute (matches what AcquireSlots returned). Out-
  // of-order release is fine — the bitmap doesn't care about the order
  // acks come back in. Releasing slots that are already free is logged
  // but non-fatal (treat as a double-ack from a buggy Prefill peer
  // rather than a panic).
  Status ReleaseSlots(uint32_t start_slot, uint32_t n_slots);

  // Force-fail any in-flight AcquireSlots wait. Subsequent AcquireSlots
  // also fail until Init is called again. Used when the peer is being
  // disconnected or the engine is shutting down — avoids the hook
  // thread getting stuck holding a lock indefinitely.
  void Shutdown();

  // Test/observability accessors.
  uint32_t free_slots() const;

 private:
  // First-fit scan for a contiguous run of n_slots free bits starting
  // at relative index 0. Returns relative slot index
  // (0..partition_n_slots_) on success, partition_n_slots_ as the
  // not-found sentinel. Caller must hold mu_.
  uint32_t FindContiguousLocked(uint32_t n_slots) const;

  mutable std::mutex mu_;
  std::condition_variable cv_;
  uint32_t partition_base_slot_ = 0;
  uint32_t partition_n_slots_ = 0;
  std::vector<bool> busy_;  // size = partition_n_slots_
  bool initialized_ = false;
  bool shutdown_ = false;
};

}  // namespace mooncake_backend
}  // namespace pd_v2
}  // namespace ksana_llm
