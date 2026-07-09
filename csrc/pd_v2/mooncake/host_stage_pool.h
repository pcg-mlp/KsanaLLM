/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

#include "utils/status.h"

namespace ksana_llm {
namespace pd_v2 {
namespace mooncake_backend {

// Prefill-side host buffer for the pd_v2 control-plane stage path. Decode
// peers RDMA-WRITE the serialized prefill_req payload here, then send a
// small Tent notify ({req_id, start_slot, n_slots, total_len}) referencing
// the slot range. See docs/technology/design/pd_v2_control_stage.md §5.1.
//
// Layout:
//   total_bytes / kSlotSize = total_slots, divided evenly across at most
//   max_peers active Decode senders. Each peer claims one contiguous range
//   [base_slot, base_slot + slots_per_peer) at first registration; the
//   range is fixed for the pool's lifetime so inflight WRITEs don't see
//   the slot map shift under them when peers churn.
//
// Threading: Init / RegisterPeer / PeerPartitionFor take an internal
// mutex. SlotPtr is lock-free (caller derives the absolute index from a
// PeerPartition snapshot, which is stable once registered).
//
// Mooncake decoupling: HostStagePool only mmap's host memory; the
// connector wires it into the engine TransferEngine via
// registerLocalMemory(base, total_bytes, "cpu:0"). Keeps this class
// unit-testable without an RDMA stack.
class HostStagePool {
 public:
  // Slot size pinned at 64 KB; design doc §11 covers the choice.
  static constexpr size_t kSlotSize = 64 * 1024;
  // Default 2 GB pool; tunable via Init(). Rationale in §8: 2 GB is the
  // text-load default, multimodal deploys raise it via config.
  static constexpr size_t kDefaultPoolBytes = 2ull << 30;

  struct PeerPartition {
    size_t base_slot = 0;  // absolute starting slot index in the pool
    size_t n_slots = 0;    // slots assigned to this peer
  };

  HostStagePool();
  ~HostStagePool();

  HostStagePool(const HostStagePool&) = delete;
  HostStagePool& operator=(const HostStagePool&) = delete;

  // Allocate `total_bytes` of host memory and split into slots distributed
  // evenly across `max_peers` partitions. Memory is pinned (mlock,
  // best-effort — Mooncake's ibv_reg_mr also pins) and zero-faulted.
  // Errors out if total_bytes is not a non-zero multiple of kSlotSize, or
  // max_peers is 0, or the slot count would be smaller than max_peers.
  Status Init(size_t total_bytes, size_t max_peers);

  Status Shutdown();

  // Idempotent: returns the existing partition if `decode_addr` was
  // registered before. Otherwise consumes the next free partition (up to
  // max_peers). Errors out only when all partitions are taken AND
  // `decode_addr` is new — re-registering an existing peer always works.
  Status RegisterPeer(const std::string& decode_addr, PeerPartition* out);

  // Look up a partition without consuming a new slot. Errors if
  // `decode_addr` was never registered.
  Status PeerPartitionFor(const std::string& decode_addr, PeerPartition* out) const;

  // Pointer to the slot at the absolute index. Caller is responsible for
  // bounds; abs_slot_idx must be < total_slots(). Typically derived from
  // a PeerPartition by `base_slot + relative_slot`.
  char* SlotPtr(size_t abs_slot_idx) const { return static_cast<char*>(base_) + abs_slot_idx * kSlotSize; }

  void* base() const { return base_; }
  size_t total_bytes() const { return total_slots_ * kSlotSize; }
  size_t total_slots() const { return total_slots_; }
  size_t slots_per_peer() const { return slots_per_peer_; }
  size_t max_peers() const { return max_peers_; }

 private:
  mutable std::mutex mu_;
  void* base_ = nullptr;
  size_t total_slots_ = 0;
  size_t slots_per_peer_ = 0;
  size_t max_peers_ = 0;
  size_t next_partition_index_ = 0;
  std::unordered_map<std::string, PeerPartition> peer_map_;
  bool initialized_ = false;
};

}  // namespace mooncake_backend
}  // namespace pd_v2
}  // namespace ksana_llm
