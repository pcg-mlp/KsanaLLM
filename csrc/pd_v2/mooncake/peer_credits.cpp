/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "pd_v2/mooncake/peer_credits.h"

#include <chrono>

#include "fmt/core.h"
#include "utils/logger.h"
#include "utils/ret_code.h"

namespace ksana_llm {
namespace pd_v2 {
namespace mooncake_backend {

PeerCredits::PeerCredits() = default;
PeerCredits::~PeerCredits() = default;

Status PeerCredits::Init(uint32_t partition_base_slot, uint32_t partition_n_slots) {
  std::lock_guard<std::mutex> lg(mu_);
  if (partition_n_slots == 0) {
    return Status(RET_INVALID_ARGUMENT, "PeerCredits partition_n_slots must be > 0");
  }
  partition_base_slot_ = partition_base_slot;
  partition_n_slots_ = partition_n_slots;
  busy_.assign(partition_n_slots, false);
  initialized_ = true;
  shutdown_ = false;
  return Status();
}

uint32_t PeerCredits::FindContiguousLocked(uint32_t n_slots) const {
  // First-fit scan from the start of the partition. With at most a few
  // hundred slots per peer and small in-flight counts, the O(N) scan
  // cost is negligible, and starting at 0 each time means out-of-order
  // releases get reused immediately rather than waiting for the cursor
  // to wrap.
  const uint32_t total = partition_n_slots_;
  uint32_t run = 0;
  uint32_t run_start = 0;
  for (uint32_t i = 0; i < total; ++i) {
    if (!busy_[i]) {
      if (run == 0) run_start = i;
      ++run;
      if (run >= n_slots) return run_start;
    } else {
      run = 0;
    }
  }
  return total;  // not found
}

Status PeerCredits::AcquireSlots(uint32_t n_slots, int64_t timeout_ms, uint32_t* out_start_slot) {
  if (out_start_slot == nullptr) {
    return Status(RET_INVALID_ARGUMENT, "out_start_slot is null");
  }
  std::unique_lock<std::mutex> lk(mu_);
  if (!initialized_) {
    return Status(RET_INVALID_ARGUMENT, "PeerCredits not initialized");
  }
  if (n_slots == 0 || n_slots > partition_n_slots_) {
    return Status(RET_INVALID_ARGUMENT, fmt::format("PeerCredits::AcquireSlots n_slots={} out of range "
                                                    "(partition_n_slots={})",
                                                    n_slots, partition_n_slots_));
  }

  auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms < 0 ? 0 : timeout_ms);

  while (true) {
    if (shutdown_) {
      return Status(RET_RUNTIME_FAILED, "PeerCredits::AcquireSlots aborted by shutdown");
    }
    uint32_t rel = FindContiguousLocked(n_slots);
    if (rel != partition_n_slots_) {
      for (uint32_t i = 0; i < n_slots; ++i) busy_[rel + i] = true;
      *out_start_slot = partition_base_slot_ + rel;
      return Status();
    }
    if (timeout_ms == 0) {
      return Status(RET_REQUEST_TIMEOUT, fmt::format("PeerCredits::AcquireSlots non-blocking, "
                                                     "no contiguous run of {} slots free",
                                                     n_slots));
    }
    if (timeout_ms < 0) {
      cv_.wait(lk);
    } else {
      if (cv_.wait_until(lk, deadline) == std::cv_status::timeout) {
        return Status(RET_REQUEST_TIMEOUT, fmt::format("PeerCredits::AcquireSlots timed out after {} ms "
                                                       "waiting for {} slots",
                                                       timeout_ms, n_slots));
      }
    }
  }
}

Status PeerCredits::ReleaseSlots(uint32_t start_slot, uint32_t n_slots) {
  std::lock_guard<std::mutex> lg(mu_);
  if (!initialized_) {
    return Status(RET_INVALID_ARGUMENT, "PeerCredits not initialized");
  }
  if (n_slots == 0) return Status();
  if (start_slot < partition_base_slot_ || start_slot + n_slots > partition_base_slot_ + partition_n_slots_) {
    return Status(RET_INVALID_ARGUMENT, fmt::format("PeerCredits::ReleaseSlots range [{} +{}) out of "
                                                    "partition [{} +{})",
                                                    start_slot, n_slots, partition_base_slot_, partition_n_slots_));
  }
  const uint32_t rel = start_slot - partition_base_slot_;
  bool already_free = false;
  for (uint32_t i = 0; i < n_slots; ++i) {
    if (!busy_[rel + i]) already_free = true;
    busy_[rel + i] = false;
  }
  if (already_free) {
    KLLM_LOG_WARNING << "PeerCredits::ReleaseSlots double-release at start=" << start_slot << " n=" << n_slots
                     << " (treating as no-op for the already-free bits)";
  }
  cv_.notify_all();
  return Status();
}

void PeerCredits::Shutdown() {
  std::lock_guard<std::mutex> lg(mu_);
  shutdown_ = true;
  cv_.notify_all();
}

uint32_t PeerCredits::free_slots() const {
  std::lock_guard<std::mutex> lg(mu_);
  uint32_t free = 0;
  for (bool b : busy_) {
    if (!b) ++free;
  }
  return free;
}

}  // namespace mooncake_backend
}  // namespace pd_v2
}  // namespace ksana_llm
