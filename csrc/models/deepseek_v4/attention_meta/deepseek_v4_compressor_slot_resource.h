/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <unordered_map>
#include <vector>

#include "configure/schedule_config_parser.h"
#include "executor/inflight_resource.h"

namespace ksana_llm {

// Manages the lifecycle of compressor state slots for DeepSeek V4.
// Each in-flight request that requires compressor attention gets assigned a
// unique slot index (int). Slots are recycled when the request finishes or is
// recomputed. The pool capacity is max_batch_size * max_pp_batch_num so that
// all concurrently in-flight requests across PP batches can be accommodated.
//
// Thread safety: Alloc/Free (called from the preprocess thread) acquire a
// unique_lock via LockForWrite(); Get/Has (called from the execute thread via
// PrepareCompressor) acquire a shared_lock via LockForRead().
class CompressorSlotResource : public BaseInflightResource<int> {
 public:
  CompressorSlotResource() = default;
  ~CompressorSlotResource() override;

  // ---- InflightResourceInterface overrides ----

  void Init(const RuntimeConfig& runtime_config) override;

  void Alloc(int64_t req_id) override;
  void Free(int64_t req_id) override;
  bool Has(int64_t req_id) override;

 protected:
  // ---- BaseInflightResource<int> override ----
  // Called by Get() under shared_lock (via LockForRead).
  int DoGetUnderLock(int64_t req_id) override;

 private:
  // Map from request id to allocated slot index.
  std::unordered_map<int64_t, int> req_to_slot_;

  // Pool of available (free) slot indices, used as a stack (LIFO).
  // Vector provides contiguous storage and better cache behaviour than std::queue/deque.
  // LIFO order means recently freed slots are reused first, improving GPU state cache locality.
  std::vector<int> free_slot_;

  // Total number of slots (= max_batch_size * max_pp_batch_num), set by Init.
  size_t slot_pool_capacity_ = 0;
};

}  // namespace ksana_llm
