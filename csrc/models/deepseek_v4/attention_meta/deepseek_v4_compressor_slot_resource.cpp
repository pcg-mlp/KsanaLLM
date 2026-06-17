/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/deepseek_v4/attention_meta/deepseek_v4_compressor_slot_resource.h"

namespace ksana_llm {

// The explicitly-defined destructor serves as the key function for vtable emission.
// It must be non-inline and defined in this translation unit.
CompressorSlotResource::~CompressorSlotResource() = default;

void CompressorSlotResource::Init(const RuntimeConfig& runtime_config) {
  const size_t max_batch_size = static_cast<size_t>(runtime_config.max_batch_size);
  const size_t max_pp_batch_num = std::max<size_t>(runtime_config.max_pp_batch_num, 1);
  slot_pool_capacity_ = max_batch_size * max_pp_batch_num;

  KLLM_LOG_INFO << "CompressorSlotResource init: slot_pool_capacity=" << slot_pool_capacity_
                << " (= max_batch_size=" << max_batch_size << " * max_pp_batch_num=" << max_pp_batch_num << ")";

  // The pool is initially full: all indices are free.
  free_slot_.reserve(slot_pool_capacity_);
  for (size_t i = 0; i < slot_pool_capacity_; ++i) {
    free_slot_.push_back(static_cast<int>(i));
  }
}

void CompressorSlotResource::Alloc(int64_t req_id) {
  auto lock = LockForWrite();
  if (req_to_slot_.find(req_id) != req_to_slot_.end()) {
    return;  // Already allocated.
  }
  KLLM_CHECK_WITH_INFO(!free_slot_.empty(),
                       fmt::format("CompressorSlotResource: free_slot pool exhausted (in-flight={}). "
                                   "Total concurrent compressor requests across all PP batches exceeded the slot pool "
                                   "capacity (max_batch_size * max_pp_batch_num = {}). This usually means "
                                   "Free was not called for some finished or recomputed requests.",
                                   req_to_slot_.size(), slot_pool_capacity_));
  req_to_slot_.emplace(req_id, free_slot_.back());
  free_slot_.pop_back();
}

void CompressorSlotResource::Free(int64_t req_id) {
  auto lock = LockForWrite();
  auto it = req_to_slot_.find(req_id);
  if (it == req_to_slot_.end()) {
    return;  // Not in map; nothing to free.
  }
  free_slot_.push_back(it->second);
  req_to_slot_.erase(it);
}

bool CompressorSlotResource::Has(int64_t req_id) {
  auto lock = LockForRead();
  return req_to_slot_.find(req_id) != req_to_slot_.end();
}

int CompressorSlotResource::DoGetUnderLock(int64_t req_id) {
  // Called under shared_lock held by BaseInflightResource::Get().
  return req_to_slot_.at(req_id);
}

}  // namespace ksana_llm
