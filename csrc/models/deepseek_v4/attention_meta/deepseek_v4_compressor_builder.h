/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <vector>

#include "configure/context.h"
#include "models/deepseek_v4/attention_meta/deepseek_v4_attention_meta.h"
#include "runtime/forward_request.h"

namespace ksana_llm {

// Prepares per-batch compressor metadata for a single forward step.
//
// Replaces CompressorMeta::Prepare(). Must be called after request categorisation
// (i.e. flash_compressor.dp_reqs is NOT used here; categorisation is re-derived from
// forward_reqs using the same GetForwardRequestType logic) and before any kernel
// that consumes flash_compressor / paged_compressor fields.
//
// Slot management (allocate new slots, tail indices) uses the full forward_reqs list
// and requires compressor_shared.req_to_slot / free_slot to be persistent across calls.
// Slot eviction is NOT performed here; call EvictFinishedRequests() instead when the
// scheduler notifies that requests have finished or are being recomputed.
void PrepareCompressor(const std::vector<ForwardRequest>& forward_reqs,
                       CompressorSharedAttentionMeta& compressor_shared, CompressorAttentionMeta& flash_compressor,
                       CompressorAttentionMeta& paged_compressor, std::shared_ptr<Context> context);

// Release slots held by finished (or recomputed) requests so they can be reused.
// Must be called by the upper layer when it receives finish/recompute notifications from the
// scheduler. Unknown req_ids are silently ignored, making it safe to broadcast across DP groups.
// This replaces the previous auto-eviction logic in PrepareCompressor(), which was incorrect
// under max_pp_batch_num > 1 (different PP batches carry different req subsets and would
// otherwise evict each other's slots).
void EvictFinishedRequests(const std::vector<int64_t>& finished_req_ids,
                           CompressorSharedAttentionMeta& compressor_shared);

}  // namespace ksana_llm
