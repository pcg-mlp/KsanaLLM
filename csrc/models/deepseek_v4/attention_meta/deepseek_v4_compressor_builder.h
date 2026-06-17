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
// Slot management is handled by InflightResourceManager::ProcessScheduleOutput in the
// preprocess thread; PrepareCompressor only verifies that every required slot is present.
void PrepareCompressor(const std::vector<ForwardRequest>& forward_reqs,
                       CompressorSharedAttentionMeta& compressor_shared, CompressorAttentionMeta& flash_compressor,
                       CompressorAttentionMeta& paged_compressor, std::shared_ptr<Context> context);

}  // namespace ksana_llm
