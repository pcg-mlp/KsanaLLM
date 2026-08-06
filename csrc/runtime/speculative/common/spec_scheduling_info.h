/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstddef>

namespace ksana_llm {

// Scheduling information provided by each speculative decoder to the scheduler.
// The scheduler aggregates these values across all registered decoders to determine
// resource allocation (KV blocks, token capacity, workload estimation).
struct SpecSchedulingInfo {
  // Additional KV cache tokens to reserve per request.
  // For MTP: mtp_step_num (extra tokens for draft, converted to blocks by GetRequestStepBlockNumber).
  // For PTP/Pure predict: 0 (no extra KV tokens needed).
  size_t extra_kv_tokens = 0;

  // Additional token capacity to reserve per request.
  // For MTP: mtp_step_num - 1 (one token shares the main tokeninary KV prefix).
  // For PTP: 2 * ptp_step_num (verify + placeholder).
  size_t extra_token_capacity = 0;

  // Workload multiplier for planning estimation.
  // For MTP: mtp_step_num + ptp_step_num.
  // For PTP/Pure predict: ptp_step_num.
  size_t workload_multiplier = 0;
};

}  // namespace ksana_llm
