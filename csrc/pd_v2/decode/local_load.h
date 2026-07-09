/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstddef>
#include <cstdint>

#include "batch_scheduler/event_driven_scheduler/schedule_context.h"

namespace ksana_llm {
namespace pd_v2 {

// Decode-engine-local load view, the Decode-side counterpart of the
// Prefill-sourced PeerHealth load fields. Produced by ReadLocalLoad from the
// EVENT_DRIVEN ScheduleContext the Decode hook already holds. Reserved as the
// LOCAL arm's feature vector for the upcoming LinUCB router (route-to-self vs.
// forward-to-Prefill); this step only collects the data — there is no
// consumer of the decision yet.
struct LocalLoadView {
  uint32_t free_kv_blocks = 0;          // GetUsableBlockNumber() on this dp's cache manager
  uint32_t future_free_kv_blocks = 0;   // reserved; always 0 since swap was removed in favor of recompute
  uint32_t waiting_reqs = 0;            // prefilling_reqs.size()
  uint32_t running_reqs = 0;            // decoding_reqs.size()
  uint64_t pending_prefill_tokens = 0;  // sum of planning sequence lengths over prefilling reqs
  uint32_t step_token_num = 0;          // EWMA of tokens processed per schedule step
  uint32_t step_latency_us = 0;         // EWMA of per-step wall time (microseconds)
  uint32_t step_latency_us_inst = 0;    // most recent (non-EWMA) per-step wall time, for TPOT gate

  // Affinity reservation (mechanism 1, NOT implemented in this step): the
  // future LOCAL arm will also carry a `cached_prefix_ratio_on_arm` feature
  // describing how much of the arriving request's prefix is already cached
  // locally. That value is to be sourced from a Decode-side PrefixAffinityTable
  // (per-request history), filled by the LinUCB feature builder — not here.
};

// Read this Decode engine's local load for the given dp_idx. The queue counts
// (waiting/running) and pending_prefill_tokens iterate prefilling_reqs, so this
// must be called on the scheduler thread (the Decode hook's Process() already
// runs there) — no extra locking is taken. The per-step EWMA values are not
// reachable from ScheduleContext; the caller passes them in from the
// EventDrivenScheduleProcessor it owns (EventDrivenScheduleProcessor::GetLocalLoad
// exposes them). They default to 0 for callers that don't track them.
LocalLoadView ReadLocalLoad(const ScheduleContext& ctx, size_t dp_idx, uint32_t step_token_num = 0,
                            uint32_t step_latency_us = 0, uint32_t step_latency_us_inst = 0);

}  // namespace pd_v2
}  // namespace ksana_llm
