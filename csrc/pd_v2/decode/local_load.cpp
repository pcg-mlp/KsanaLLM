/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "pd_v2/decode/local_load.h"

#include "batch_scheduler/event_driven_scheduler/event_driven_state.h"
#include "cache_manager/cache_manager_interface.h"
#include "runtime/infer_request.h"

namespace ksana_llm {
namespace pd_v2 {

LocalLoadView ReadLocalLoad(const ScheduleContext& ctx, size_t dp_idx, uint32_t step_token_num,
                            uint32_t step_latency_us, uint32_t step_latency_us_inst) {
  LocalLoadView view;
  view.step_token_num = step_token_num;
  view.step_latency_us = step_latency_us;
  view.step_latency_us_inst = step_latency_us_inst;

  const DpScheduleState& dp_state = ctx.GetDpScheduleState(ctx.GetCurrentMultiBatchId(), dp_idx);
  view.waiting_reqs = static_cast<uint32_t>(dp_state.prefilling_reqs.size());
  view.running_reqs = static_cast<uint32_t>(dp_state.decoding_reqs.size());
  for (const auto& req : dp_state.prefilling_reqs) {
    if (req != nullptr) {
      view.pending_prefill_tokens += static_cast<uint64_t>(req->GetPlanningSequenceLen());
    }
  }

  const std::shared_ptr<CacheManagerInterface>& cm = ctx.GetDpCacheManager(dp_idx);
  if (cm != nullptr) {
    view.free_kv_blocks = static_cast<uint32_t>(cm->GetUsableBlockNumber());
    // swap removed: no swapout-in-flight blocks, so future-free is always 0.
    view.future_free_kv_blocks = 0;
  }

  return view;
}

}  // namespace pd_v2
}  // namespace ksana_llm
