/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "batch_scheduler/event_driven_scheduler/event_driven_schedule_processor.h"  // ScheduleLoadStat
#include "batch_scheduler/event_driven_scheduler/schedule_hook_interface.h"
#include "pd_v2/public/pd_v2_connector.h"
#include "pd_v2/public/pd_v2_types.h"
#include "pd_v2/wire/sampling_config_wire.h"
#include "utils/request.h"  // ksana_llm::SamplingConfig

namespace ksana_llm {
namespace pd_v2 {

// Decode-side schedule hook for pd_v2 (token-based PD disaggregation).
//
// Slots into the EDS schedule-hook chain AFTER GlobalCacheQueryHook (which
// is hook_idx=0 and populates req->kv_cache_blocks). Wire-up location:
// csrc/engine/inference_engine.cpp adds the hook with hook_idx=1 when the
// connector is in Decode role.
//
// Process(req):
//   1. PeerSelector pick via connector->CanAccept(req_id, input_len).
//      No healthy peer ⇒ NextStep{kFailAndEnd}.
//   2. Translate req->sampling_config → SamplingConfigWire with
//      max_new_tokens = 1 + max(0, ptp_token_num) so Prefill produces
//      exactly the tokens Decode needs to bootstrap its loop.
//   3. connector->SubmitPrefillRequest(...). Returns immediately; the
//      on_complete callback (OnPrefillComplete trampoline) fires later
//      on the connector dispatch thread.
//   4. Return NextStep{kWaitAsyncResult}. EDS parks the req in
//      preprocessing_reqs.
//
// OnPrefillComplete (off EDS thread):
//   - Stash (status, generated_tokens) in completion_results_ keyed by
//     req_id.
//   - Fire async_callback_(EventAsyncTaskFinishedInfo) which posts
//     onto the EDS event_queue_; the main loop picks it up and calls
//     UpdateAsyncResult on the EDS thread.
//
// UpdateAsyncResult (EDS thread):
//   - Drain completion_results_. For each completed req:
//     - kOk: req->output_tokens.push_back(tokens[0]),
//            req->draft_tokens.mtp = tokens[1..M-1],
//            req->kv_cached_token_num = req->input_tokens.size(),
//            req->infer_stage = kDecode,
//            req->NotifyStep() ← TTFT: streaming iterator wakes
//                immediately, doesn't wait for next forward.
//            NextStep{kContinue}.
//     - failed/cancelled/timeout: NextStep{kFailAndEnd}.
//   - Note: structured_generator is NOT touched here. It lives in the
//     executor process and is built/fed by GenerationController on the
//     first decode step.
class PdV2DecodeHook : public ScheduleHookInterface {
 public:
  // `local_complete_threshold` (token count): when
  // (input_tokens.size() − prefix_cache_len) ≤ threshold, this request's
  // remaining prefill workload is small enough that PD round-trip
  // overhead would dominate. The hook returns kContinue with no
  // SubmitPrefillRequest, letting the request flow through normal
  // ksana_llm prefill+decode locally on this Decode node. 0 disables
  // (always forward to Prefill, no fast-path). Sourced from
  // EngineConfig.local_complete_threshold.
  PdV2DecodeHook(std::shared_ptr<PdV2ConnectorInterface> connector, int hook_idx, int local_complete_threshold = 0,
                 SelectorPolicy selector_policy = SelectorPolicy::kRoundRobin,
                 std::function<ScheduleLoadStat()> load_stat_fn = {});
  ~PdV2DecodeHook() override = default;

  // ScheduleHookInterface impl
  ReqIdToNextStep Process(const std::vector<InferRequestPtr>& requests, ScheduleContext& context) override;
  ReqIdToNextStep UpdateAsyncResult(const std::vector<AsyncTaskResult>& finished_tasks,
                                    ScheduleContext& context) override;
  void SetAsyncTaskFinishedCallback(AsyncTaskFinishedCallback callback) override;

  // For tests: access internal state without exposing the connector.
  size_t PendingCountForTest() const;

 private:
  // Static trampoline; ctx == this. Fires on the connector dispatch
  // thread (off EDS), so all member access goes through mu_.
  static void OnPrefillComplete(uint64_t req_id, PdCompleteStatus status, const std::vector<int>& generated_tokens,
                                const PdPrefillMetrics& prefill_metrics, const std::string& prefill_peer, void* ctx);

  // Build SamplingConfigWire from ksana_llm::SamplingConfig. Pins
  // max_new_tokens = 1 + max(0, ptp_token_num).
  static SamplingConfigWire MakeWireSamplingConfig(const SamplingConfig& sc);

  std::shared_ptr<PdV2ConnectorInterface> connector_;
  int hook_idx_;
  // Threshold (token count) for the "remaining prefill < N stays local"
  // fast-path. See ctor doc.
  int local_complete_threshold_ = 0;
  // Decode peer-selection policy. kLinUCB routes LOCAL-vs-REMOTE through the
  // connector's contextual bandit (and disables the static
  // local_complete_threshold_ soft-bypass, which the bandit subsumes via its
  // remaining_small feature). All other policies behave as before.
  SelectorPolicy selector_policy_ = SelectorPolicy::kRoundRobin;
  // LinUCB only: supplies this Decode engine's per-step EWMA load
  // (step_token_num / step_latency_us) for the LocalArmContext. Injected by
  // inference_engine bootstrap; empty for non-kLinUCB deployments.
  std::function<ScheduleLoadStat()> load_stat_fn_;
  AsyncTaskFinishedCallback async_callback_;

  mutable std::mutex mu_;
  // req_id → (status, generated_tokens). Populated by OnPrefillComplete
  // off-thread, drained by UpdateAsyncResult on the EDS thread.
  struct PendingResult {
    PdCompleteStatus status = PdCompleteStatus::kOk;
    std::vector<int> generated_tokens;
    PdPrefillMetrics prefill_metrics;  // Prefill-side timing/prefix-cache for the decode access log
    std::string prefill_peer;          // selected peer Prefill engine's server_name (for the access log)
    uint64_t complete_time_us = 0;     // decode-clock arrival of PrefillComplete (T3)
  };
  std::unordered_map<uint64_t, PendingResult> completion_results_;
};

}  // namespace pd_v2
}  // namespace ksana_llm
