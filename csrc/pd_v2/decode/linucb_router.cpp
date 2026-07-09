/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "pd_v2/decode/linucb_router.h"

#include <algorithm>
#include <cmath>

namespace ksana_llm {
namespace pd_v2 {

namespace {

double Clamp01(double v) { return std::max(0.0, std::min(1.0, v)); }

}  // namespace

LinUCBRouter::LinUCBRouter(const LinUCBConfig& config) : config_(config) {
  // A0 = lambda*I with lambda fixed at 1.0 => Ainv0 = I.
  local_.ainv = ScaledIdentity(1.0);
  remote_.ainv = ScaledIdentity(1.0);
}

LinUCBVec LinUCBRouter::BuildFeatures(const LocalArmContext& ctx) const {
  LinUCBVec x{};
  x[0] = 1.0;  // bias
  // idle: low running_reqs (node waiting on Prefill returns) => high.
  const double running_ref = std::max(1.0, config_.running_ref);
  x[1] = 1.0 - std::min(static_cast<double>(ctx.running_reqs) / running_ref, 1.0);
  // tpot_headroom: how far latest per-step latency sits below the target.
  x[2] = LocalReward(ctx);
  // remaining_small: small residual prefill fits inside the wait window => high.
  const double remaining_ref = std::max(1.0, config_.remaining_ref_tokens);
  x[3] = 1.0 - std::min(static_cast<double>(ctx.remaining_prefill_tokens) / remaining_ref, 1.0);
  return x;
}

double LinUCBRouter::LocalReward(const LocalArmContext& ctx) const {
  // Aim BELOW the hard TPOT target, not at it. Keeping prefill local makes the
  // decode node run the residual prefill itself, which lengthens the shared
  // continuous-batching step and pushes TPOT up *after* the decision is made.
  // If the gate stayed open all the way to tpot_target_us, that post-decision
  // bump would carry TPOT past the budget (observed: 40ms -> 55ms). Reserve a
  // safety margin (config_.tpot_safety_factor, default 0.85) so the gate closes
  // early enough to absorb the bump. Lower factor => TPOT weighted higher,
  // prefill kept local less often.
  const double tpot_target_us = std::max(1.0, config_.tpot_target_us * config_.tpot_safety_factor);
  // Conservative: take the worse (larger) of the instantaneous and the EWMA
  // per-step latency. A load spike shows up in the instantaneous value
  // immediately, closing the gate on the next decision; the EWMA guards the
  // tail while it is still catching up downward. step_latency_us_inst is 0
  // before the first step completes — fall back to the EWMA then.
  const double inst_us = static_cast<double>(ctx.step_latency_us_inst);
  const double ewma_us = static_cast<double>(ctx.step_latency_us);
  const double worst_us = std::max(inst_us, ewma_us);
  return Clamp01((tpot_target_us - worst_us) / tpot_target_us);
}

double LinUCBRouter::TtftCredit(double peer_ttft_us) const {
  const double threshold_us = std::max(1.0, config_.ttft_threshold_us);
  // 0 below the TTFT budget, rising to 1.0 at 2x the budget.
  return Clamp01((peer_ttft_us - threshold_us) / threshold_us);
}

double LinUCBRouter::RemoteReward(double elapsed_us) const {
  // Pivot on the TTFT budget itself: 1.0 at zero latency, linearly down to 0 at
  // ttft_threshold_us. A remote round-trip that breaches the budget earns 0,
  // ceding the slot to the LOCAL arm whenever it still has TPOT headroom. (Was
  // 2x the budget, which pushed the real diversion point out to 2x the
  // threshold and starved the LOCAL arm even when REMOTE was clearly slow.)
  const double ref = std::max(1.0, config_.ttft_threshold_us);
  return Clamp01(1.0 - elapsed_us / ref);
}

double LinUCBRouter::ScoreLocked(const Arm& arm, const LinUCBVec& x) const {
  const LinUCBVec theta = MatVec(arm.ainv, arm.b);
  const double mean = Dot(theta, x);
  const double var = std::max(0.0, QuadForm(arm.ainv, x));  // PSD, guard rounding
  return mean + config_.alpha * std::sqrt(var);
}

RouteArm LinUCBRouter::SelectArm(const LocalArmContext& ctx, LinUCBVec* out_features) {
  const LinUCBVec x = BuildFeatures(ctx);
  if (out_features != nullptr) *out_features = x;

  std::lock_guard<std::mutex> lock(router_mu_);
  const double local_score = ScoreLocked(local_, x);
  const double remote_score = ScoreLocked(remote_, x);
  // Tie favors REMOTE (the established default path).
  return local_score > remote_score ? RouteArm::kLocal : RouteArm::kRemotePeer;
}

void LinUCBRouter::RecordReward(RouteArm arm, const LinUCBVec& features, double reward) {
  const double r = Clamp01(reward);
  std::lock_guard<std::mutex> lock(router_mu_);
  Arm& a = ArmRef(arm);
  ShermanMorrisonUpdate(a.ainv, features);
  for (std::size_t i = 0; i < kLinUCBDim; ++i) a.b[i] += r * features[i];
}

double LinUCBRouter::ScoreForTest(RouteArm arm, const LinUCBVec& x) const {
  std::lock_guard<std::mutex> lock(router_mu_);
  return ScoreLocked(ArmRef(arm), x);
}

}  // namespace pd_v2
}  // namespace ksana_llm
