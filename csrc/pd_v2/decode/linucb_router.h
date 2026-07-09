/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <mutex>

#include "pd_v2/decode/linucb_math.h"
#include "pd_v2/public/pd_v2_connector.h"  // LocalArmContext, RouteArm

namespace ksana_llm {
namespace pd_v2 {

// Tunables for the LinUCB router. All sourced from EngineConfig; the
// regularization lambda is fixed at 1.0 and rewards are single-term (no
// weights) per the v1 minimal design.
struct LinUCBConfig {
  double alpha = 1.0;                    // exploration weight on sqrt(x^T Ainv x)
  double tpot_target_us = 50000.0;       // TPOT budget (us); LOCAL gate closes as per-step latency nears this
  double tpot_safety_factor = 0.85;      // LOCAL gate aims at tpot_target_us * this, reserving headroom for the
                                         // TPOT bump that keeping prefill local induces. <1 => TPOT prioritized
                                         // harder / prefill kept local less often. 1.0 = aim straight at target.
  double ttft_threshold_us = 1500000.0;  // TTFT budget (us); keeping prefill LOCAL only earns credit once the
                                         // remote round-trip (TTFT proxy) would exceed this
  double running_ref = 16.0;             // running_reqs that count as "fully busy" (idle feature denom)
  double remaining_ref_tokens = 4096.0;  // remaining prefill that counts as "too big to keep local"
  double remote_lat_ref_us = 200000.0;   // DEPRECATED: REMOTE arm reward now scales off ttft_threshold_us;
                                         // kept only so existing config plumbing/YAML keys don't break
};

// Two-arm disjoint LinUCB contextual bandit for the LOCAL-vs-REMOTE decision.
// Both arms score the SAME 4-dim context (bias, idle, tpot_headroom,
// remaining_small); each arm learns its own theta, so the bandit discovers
// "in which contexts does keeping prefill local earn more reward than
// forwarding it." Remote peer *selection* is NOT part of this — PeerSelector
// still picks the peer.
//
// Threading: a single router_mu_ guards both arms. SelectArm (scheduler
// thread) and RecordReward (scheduler thread for LOCAL, connector notify
// thread for REMOTE) are all short critical sections over fixed-size arrays.
class LinUCBRouter {
 public:
  explicit LinUCBRouter(const LinUCBConfig& config);

  // Build the feature vector for `ctx` and return the arm with the higher UCB
  // score (argmax over {kLocal, kRemotePeer}). The feature vector actually
  // used is written to *out_features so the caller can stash it and hand it
  // back to RecordReward when the (possibly deferred) reward arrives.
  RouteArm SelectArm(const LocalArmContext& ctx, LinUCBVec* out_features);

  // Apply a reward in [0,1] to `arm` for the context `features` (the vector
  // SelectArm returned). Rank-1 Sherman-Morrison update of Ainv + b += r*x.
  void RecordReward(RouteArm arm, const LinUCBVec& features, double reward);

  // TPOT headroom for the LOCAL arm: 1.0 while the node is idle, decaying to
  // 0 as the per-step latency approaches/exceeds tpot_target_us. This is the
  // "can we afford to keep prefill local" gate multiplier. To守住 50ms even
  // when load spikes between decisions, it reads max(instantaneous, EWMA)
  // per-step latency — the conservative (worse) of the two — so a fresh spike
  // closes the gate on the very next decision instead of lagging the EWMA.
  double LocalReward(const LocalArmContext& ctx) const;

  // TTFT credit for the LOCAL arm: 0 until the (peer) round-trip latency that
  // a REMOTE dispatch would incur exceeds ttft_threshold_us, then rising
  // linearly to 1.0 at 2x the threshold. "How much keeping prefill local is
  // worth", i.e. how badly REMOTE would blow the TTFT budget. peer_ttft_us and
  // the threshold are both in microseconds — no unit conversion. The LOCAL
  // arm's deferred reward is TtftCredit(select-time peer RTT) *
  // LocalReward(settle-time load): a multiplicative AND gate — keep local only
  // when REMOTE would be slow AND we still have TPOT headroom to absorb it.
  double TtftCredit(double peer_ttft_us) const;

  // REMOTE arm reward: faster Submit->Complete round-trip => higher reward,
  // on the SAME ttft scale as TtftCredit so the two arms are directly
  // comparable. 1.0 at zero latency, linearly to 0 at ttft_threshold_us — the
  // diversion pivot: once REMOTE breaches the TTFT budget it scores 0 and the
  // LOCAL arm wins whenever it has any TPOT headroom. elapsed_us in micros.
  double RemoteReward(double elapsed_us) const;

  // Build the shared 4-dim feature vector from a context. Exposed for tests.
  LinUCBVec BuildFeatures(const LocalArmContext& ctx) const;

  // Test accessor: UCB score for an arm under feature vector x.
  double ScoreForTest(RouteArm arm, const LinUCBVec& x) const;

 private:
  struct Arm {
    LinUCBMat ainv;  // (lambda*I + sum x x^T)^{-1}
    LinUCBVec b{};   // sum r*x
  };

  const Arm& ArmRef(RouteArm arm) const { return arm == RouteArm::kLocal ? local_ : remote_; }
  Arm& ArmRef(RouteArm arm) { return arm == RouteArm::kLocal ? local_ : remote_; }

  // UCB score under router_mu_ held by caller.
  double ScoreLocked(const Arm& arm, const LinUCBVec& x) const;

  LinUCBConfig config_;
  mutable std::mutex router_mu_;
  Arm local_;
  Arm remote_;
};

}  // namespace pd_v2
}  // namespace ksana_llm
