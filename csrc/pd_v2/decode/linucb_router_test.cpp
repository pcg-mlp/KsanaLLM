/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "pd_v2/decode/linucb_router.h"

#include <gtest/gtest.h>

#include <cmath>

namespace ksana_llm {
namespace pd_v2 {
namespace {

LocalArmContext IdleContext() {
  // No running decode reqs (waiting on Prefill), per-step latency well under
  // the 50ms target, tiny residual prefill -> all three signals favor LOCAL.
  LocalArmContext c;
  c.running_reqs = 0;
  c.step_latency_us = 10000;  // 10ms, far below 50ms
  c.remaining_prefill_tokens = 64;
  return c;
}

LocalArmContext BusyContext() {
  LocalArmContext c;
  c.running_reqs = 64;        // well past running_ref
  c.step_latency_us = 49000;  // 49ms, hugging the TPOT budget
  c.remaining_prefill_tokens = 8192;
  return c;
}

LinUCBConfig DefaultConfig() {
  LinUCBConfig cfg;
  cfg.alpha = 1.0;
  cfg.tpot_target_us = 50000.0;
  cfg.running_ref = 16.0;
  cfg.remaining_ref_tokens = 4096.0;
  cfg.remote_lat_ref_us = 200000.0;
  return cfg;
}

TEST(LinUCBRouterTest, FeatureRangesAndDirection) {
  LinUCBRouter router(DefaultConfig());
  const LinUCBVec idle = router.BuildFeatures(IdleContext());
  const LinUCBVec busy = router.BuildFeatures(BusyContext());

  EXPECT_DOUBLE_EQ(idle[0], 1.0);  // bias
  EXPECT_DOUBLE_EQ(busy[0], 1.0);
  for (std::size_t i = 1; i < kLinUCBDim; ++i) {
    EXPECT_GE(idle[i], 0.0);
    EXPECT_LE(idle[i], 1.0);
    // Idle context scores higher on every "favor LOCAL" feature than busy.
    EXPECT_GT(idle[i], busy[i]) << "feature " << i;
  }
}

TEST(LinUCBRouterTest, RewardMapping) {
  LinUCBConfig cfg = DefaultConfig();
  LinUCBRouter router(cfg);
  // LOCAL reward = TPOT headroom against the safety-margined target
  // (tpot_target_us * 0.85 = 42500us). IdleContext step = 10000us.
  EXPECT_NEAR(router.LocalReward(IdleContext()), (50000.0 * 0.85 - 10000.0) / (50000.0 * 0.85), 1e-12);
  LocalArmContext over;
  over.step_latency_us = 60000;  // 60ms, past both 50ms and the 42.5ms margin -> clamped to 0
  EXPECT_DOUBLE_EQ(router.LocalReward(over), 0.0);
  // REMOTE reward = 1 - elapsed_us/ref, ref = ttft_threshold_us (default 1.5s
  // -> ref = 1.5e6 us). 0.5 at half the budget, 1.0 at 0, 0 at the budget.
  EXPECT_NEAR(router.RemoteReward(750000.0), 0.5, 1e-12);
  EXPECT_DOUBLE_EQ(router.RemoteReward(0.0), 1.0);
  EXPECT_DOUBLE_EQ(router.RemoteReward(1500000.0), 0.0);
}

TEST(LinUCBRouterTest, LearnsLocalWhenIdleAndHeadroomy) {
  LinUCBConfig cfg = DefaultConfig();
  cfg.alpha = 0.0;  // pure exploitation so learned means decide
  LinUCBRouter router(cfg);

  const LocalArmContext idle = IdleContext();
  const LinUCBVec x = router.BuildFeatures(idle);
  // LOCAL kept and TPOT stayed healthy -> high reward; REMOTE would have been
  // a slow round-trip -> low reward.
  for (int i = 0; i < 50; ++i) {
    router.RecordReward(RouteArm::kLocal, x, /*reward=*/1.0);
    router.RecordReward(RouteArm::kRemotePeer, x, /*reward=*/0.1);
  }
  EXPECT_EQ(router.SelectArm(idle, nullptr), RouteArm::kLocal);
  EXPECT_GT(router.ScoreForTest(RouteArm::kLocal, x), router.ScoreForTest(RouteArm::kRemotePeer, x));
}

TEST(LinUCBRouterTest, LearnsRemoteWhenBusy) {
  LinUCBConfig cfg = DefaultConfig();
  cfg.alpha = 0.0;
  LinUCBRouter router(cfg);

  const LocalArmContext busy = BusyContext();
  const LinUCBVec x = router.BuildFeatures(busy);
  // Forwarding paid off (fast peer) while keeping local would have blown TPOT.
  for (int i = 0; i < 50; ++i) {
    router.RecordReward(RouteArm::kRemotePeer, x, /*reward=*/1.0);
    router.RecordReward(RouteArm::kLocal, x, /*reward=*/0.1);
  }
  EXPECT_EQ(router.SelectArm(busy, nullptr), RouteArm::kRemotePeer);
}

TEST(LinUCBRouterTest, LargeAlphaExploresLessTriedArm) {
  LinUCBConfig cfg = DefaultConfig();
  cfg.alpha = 100.0;  // exploration dominates
  LinUCBRouter router(cfg);

  const LocalArmContext idle = IdleContext();
  const LinUCBVec x = router.BuildFeatures(idle);
  // Train only LOCAL (shrinks its uncertainty); REMOTE stays at Ainv=I, so its
  // exploration bonus is larger. High alpha should send us to the unexplored
  // REMOTE arm even though LOCAL has the only nonzero learned mean.
  for (int i = 0; i < 30; ++i) {
    router.RecordReward(RouteArm::kLocal, x, /*reward=*/1.0);
  }
  EXPECT_EQ(router.SelectArm(idle, nullptr), RouteArm::kRemotePeer);
}

TEST(LinUCBRouterTest, NoNaNUnderManyUpdates) {
  LinUCBRouter router(DefaultConfig());
  LocalArmContext c = IdleContext();
  for (int i = 0; i < 500; ++i) {
    c.running_reqs = static_cast<uint32_t>(i % 70);
    c.step_latency_us = static_cast<uint32_t>((i * 137) % 60000);
    c.remaining_prefill_tokens = static_cast<uint32_t>((i * 911) % 9000);
    LinUCBVec feat;
    RouteArm arm = router.SelectArm(c, &feat);
    const double r = (arm == RouteArm::kLocal) ? router.LocalReward(c) : router.RemoteReward(i % 300);
    router.RecordReward(arm, feat, r);
    EXPECT_TRUE(std::isfinite(router.ScoreForTest(RouteArm::kLocal, feat)));
    EXPECT_TRUE(std::isfinite(router.ScoreForTest(RouteArm::kRemotePeer, feat)));
  }
}

TEST(LinUCBRouterTest, SelectArmWritesFeatures) {
  LinUCBRouter router(DefaultConfig());
  LinUCBVec feat{};
  router.SelectArm(IdleContext(), &feat);
  EXPECT_DOUBLE_EQ(feat[0], 1.0);
  EXPECT_GT(feat[2], 0.0);
}

}  // namespace
}  // namespace pd_v2
}  // namespace ksana_llm
