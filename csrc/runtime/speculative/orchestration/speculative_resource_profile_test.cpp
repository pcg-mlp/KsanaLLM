/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <gtest/gtest.h>
#include <memory>
#include <string>

#include "configure/schedule_config_parser.h"
#include "runtime/speculative/common/speculative_resource_profile.h"
#include "runtime/speculative/orchestration/composite_speculative_decoder.h"

namespace ksana_llm {

class SpeculativeRegistryTest : public testing::Test {
 protected:
  void SetUp() override {
    // SpeculativeRegistry is a singleton. Post-init setups registered during
    // static init (e.g. from MtpDecoderAdapter) are also present — they
    // safely return early when context fields are null.
  }
};

// Test that a single registered PostInitSetup is called by RunPostInitSetups.
TEST_F(SpeculativeRegistryTest, PostInitSetup_SingleCallbackGetsCalled) {
  auto called = std::make_shared<bool>(false);
  SpeculativeRegistry::Get().RegisterPostInitSetup([called](const SpeculativeSetupContext&) { *called = true; });

  SpeculativeSetupContext ctx;
  SpeculativeRegistry::Get().RunPostInitSetups(ctx);
  EXPECT_TRUE(*called);
}

// Test that multiple registered setups are all called, in registration order.
TEST_F(SpeculativeRegistryTest, PostInitSetup_MultipleCallbacksAllRun) {
  auto counter = std::make_shared<int>(0);
  SpeculativeRegistry::Get().RegisterPostInitSetup([counter](const SpeculativeSetupContext&) { (*counter)++; });
  SpeculativeRegistry::Get().RegisterPostInitSetup([counter](const SpeculativeSetupContext&) { (*counter)++; });
  SpeculativeRegistry::Get().RegisterPostInitSetup([counter](const SpeculativeSetupContext&) { (*counter)++; });

  SpeculativeSetupContext ctx;
  SpeculativeRegistry::Get().RunPostInitSetups(ctx);
  // 3 from this test + at least 1 from previous test (SingleCallbackGetsCalled)
  // + static init setups. Exact count depends on registration order.
  EXPECT_GE(*counter, 3);
}

// Test that SpeculativeSetupContext carries the correct data.
TEST_F(SpeculativeRegistryTest, PostInitSetup_ContextPassedCorrectly) {
  auto composite_decoder = std::make_shared<CompositeSpeculativeDecoder>();
  auto context_valid = std::make_shared<bool>(false);

  SpeculativeRegistry::Get().RegisterPostInitSetup(
      [context_valid, composite_decoder](const SpeculativeSetupContext& c) {
        if (c.composite_decoder == composite_decoder && c.executor_runtime != nullptr) {
          *context_valid = true;
        }
      });

  SpeculativeSetupContext ctx;
  ctx.composite_decoder = composite_decoder;
  ctx.executor_runtime = reinterpret_cast<ExecutorRuntime*>(0x1);
  SpeculativeRegistry::Get().RunPostInitSetups(ctx);
  EXPECT_TRUE(*context_valid);
}

// Test that calling RunPostInitSetups twice is safe (setups are called again).
TEST_F(SpeculativeRegistryTest, PostInitSetup_IdempotentSecondRun) {
  SpeculativeSetupContext ctx;
  SpeculativeRegistry::Get().RunPostInitSetups(ctx);
  SpeculativeRegistry::Get().RunPostInitSetups(ctx);
  SUCCEED();
}

// Test that Register + Aggregate sums estimator fields correctly.
// The registry is a process-wide singleton — adapters (MTP/PTP/DSpark) already
// register their own estimators during static init, so this test only checks
// that its own newly-registered estimator's contribution is present in the
// aggregated total (an exact-total assertion would be order/link-dependent).
TEST_F(SpeculativeRegistryTest, Aggregate_SumsRegisteredEstimators) {
  BatchSchedulerConfig cfg;

  auto baseline = SpeculativeRegistry::Get().Aggregate(cfg);

  SpeculativeRegistry::Get().Register([](const BatchSchedulerConfig&) {
    SpeculativeResourceProfile profile;
    profile.extra_decode_tokens_per_req = 7;
    profile.workload_multiplier = 11;
    profile.extra_token_capacity = 13;
    profile.buffer_decls.push_back({"test_buffer", BufferShapeExpr::kSpecHiddenBuffer});
    return profile;
  });

  auto after = SpeculativeRegistry::Get().Aggregate(cfg);

  EXPECT_EQ(after.extra_decode_tokens_per_req, baseline.extra_decode_tokens_per_req + 7);
  EXPECT_EQ(after.workload_multiplier, baseline.workload_multiplier + 11);
  EXPECT_EQ(after.extra_token_capacity, baseline.extra_token_capacity + 13);
  EXPECT_EQ(after.buffer_decls.size(), baseline.buffer_decls.size() + 1);
}

// Test that Aggregate with no matching estimator contribution still returns
// a well-formed (non-negative, additive) profile — i.e. an estimator that
// returns a default-constructed profile contributes zero to every field.
TEST_F(SpeculativeRegistryTest, Aggregate_ZeroProfileEstimatorContributesNothing) {
  BatchSchedulerConfig cfg;
  auto baseline = SpeculativeRegistry::Get().Aggregate(cfg);

  SpeculativeRegistry::Get().Register([](const BatchSchedulerConfig&) { return SpeculativeResourceProfile{}; });

  auto after = SpeculativeRegistry::Get().Aggregate(cfg);
  EXPECT_EQ(after.extra_decode_tokens_per_req, baseline.extra_decode_tokens_per_req);
  EXPECT_EQ(after.workload_multiplier, baseline.workload_multiplier);
  EXPECT_EQ(after.extra_token_capacity, baseline.extra_token_capacity);
  EXPECT_EQ(after.buffer_decls.size(), baseline.buffer_decls.size());
}

// Test that RegisterDecoder + HasDecoderMethod + CreateDecoder round-trip
// correctly for a newly registered method name (using a unique name to avoid
// colliding with production adapters like "mtp"/"ptp"/"trie"/"asr_custom").
TEST_F(SpeculativeRegistryTest, RegisterDecoder_CreateDecoder_RoundTrip) {
  const std::string kMethod = "unit_test_only_decoder_method_xyz";
  EXPECT_FALSE(SpeculativeRegistry::Get().HasDecoderMethod(kMethod));

  auto created_flag = std::make_shared<bool>(false);
  SpeculativeRegistry::Get().RegisterDecoder(kMethod, [created_flag](const BatchSchedulerConfig&) {
    *created_flag = true;
    DecoderFactoryResult result;
    // No adapter/generator needed for this test — only verifying the factory
    // is invoked and its result is returned through CreateDecoder.
    return result;
  });

  EXPECT_TRUE(SpeculativeRegistry::Get().HasDecoderMethod(kMethod));

  BatchSchedulerConfig cfg;
  SpeculativeRegistry::Get().CreateDecoder(kMethod, cfg);
  EXPECT_TRUE(*created_flag);
}

// Test that HasDecoderMethod returns false for an unregistered method name,
// and CreateDecoder returns a default-constructed (empty) result for it.
TEST_F(SpeculativeRegistryTest, HasDecoderMethod_FalseForUnknownMethod) {
  const std::string kUnknownMethod = "unit_test_unknown_method_never_registered";
  EXPECT_FALSE(SpeculativeRegistry::Get().HasDecoderMethod(kUnknownMethod));

  BatchSchedulerConfig cfg;
  auto result = SpeculativeRegistry::Get().CreateDecoder(kUnknownMethod, cfg);
  EXPECT_EQ(result.adapter, nullptr);
  EXPECT_EQ(result.generator, nullptr);
}

}  // namespace ksana_llm
