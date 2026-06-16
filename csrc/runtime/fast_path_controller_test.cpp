/* Copyright 2026 Tencent Inc. All rights reserved.

==============================================================================*/

// 单元测试：FastPathController 启动期硬条件 check。
// 验证各配置组合下 build-ahead 是否正确启用/禁用。

#include "runtime/fast_path_controller.h"

#include "configure/model_config.h"
#include "configure/schedule_config_parser.h"
#include "test.h"

namespace ksana_llm {

namespace {

// 构造默认通过所有 check 的配置三元组（启用 build-ahead）。
struct FastPathConfigFixture {
  BatchSchedulerConfig sched_cfg;
  RuntimeConfig runtime_cfg;
  ModelConfig model_cfg;

  FastPathConfigFixture() {
    sched_cfg.enable_executor_build_ahead = true;
    sched_cfg.enable_async = true;
    sched_cfg.scheduler_type = SchedulerType::DEFAULT;
    sched_cfg.enable_speculative_decoding = false;
    sched_cfg.mtp_step_num = 0;
    sched_cfg.enable_xgrammar = false;
    sched_cfg.split_fuse_token_num = 0;

    runtime_cfg.parallel_basic_config.tensor_parallel_size = 1;
    runtime_cfg.parallel_basic_config.attn_data_parallel_size = 1;
    runtime_cfg.parallel_basic_config.expert_parallel_size = 1;
    runtime_cfg.max_pp_batch_num = 1;
    runtime_cfg.enable_prefix_caching = false;
    runtime_cfg.enable_flexible_caching = false;
    runtime_cfg.separate_prefill_decode = false;

    model_cfg.type = "qwen2";
    model_cfg.is_moe = false;
    model_cfg.use_mla = false;
    model_cfg.use_dsa = false;
    model_cfg.use_dsv4_attn = false;
    model_cfg.is_visual = false;
  }
};

}  // namespace

class FastPathControllerTest : public testing::Test {
 protected:
  void SetUp() override { FastPathController::GetInstance().ResetForTesting(); }
  void TearDown() override { FastPathController::GetInstance().ResetForTesting(); }
};

TEST_F(FastPathControllerTest, AllConditionsMetEnables) {
  FastPathConfigFixture f;
  FastPathController::GetInstance().InitializeAtStartup(f.sched_cfg, f.runtime_cfg, f.model_cfg);
  EXPECT_TRUE(FastPathController::GetInstance().IsEnabledAtStartup());
  EXPECT_TRUE(FastPathController::GetInstance().DisableReason().empty());
}

TEST_F(FastPathControllerTest, YamlSwitchOffDisables) {
  FastPathConfigFixture f;
  f.sched_cfg.enable_executor_build_ahead = false;
  FastPathController::GetInstance().InitializeAtStartup(f.sched_cfg, f.runtime_cfg, f.model_cfg);
  EXPECT_FALSE(FastPathController::GetInstance().IsEnabledAtStartup());
  EXPECT_FALSE(FastPathController::GetInstance().DisableReason().empty());
}

TEST_F(FastPathControllerTest, EnableAsyncRequired) {
  FastPathConfigFixture f;
  f.sched_cfg.enable_async = false;
  FastPathController::GetInstance().InitializeAtStartup(f.sched_cfg, f.runtime_cfg, f.model_cfg);
  EXPECT_FALSE(FastPathController::GetInstance().IsEnabledAtStartup());
}

TEST_F(FastPathControllerTest, EventDrivenSchedulerDisables) {
  FastPathConfigFixture f;
  f.sched_cfg.scheduler_type = SchedulerType::EVENT_DRIVEN;
  FastPathController::GetInstance().InitializeAtStartup(f.sched_cfg, f.runtime_cfg, f.model_cfg);
  EXPECT_FALSE(FastPathController::GetInstance().IsEnabledAtStartup());
}

TEST_F(FastPathControllerTest, TpGreaterThanOneStillEnables) {
  FastPathConfigFixture f;
  f.runtime_cfg.parallel_basic_config.tensor_parallel_size = 2;
  FastPathController::GetInstance().InitializeAtStartup(f.sched_cfg, f.runtime_cfg, f.model_cfg);
  EXPECT_TRUE(FastPathController::GetInstance().IsEnabledAtStartup());
}

TEST_F(FastPathControllerTest, DpGreaterThanOneDisables) {
  FastPathConfigFixture f;
  f.runtime_cfg.parallel_basic_config.attn_data_parallel_size = 2;
  FastPathController::GetInstance().InitializeAtStartup(f.sched_cfg, f.runtime_cfg, f.model_cfg);
  EXPECT_FALSE(FastPathController::GetInstance().IsEnabledAtStartup());
}

TEST_F(FastPathControllerTest, PpBatchNumGreaterThanOneDisables) {
  FastPathConfigFixture f;
  f.runtime_cfg.max_pp_batch_num = 2;
  FastPathController::GetInstance().InitializeAtStartup(f.sched_cfg, f.runtime_cfg, f.model_cfg);
  EXPECT_FALSE(FastPathController::GetInstance().IsEnabledAtStartup());
}

TEST_F(FastPathControllerTest, SpeculativeDecodingDisables) {
  FastPathConfigFixture f;
  f.sched_cfg.enable_speculative_decoding = true;
  FastPathController::GetInstance().InitializeAtStartup(f.sched_cfg, f.runtime_cfg, f.model_cfg);
  EXPECT_FALSE(FastPathController::GetInstance().IsEnabledAtStartup());
}

TEST_F(FastPathControllerTest, MtpStepNumDisables) {
  FastPathConfigFixture f;
  f.sched_cfg.mtp_step_num = 1;
  FastPathController::GetInstance().InitializeAtStartup(f.sched_cfg, f.runtime_cfg, f.model_cfg);
  EXPECT_FALSE(FastPathController::GetInstance().IsEnabledAtStartup());
}

TEST_F(FastPathControllerTest, XgrammarDisables) {
  FastPathConfigFixture f;
  f.sched_cfg.enable_xgrammar = true;
  FastPathController::GetInstance().InitializeAtStartup(f.sched_cfg, f.runtime_cfg, f.model_cfg);
  EXPECT_FALSE(FastPathController::GetInstance().IsEnabledAtStartup());
}

TEST_F(FastPathControllerTest, SplitFuseDisables) {
  FastPathConfigFixture f;
  f.sched_cfg.split_fuse_token_num = 256;
  FastPathController::GetInstance().InitializeAtStartup(f.sched_cfg, f.runtime_cfg, f.model_cfg);
  EXPECT_FALSE(FastPathController::GetInstance().IsEnabledAtStartup());
}

TEST_F(FastPathControllerTest, PrefixCachingDisables) {
  FastPathConfigFixture f;
  f.runtime_cfg.enable_prefix_caching = true;
  FastPathController::GetInstance().InitializeAtStartup(f.sched_cfg, f.runtime_cfg, f.model_cfg);
  EXPECT_FALSE(FastPathController::GetInstance().IsEnabledAtStartup());
}

TEST_F(FastPathControllerTest, FlexibleCachingDisables) {
  FastPathConfigFixture f;
  f.runtime_cfg.enable_flexible_caching = true;
  FastPathController::GetInstance().InitializeAtStartup(f.sched_cfg, f.runtime_cfg, f.model_cfg);
  EXPECT_FALSE(FastPathController::GetInstance().IsEnabledAtStartup());
}

TEST_F(FastPathControllerTest, SeparatePrefillDecodeDisables) {
  FastPathConfigFixture f;
  f.runtime_cfg.separate_prefill_decode = true;
  FastPathController::GetInstance().InitializeAtStartup(f.sched_cfg, f.runtime_cfg, f.model_cfg);
  EXPECT_FALSE(FastPathController::GetInstance().IsEnabledAtStartup());
}

TEST_F(FastPathControllerTest, EpGreaterThanOneDisables) {
  FastPathConfigFixture f;
  f.runtime_cfg.parallel_basic_config.expert_parallel_size = 2;
  FastPathController::GetInstance().InitializeAtStartup(f.sched_cfg, f.runtime_cfg, f.model_cfg);
  EXPECT_FALSE(FastPathController::GetInstance().IsEnabledAtStartup());
}

TEST_F(FastPathControllerTest, AnyModelTypeEnablesWhenOtherChecksMet) {
  FastPathConfigFixture f;
  f.model_cfg.type = "llama";
  f.model_cfg.is_moe = true;
  f.model_cfg.use_mla = true;
  f.model_cfg.is_visual = true;
  FastPathController::GetInstance().InitializeAtStartup(f.sched_cfg, f.runtime_cfg, f.model_cfg);
  EXPECT_TRUE(FastPathController::GetInstance().IsEnabledAtStartup());
}

}  // namespace ksana_llm
