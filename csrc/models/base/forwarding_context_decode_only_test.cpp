/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

// Unit tests for decode-only forwarding buffer shrink logic:
//   - ComputeDecodeOnlyStepTokenNum()   (pure function)
//   - SelectEffectiveStepTokenNum()     (pure function, dispatches by is_decode_only)
//   - FormatDecodeOnlyShrinkLog()       (pure function, log line composer)
//   - RuntimeConfig::is_decode_only     (set via ScheduleConfigParser)

#include "models/base/forwarding_context.h"

#include <gtest/gtest.h>

#include "configure/schedule_config_parser.h"
#include "connector/config.h"

namespace ksana_llm {

namespace {

// Helper：按 schedule_config_parser 一致的公式构造 max_decode_tokens_per_req。
constexpr size_t MaxDecodeTokensPerReq(size_t mtp_step_num, size_t ptp_step_num) {
  return mtp_step_num + 2 * ptp_step_num + 1;
}

BatchSchedulerConfig MakeBSC(size_t mtp_step_num, size_t ptp_step_num) {
  BatchSchedulerConfig bsc;
  bsc.mtp_step_num = mtp_step_num;
  bsc.ptp_step_num = ptp_step_num;
  bsc.max_decode_tokens_per_req = MaxDecodeTokensPerReq(mtp_step_num, ptp_step_num);
  return bsc;
}

}  // namespace

// -----------------------------------------------------------------------------
// ComputeDecodeOnlyStepTokenNum：参数语义已收敛为 max_decode_tokens_per_req，
// 该值已综合 MTP / PTP / generated token，与 schedule_config_parser 保持一致。
// -----------------------------------------------------------------------------
TEST(DecodeOnlyBufferShrinkTest, ComputeStepTokenNum_TypicalDeepSeekR1_Mtp2) {
  // DSR1 TP=8/DP=8, batch=1024, MTP=2, PTP=0 → max_decode_tokens_per_req=3
  // 1024 * 3 * 8 = 24576
  EXPECT_EQ(ComputeDecodeOnlyStepTokenNum(1024, MaxDecodeTokensPerReq(2, 0), 8), 24576u);
}

TEST(DecodeOnlyBufferShrinkTest, ComputeStepTokenNum_TypicalDP1_Mtp2) {
  // DP=1: batch=512, MTP=2 → max_decode_tokens_per_req=3 → 512*3*1 = 1536
  EXPECT_EQ(ComputeDecodeOnlyStepTokenNum(512, MaxDecodeTokensPerReq(2, 0), 1), 1536u);
}

TEST(DecodeOnlyBufferShrinkTest, ComputeStepTokenNum_PtpOnly) {
  // 纯 PTP：mtp=0, ptp=2 → max_decode_tokens_per_req = 0 + 2*2 + 1 = 5
  // 这是修复点：旧公式只用 mtp_step_num 会严重低估 PTP 场景
  EXPECT_EQ(ComputeDecodeOnlyStepTokenNum(256, MaxDecodeTokensPerReq(0, 2), 1), 256u * 5);
  EXPECT_EQ(ComputeDecodeOnlyStepTokenNum(64, MaxDecodeTokensPerReq(0, 4), 8), 64u * 9 * 8);
}

TEST(DecodeOnlyBufferShrinkTest, ComputeStepTokenNum_NoMtpNoPtp) {
  // mtp=0, ptp=0 → max_decode_tokens_per_req=1（仅 generated token）
  EXPECT_EQ(ComputeDecodeOnlyStepTokenNum(256, MaxDecodeTokensPerReq(0, 0), 1), 256u);
  EXPECT_EQ(ComputeDecodeOnlyStepTokenNum(64, MaxDecodeTokensPerReq(0, 0), 8), 512u);
}

TEST(DecodeOnlyBufferShrinkTest, ComputeStepTokenNum_ZeroDecodeTokensClampedToOne) {
  // 防御性：异常配置下 max_decode_tokens_per_req=0 时不应使 buffer=0
  EXPECT_EQ(ComputeDecodeOnlyStepTokenNum(128, 0, 4), 128u * 4);
}

TEST(DecodeOnlyBufferShrinkTest, ComputeStepTokenNum_HighMtp) {
  // mtp=4, ptp=0 → max_decode_tokens_per_req=5
  EXPECT_EQ(ComputeDecodeOnlyStepTokenNum(128, MaxDecodeTokensPerReq(4, 0), 2), 128u * 5 * 2);
}

TEST(DecodeOnlyBufferShrinkTest, ComputeStepTokenNum_MuchSmallerThanPrefillChunk) {
  // 验证“比 max_step_token_num 小很多”这一核心前提仍然成立（即便算上 +1 generated）。
  const size_t prefill_chunk = 131072;
  const size_t decode_only_tokens = ComputeDecodeOnlyStepTokenNum(512, MaxDecodeTokensPerReq(2, 0), 1);
  EXPECT_LT(decode_only_tokens, prefill_chunk);
  EXPECT_GT(prefill_chunk / decode_only_tokens, 80u);  // 仍至少 80×
}

// -----------------------------------------------------------------------------
// RuntimeConfig::is_decode_only 由 group_role 自动派生
// -----------------------------------------------------------------------------
TEST(DecodeOnlyBufferShrinkTest, IsDecodeOnlyFlag_DefaultFalse) {
  RuntimeConfig rt;
  EXPECT_FALSE(rt.is_decode_only);
}

TEST(DecodeOnlyBufferShrinkTest, IsDecodeOnlyFlag_SettableAndReadable) {
  RuntimeConfig rt;
  rt.is_decode_only = true;
  EXPECT_TRUE(rt.is_decode_only);
  rt.is_decode_only = false;
  EXPECT_FALSE(rt.is_decode_only);
}

TEST(DecodeOnlyBufferShrinkTest, IsDecodeOnlyFlag_DerivationRule) {
  // 复刻 schedule_config_parser.cpp 中的赋值语义
  auto derive_is_decode_only = [](GroupRole role) { return role == GroupRole::DECODE; };

  EXPECT_FALSE(derive_is_decode_only(GroupRole::NONE));
  EXPECT_FALSE(derive_is_decode_only(GroupRole::PREFILL));
  EXPECT_TRUE(derive_is_decode_only(GroupRole::DECODE));
  EXPECT_FALSE(derive_is_decode_only(GroupRole::BOTH));
}

// -----------------------------------------------------------------------------
// SelectEffectiveStepTokenNum：复现 Init 里的分发决策
// -----------------------------------------------------------------------------
TEST(DecodeOnlyBufferShrinkTest, InitDecisionBranch_ShrinkAppliedOnlyForDecodeOnly) {
  RuntimeConfig rt;
  rt.max_batch_size = 512;
  rt.max_step_token_num = 131072;
  rt.parallel_basic_config.attn_data_parallel_size = 1;
  const auto bsc_mtp2 = MakeBSC(/*mtp=*/2, /*ptp=*/0);  // max_decode_tokens_per_req=3

  // 非 decode-only：保留原 max_step_token_num
  rt.is_decode_only = false;
  EXPECT_EQ(SelectEffectiveStepTokenNum(rt, bsc_mtp2), 131072u);

  // decode-only：缩减到 batch * max_decode_tokens_per_req * dp = 512*3*1 = 1536
  rt.is_decode_only = true;
  EXPECT_EQ(SelectEffectiveStepTokenNum(rt, bsc_mtp2), 1536u);

  // 边界：mtp=ptp=0（max_decode_tokens_per_req=1）不应使 buffer=0
  EXPECT_EQ(SelectEffectiveStepTokenNum(rt, MakeBSC(0, 0)), 512u);

  // 边界：DP=8、batch=1024、MTP=2 → 1024*3*8=24576
  rt.parallel_basic_config.attn_data_parallel_size = 8;
  rt.max_batch_size = 1024;
  rt.max_step_token_num = 65536;
  EXPECT_EQ(SelectEffectiveStepTokenNum(rt, bsc_mtp2), 24576u);
  EXPECT_LT(SelectEffectiveStepTokenNum(rt, bsc_mtp2), rt.max_step_token_num);
}

TEST(DecodeOnlyBufferShrinkTest, SelectEffectiveStepTokenNum_NonDecodeOnlyReturnsMaxStepTokenNum) {
  RuntimeConfig rt;
  rt.is_decode_only = false;
  rt.max_step_token_num = 4096;
  rt.max_batch_size = 32;
  rt.parallel_basic_config.attn_data_parallel_size = 2;
  const auto bsc = MakeBSC(/*mtp=*/2, /*ptp=*/0);
  EXPECT_EQ(SelectEffectiveStepTokenNum(rt, bsc), 4096u);

  // 即便 decode-only 的公式结果可能更大，非 decode-only 也不应走那条路径
  rt.max_step_token_num = 10;
  EXPECT_EQ(SelectEffectiveStepTokenNum(rt, bsc), 10u);
}

TEST(DecodeOnlyBufferShrinkTest, SelectEffectiveStepTokenNum_PtpDecodeOnlyUsesPtpFormula) {
  // 关键回归测试：PTP 场景下 decode-only 必须按 max_decode_tokens_per_req 计算（而非 mtp_step_num）
  RuntimeConfig rt;
  rt.is_decode_only = true;
  rt.max_batch_size = 256;
  rt.max_step_token_num = 65536;
  rt.parallel_basic_config.attn_data_parallel_size = 1;
  const auto bsc_ptp4 = MakeBSC(/*mtp=*/0, /*ptp=*/4);  // max_decode_tokens_per_req=9
  EXPECT_EQ(SelectEffectiveStepTokenNum(rt, bsc_ptp4), 256u * 9);
}

// -----------------------------------------------------------------------------
// FormatDecodeOnlyShrinkLog：日志包含定位排障所需关键字段
// -----------------------------------------------------------------------------
TEST(DecodeOnlyBufferShrinkTest, FormatDecodeOnlyShrinkLog_ContainsAllKeyFields) {
  RuntimeConfig rt;
  rt.max_batch_size = 512;
  rt.max_step_token_num = 131072;
  rt.parallel_basic_config.attn_data_parallel_size = 1;
  const auto bsc = MakeBSC(/*mtp=*/2, /*ptp=*/0);  // max_decode_tokens_per_req=3

  const std::string log = FormatDecodeOnlyShrinkLog(rt, bsc, 1536);
  // 缩减前后值
  EXPECT_NE(log.find("131072"), std::string::npos);
  EXPECT_NE(log.find("1536"), std::string::npos);
  // 决定因子
  EXPECT_NE(log.find("max_batch_size=512"), std::string::npos);
  EXPECT_NE(log.find("max_decode_tokens_per_req=3"), std::string::npos);
  EXPECT_NE(log.find("mtp=2"), std::string::npos);
  EXPECT_NE(log.find("ptp=0"), std::string::npos);
  EXPECT_NE(log.find("attn_dp=1"), std::string::npos);
  // 路径标记，方便 grep
  EXPECT_NE(log.find("decode-only"), std::string::npos);
  EXPECT_NE(log.find("shrink"), std::string::npos);
}

TEST(DecodeOnlyBufferShrinkTest, FormatDecodeOnlyShrinkLog_HandlesPtpAndDP8) {
  RuntimeConfig rt;
  rt.max_batch_size = 1024;
  rt.max_step_token_num = 65536;
  rt.parallel_basic_config.attn_data_parallel_size = 8;
  const auto bsc = MakeBSC(/*mtp=*/0, /*ptp=*/3);  // max_decode_tokens_per_req=7

  const std::string log = FormatDecodeOnlyShrinkLog(rt, bsc, 12345);
  EXPECT_NE(log.find("65536"), std::string::npos);
  EXPECT_NE(log.find("12345"), std::string::npos);
  EXPECT_NE(log.find("max_batch_size=1024"), std::string::npos);
  EXPECT_NE(log.find("max_decode_tokens_per_req=7"), std::string::npos);
  EXPECT_NE(log.find("mtp=0"), std::string::npos);
  EXPECT_NE(log.find("ptp=3"), std::string::npos);
  EXPECT_NE(log.find("attn_dp=8"), std::string::npos);
}

}  // namespace ksana_llm
