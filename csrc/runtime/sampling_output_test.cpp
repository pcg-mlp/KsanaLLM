/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include <gtest/gtest.h>

#include "runtime/sampling_output.h"

namespace ksana_llm {
namespace {

// 基本序列化/反序列化：所有字段非空
TEST(GenerationResultTest, SerializeAndDeserializeBasic) {
  GenerationResult src;
  src.accepted_tokens = {1, 2, 3};
  src.generated_tokens = {10, 20};
  src.draft_tokens.mtp = {100, 200, 300};
  src.draft_tokens.spec = {400, 500};
  src.draft_tokens.ptp = {600, 700};

  std::vector<int> encoded;
  src.SerializeToVector(encoded);

  // 预期编码长度: 5 个 size 字段 + 3 + 2 + 3 + 2 + 2 = 17
  EXPECT_EQ(encoded.size(), 17u);

  GenerationResult dst;
  dst.DeserializeFromVector(encoded);

  EXPECT_EQ(dst.accepted_tokens, src.accepted_tokens);
  EXPECT_EQ(dst.generated_tokens, src.generated_tokens);
  EXPECT_EQ(dst.draft_tokens.mtp, src.draft_tokens.mtp);
  EXPECT_EQ(dst.draft_tokens.spec, src.draft_tokens.spec);
  EXPECT_EQ(dst.draft_tokens.ptp, src.draft_tokens.ptp);
}

// 所有字段为空
TEST(GenerationResultTest, SerializeAndDeserializeEmpty) {
  GenerationResult src;

  std::vector<int> encoded;
  src.SerializeToVector(encoded);

  // 仅 5 个 size 字段，每个值为 0
  ASSERT_EQ(encoded.size(), 5u);
  for (int v : encoded) {
    EXPECT_EQ(v, 0);
  }

  GenerationResult dst;
  dst.DeserializeFromVector(encoded);

  EXPECT_TRUE(dst.accepted_tokens.empty());
  EXPECT_TRUE(dst.generated_tokens.empty());
  EXPECT_TRUE(dst.draft_tokens.mtp.empty());
  EXPECT_TRUE(dst.draft_tokens.spec.empty());
  EXPECT_TRUE(dst.draft_tokens.ptp.empty());
}

// 部分字段为空，部分非空
TEST(GenerationResultTest, SerializeAndDeserializePartialEmpty) {
  GenerationResult src;
  src.accepted_tokens = {7, 8, 9, 10, 11};
  // generated_tokens 为空
  src.draft_tokens.mtp = {42};
  // draft_tokens.spec 为空
  // draft_tokens.ptp 为空

  std::vector<int> encoded;
  src.SerializeToVector(encoded);

  // 5 个 size 字段 + 5 + 0 + 1 + 0 + 0 = 11
  EXPECT_EQ(encoded.size(), 11u);

  GenerationResult dst;
  dst.DeserializeFromVector(encoded);

  EXPECT_EQ(dst.accepted_tokens, src.accepted_tokens);
  EXPECT_TRUE(dst.generated_tokens.empty());
  EXPECT_EQ(dst.draft_tokens.mtp, src.draft_tokens.mtp);
  EXPECT_TRUE(dst.draft_tokens.spec.empty());
  EXPECT_TRUE(dst.draft_tokens.ptp.empty());
}

// 反序列化应覆盖已有数据
TEST(GenerationResultTest, DeserializeOverwritesExistingData) {
  GenerationResult src;
  src.accepted_tokens = {1};
  src.generated_tokens = {2};
  src.draft_tokens.mtp = {3};
  src.draft_tokens.spec = {4};
  src.draft_tokens.ptp = {5};

  std::vector<int> encoded;
  src.SerializeToVector(encoded);

  // dst 预先填充不同的数据
  GenerationResult dst;
  dst.accepted_tokens = {99, 98, 97};
  dst.generated_tokens = {88, 87};
  dst.draft_tokens.mtp = {77, 76, 75};
  dst.draft_tokens.spec = {66, 65, 64, 63};
  dst.draft_tokens.ptp = {55, 54};

  dst.DeserializeFromVector(encoded);

  EXPECT_EQ(dst.accepted_tokens, src.accepted_tokens);
  EXPECT_EQ(dst.generated_tokens, src.generated_tokens);
  EXPECT_EQ(dst.draft_tokens.mtp, src.draft_tokens.mtp);
  EXPECT_EQ(dst.draft_tokens.spec, src.draft_tokens.spec);
  EXPECT_EQ(dst.draft_tokens.ptp, src.draft_tokens.ptp);
}

// 单元素场景
TEST(GenerationResultTest, SerializeAndDeserializeSingleElement) {
  GenerationResult src;
  src.accepted_tokens = {42};
  src.generated_tokens = {0};
  src.draft_tokens.mtp = {-1};
  src.draft_tokens.spec = {INT32_MAX};
  src.draft_tokens.ptp = {0};

  std::vector<int> encoded;
  src.SerializeToVector(encoded);

  GenerationResult dst;
  dst.DeserializeFromVector(encoded);

  EXPECT_EQ(dst.accepted_tokens, src.accepted_tokens);
  EXPECT_EQ(dst.generated_tokens, src.generated_tokens);
  EXPECT_EQ(dst.draft_tokens.mtp, src.draft_tokens.mtp);
  EXPECT_EQ(dst.draft_tokens.spec, src.draft_tokens.spec);
  EXPECT_EQ(dst.draft_tokens.ptp, src.draft_tokens.ptp);
}

// 多次序列化/反序列化循环：确保幂等
TEST(GenerationResultTest, RoundTripMultipleTimes) {
  GenerationResult src;
  src.accepted_tokens = {5, 6, 7};
  src.generated_tokens = {15, 16};
  src.draft_tokens.mtp = {25};
  src.draft_tokens.spec = {35, 36};
  src.draft_tokens.ptp = {45, 46, 47};

  std::vector<int> encoded;
  for (int i = 0; i < 3; ++i) {
    src.SerializeToVector(encoded);

    GenerationResult dst;
    dst.DeserializeFromVector(encoded);

    EXPECT_EQ(dst.accepted_tokens, src.accepted_tokens);
    EXPECT_EQ(dst.generated_tokens, src.generated_tokens);
    EXPECT_EQ(dst.draft_tokens.mtp, src.draft_tokens.mtp);
    EXPECT_EQ(dst.draft_tokens.spec, src.draft_tokens.spec);
    EXPECT_EQ(dst.draft_tokens.ptp, src.draft_tokens.ptp);

    // 用 dst 作为下一轮的 src
    src = dst;
  }
}

}  // namespace
}  // namespace ksana_llm
