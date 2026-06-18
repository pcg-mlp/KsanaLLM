/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "kernels/nvidia/fused_moe_tuned_config.h"

#include <gtest/gtest.h>
#include <unordered_map>
#include <string>

namespace ksana_llm {

TEST(TunedFusedMoeConfigTest, ReturnsTrueForQwen3Moe) {
  std::unordered_map<std::string, int> config;
  EXPECT_TRUE(GetTunedFusedMoeConfig(128, 768, 1, config));
  EXPECT_TRUE(GetTunedFusedMoeConfig(128, 768, 32, config));
}

TEST(TunedFusedMoeConfigTest, ReturnsFalseForNonQwen3Models) {
  std::unordered_map<std::string, int> config;
  EXPECT_FALSE(GetTunedFusedMoeConfig(256, 1536, 32, config));  // DeepSeek V3
  EXPECT_FALSE(GetTunedFusedMoeConfig(64, 768, 32, config));    // EP=2 scenario
  EXPECT_FALSE(GetTunedFusedMoeConfig(128, 1536, 32, config));  // wrong inter_size
  EXPECT_FALSE(GetTunedFusedMoeConfig(8, 768, 1, config));
}

TEST(TunedFusedMoeConfigTest, DoesNotMutateConfigOnFalse) {
  std::unordered_map<std::string, int> config = {{"sentinel", 42}};
  EXPECT_FALSE(GetTunedFusedMoeConfig(64, 768, 32, config));
  EXPECT_EQ(config.size(), 1u);
  EXPECT_EQ(config["sentinel"], 42);
}

TEST(TunedFusedMoeConfigTest, TinyDecodeConfig) {
  std::unordered_map<std::string, int> config;
  ASSERT_TRUE(GetTunedFusedMoeConfig(128, 768, 1, config));
  EXPECT_EQ(config["block_size_m"], 16);
  EXPECT_EQ(config["block_size_n"], 64);
  EXPECT_EQ(config["block_size_k"], 64);
  EXPECT_EQ(config["group_size_m"], 1);

  ASSERT_TRUE(GetTunedFusedMoeConfig(128, 768, 4, config));
  EXPECT_EQ(config["block_size_m"], 16);
  EXPECT_EQ(config["block_size_n"], 64);
}

TEST(TunedFusedMoeConfigTest, SmallPrefillConfig) {
  std::unordered_map<std::string, int> config;
  ASSERT_TRUE(GetTunedFusedMoeConfig(128, 768, 5, config));
  EXPECT_EQ(config["block_size_m"], 16);
  EXPECT_EQ(config["block_size_n"], 128);
  EXPECT_EQ(config["block_size_k"], 128);
  EXPECT_EQ(config["group_size_m"], 1);

  ASSERT_TRUE(GetTunedFusedMoeConfig(128, 768, 256, config));
  EXPECT_EQ(config["block_size_n"], 128);
}

TEST(TunedFusedMoeConfigTest, LargePrefillConfig) {
  std::unordered_map<std::string, int> config;
  ASSERT_TRUE(GetTunedFusedMoeConfig(128, 768, 257, config));
  EXPECT_EQ(config["block_size_m"], 64);
  EXPECT_EQ(config["block_size_n"], 64);
  EXPECT_EQ(config["block_size_k"], 64);
  EXPECT_EQ(config["group_size_m"], 1);

  ASSERT_TRUE(GetTunedFusedMoeConfig(128, 768, 1024, config));
  EXPECT_EQ(config["block_size_m"], 64);
}

TEST(TunedFusedMoeConfigTest, FallbackConfigWhenNotMatched) {
  std::unordered_map<std::string, int> config;
  // DeepSeek V3: num_experts=256, inter_size=1536, M=512 -> fallback large-M
  GetFusedMoeConfig(256, 1536, 512, false, config);
  EXPECT_EQ(config["block_size_m"], 64);
  EXPECT_EQ(config["block_size_n"], 64);
  EXPECT_EQ(config["block_size_k"], 32);
  EXPECT_EQ(config["group_size_m"], 8);
}

TEST(TunedFusedMoeConfigTest, FallbackSmallMConfig) {
  std::unordered_map<std::string, int> config;
  // M <= num_experts_per_node -> fallback small-M config
  GetFusedMoeConfig(256, 1536, 32, true, config);
  EXPECT_EQ(config["block_size_m"], 16);
  EXPECT_EQ(config["block_size_n"], 32);
  EXPECT_EQ(config["block_size_k"], 64);
  EXPECT_EQ(config["group_size_m"], 1);
}

TEST(TunedFusedMoeConfigTest, TunedConfigTakesPriority) {
  std::unordered_map<std::string, int> config;
  // Qwen3 params -> should get tuned config, not fallback
  GetFusedMoeConfig(128, 768, 32, false, config);
  EXPECT_EQ(config["block_size_m"], 16);
  EXPECT_EQ(config["block_size_n"], 128);
  EXPECT_EQ(config["block_size_k"], 128);
  EXPECT_EQ(config["group_size_m"], 1);
}

}  // namespace ksana_llm
