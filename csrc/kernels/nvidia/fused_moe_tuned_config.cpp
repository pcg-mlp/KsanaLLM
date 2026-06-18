/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "kernels/nvidia/fused_moe_tuned_config.h"

namespace ksana_llm {

// Tuned fused_moe config for Qwen3 MOE (E=128, inter_size=768) on H20.
// Per-M optimal configs obtained via deep_tune on Qwen3-30B-A3B (BF16, TP=1).
// Hardware: NVIDIA H20 (sm_90a), triton 3.6.0.

namespace {
constexpr int kQwen3MoeNumExperts = 128;
constexpr int kQwen3MoeInterSize = 768;
constexpr int kTinyDecodeMaxTokens = 4;
constexpr int kSmallPrefillMaxTokens = 256;

static const std::unordered_map<std::string, int> kTinyDecodeConfig =
    {{"block_size_m", 16}, {"block_size_n", 64}, {"block_size_k", 64}, {"group_size_m", 1}};
static const std::unordered_map<std::string, int> kSmallPrefillConfig =
    {{"block_size_m", 16}, {"block_size_n", 128}, {"block_size_k", 128}, {"group_size_m", 1}};
static const std::unordered_map<std::string, int> kLargePrefillConfig =
    {{"block_size_m", 64}, {"block_size_n", 64}, {"block_size_k", 64}, {"group_size_m", 1}};
}  // namespace

bool GetTunedFusedMoeConfig(int num_experts, int inter_size, int num_tokens,
                            std::unordered_map<std::string, int>& config) {
  if (num_experts != kQwen3MoeNumExperts || inter_size != kQwen3MoeInterSize || num_tokens <= 0) {
    return false;
  }

  if (num_tokens <= kTinyDecodeMaxTokens) {
    config = kTinyDecodeConfig;
  } else if (num_tokens <= kSmallPrefillMaxTokens) {
    config = kSmallPrefillConfig;
  } else {
    config = kLargePrefillConfig;
  }
  return true;
}

void GetFusedMoeConfig(int num_experts_per_node, int inter_size, int M, bool is_marlin,
                       std::unordered_map<std::string, int>& config) {
  if (!GetTunedFusedMoeConfig(num_experts_per_node, inter_size, M, config)) {
    config = {{"block_size_m", 64}, {"block_size_n", 64}, {"block_size_k", 32}, {"group_size_m", 8}};
    if (M <= num_experts_per_node || (M <= 32 && is_marlin)) {
      config = {{"block_size_m", 16}, {"block_size_n", 32}, {"block_size_k", 64}, {"group_size_m", 1}};
    }
  }
}

}  // namespace ksana_llm
