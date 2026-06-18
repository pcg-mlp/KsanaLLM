/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <unordered_map>
#include <string>

namespace ksana_llm {

// Per-M tuned fused_moe kernel configs for specific model/GPU combinations.
// These configs are obtained via deep_tune (triton autotuning) and provide
// significantly better performance than the default fixed configs.
//
// Only call this for BF16 non-quantized fused_moe on supported configurations.
// Returns true if a tuned config was found, false to fall back to default.
bool GetTunedFusedMoeConfig(int num_experts, int inter_size, int num_tokens,
                            std::unordered_map<std::string, int>& config);

// Select fused_moe kernel config for BF16 non-quantized path.
// Tries tuned config first; falls back to default if not matched.
void GetFusedMoeConfig(int num_experts_per_node, int inter_size, int M, bool is_marlin,
                       std::unordered_map<std::string, int>& config);

}  // namespace ksana_llm
