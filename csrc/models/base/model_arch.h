/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <string>
#include <vector>

#include "utils/status.h"

namespace ksana_llm {

enum class ModelArchitecture {
  // llama
  ARCH_LLAMA,
  // qwen2_moe
  ARCH_QWEN2_MOE,
  // qwen3_moe,
  ARCH_QWEN3_MOE,
  // qwen2_vl
  ARCH_QWEN2_VL,
  // qwen
  ARCH_QWEN,
  // chatglm
  ARCH_CHATGLM,
  // bge reranker
  ARCH_MINICPM,
  // mixtral
  ARCH_MIXTRAL,
  // DeepSeek V2/V3/R1/V32 or Kimi K2 or GLM-5
  ARCH_DEEPSEEK,
  // internlm2
  ARCH_INTERNLM2,
  // internlmxcomposer2
  ARCH_INTERNLMXCOMPOSER2,
  // arc_hunyuan_video
  ARCH_ARC_HUNYUAN_VIDEO,
  // Hunyuan Turbo
  ARCH_HUNYUAN_TURBO,
  // GLM4-MoE
  ARCH_GLM4_MOE,
  // DeepSeek V4
  ARCH_DEEPSEEK_V4,
  // Kimi K3
  ARCH_KIMI_K3,
  // unknown
  ARCH_UNKNOWN,
};

// Get model architecture.
// If architectures list is provided, it may be used to resolve ambiguous model types.
Status GetModelArchitectureFromString(const std::string& model_type, ModelArchitecture& model_arch,
                                      const std::vector<std::string>& architectures = {});

// Get model type string.
Status GetModelTypeFromArchitecture(ModelArchitecture model_arch, std::string& model_type);

}  // namespace ksana_llm
