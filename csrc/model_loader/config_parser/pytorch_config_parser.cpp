/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "model_loader/config_parser/pytorch_config_parser.h"

#include <vector>

#include "configure/json_config_utils.h"
#include "model_loader/model_loader_utils.h"
#include "utils/string_utils.h"

namespace ksana_llm {

Status PytorchConfigParser::GetJsonConfig(const std::string& model_dir, nlohmann::json& config_json,
                                          ModelArchitecture& model_arch) {
  // Get config path.
  std::filesystem::path abs_model_dir = std::filesystem::absolute(model_dir);
  std::string model_config_file = abs_model_dir.u8string() + "/config.json";

  if (!std::filesystem::exists(model_config_file)) {
    return Status(RET_INVALID_ARGUMENT, "GetJsonConfig error, config.json not found.");
  }

  config_json = ReadJsonFromFile(model_config_file);
  if (config_json.empty()) {
    return Status(RET_INVALID_ARGUMENT,
                  FormatStr("Load model config config_file %s error.", model_config_file.c_str()));
  }

  // Get model arch
  std::string model_type = config_json.at("model_type");

  // Get architectures if available for resolving ambiguous model types
  std::vector<std::string> architectures;
  if (config_json.contains("architectures")) {
    architectures = config_json["architectures"].get<std::vector<std::string>>();
  }

  if (model_type == "qwen2_vl" || model_type == "qwen2_5_vl") {
    KLLM_LOG_INFO << fmt::format("Reset model_type: {} to qwen2", model_type);
    // qwen2-vl/qwen2.5-vl的LLM部分本质上就是qwen2
    model_type = "qwen2";
  }

  if (model_type == "qwen3_vl") {
    KLLM_LOG_INFO << fmt::format("Reset model_type: {} to qwen3", model_type);
    // Qwen3-VL LLM backbone is Qwen3.
    model_type = "qwen3";
    // Qwen3-VL uses nested text_config structure. Extract text_config for config parser
    // and weight loader compatibility (all needed keys are inside text_config).
    if (config_json.contains("text_config")) {
      config_json = config_json["text_config"];
    }
  }

  // Kimi-K2.5 is a multimodal model whose LLM backbone is kimi_k2 (DeepSeek V3 variant).
  // Extract text_config which contains the actual LLM parameters.
  if (model_type == "kimi_k25") {
    KLLM_LOG_INFO << fmt::format("Reset model_type: {} to kimi_k2", model_type);
    if (config_json.contains("text_config")) {
      config_json = config_json["text_config"];
    }
    model_type = config_json.value("model_type", "kimi_k2");
  }

  if (model_type != "arc_hunyuan_video" && model_type.find("hunyuan") != std::string::npos) {
    int num_experts = config_json.value("num_experts", 1);
    if (config_json.value("use_mixed_mlp_moe", false)) {
      model_type = "hunyuan_large";
      KLLM_CHECK_WITH_INFO(num_experts > 1, "Only MoE hunyuan_large is supported.");
    } else {
      model_type = "hunyuan_turbo";
      KLLM_CHECK_WITH_INFO(num_experts == 1, "Only dense hunyuan_turbo is supported.");
    }
  }

  Status status = GetModelArchitectureFromString(model_type, model_arch, architectures);
  if (!status.OK()) {
    return status;
  }

  return Status();
}

}  // namespace ksana_llm
