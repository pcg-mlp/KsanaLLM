/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/deepseek_v3/deepseek_v3_config.h"

namespace ksana_llm {

bool DeepSeekV3Config::ContainGptqWeights() const {
  if (quant_config.method == QUANT_GPTQ) {
    return true;
  } else {
    for (auto& sub_quant_config : sub_quant_configs) {
      if (sub_quant_config.method == QUANT_GPTQ) {
        return true;
      }
    }
  }
  return false;
}

QuantConfig DeepSeekV3Config::GetGptqQuantConfig() {
  if (quant_config.method == QUANT_GPTQ) {
    return quant_config;
  } else {
    for (auto& sub_quant_config : sub_quant_configs) {
      if (sub_quant_config.method == QUANT_GPTQ) {
        return sub_quant_config;
      }
    }
  }
  KLLM_THROW("Try to Get GPTQ Quant Config But Not Found.");
}

bool DeepSeekV3Config::IsWeightMatchGptq(const std::string& weight_name) {
  if (quant_config.method == QUANT_GPTQ) {
    return true;
  } else {
    for (auto& sub_quant_config : sub_quant_configs) {
      if (sub_quant_config.method == QUANT_GPTQ) {
        bool match = false;
        for (std::string& pattern_layer : sub_quant_config.pattern_layers) {
          if (weight_name.find(pattern_layer) != std::string::npos) {
            match = true;
            break;
          }
        }
        for (std::string& ignored_layer : sub_quant_config.ignored_layers) {
          if (weight_name.find(ignored_layer) != std::string::npos) {
            match = false;
            break;
          }
        }
        return match;
      }
    }
    return false;
  }
}

bool DeepSeekV3Config::IsGptqContain(const std::string& weight_name) {
  std::vector<std::string> gptq_weights_suffix = {"qweight", "scales", "qzeros", "g_idx"};
  if (GetGptqQuantConfig().input_scale) {
    gptq_weights_suffix.push_back("input_scale");
  }
  for (std::string& gptq_weight_suffix : gptq_weights_suffix) {
    if (weight_name.find(gptq_weight_suffix) != std::string::npos) {
      return true;
    }
  }
  return false;
}

}  // namespace ksana_llm
