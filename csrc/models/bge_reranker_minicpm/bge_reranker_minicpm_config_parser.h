/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include "models/base/base_model_config_parser.h"
#include "models/bge_reranker_minicpm/bge_reranker_minicpm_config.h"

namespace ksana_llm {

// Model config parser for bge-reranker-minicpm loader.
class BgeRerankerMinicpmConfigParser : public BaseModelConfigParser {
 public:
  // Parse config from config.json
  Status ParseModelConfig(const nlohmann::json &config_json, const ParallelismBasicConfig &parallel_basic_config,
                          const std::string &model_dir, std::shared_ptr<BaseModelConfig> &model_config) override;

 private:
  // Parse quant config
  Status ParseQuantConfig(const nlohmann::json &quant_config_json,
                          std::shared_ptr<BgeRerankerMinicpmConfig> bge_reranker_minicpm_model_config,
                          const std::string &yaml_weight_quant_method);
};

}  // namespace ksana_llm
