/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include "models/base/base_model_config_parser.h"
#include "models/mixtral/mixtral_config.h"

namespace ksana_llm {

// Model config parser for mixtral.
class MixtralConfigParser : public BaseModelConfigParser {
 public:
  MixtralConfigParser();
  virtual ~MixtralConfigParser() override;

  virtual Status ParseModelConfig(const nlohmann::json &config_json,
                                  const ParallelismBasicConfig &parallel_basic_config, const std::string &model_dir,
                                  std::shared_ptr<BaseModelConfig> &model_config) override;

 private:
  Status ParseMoeConfig(const nlohmann::json &config_json, std::shared_ptr<MixtralConfig> mixtral_config);
};

}  // namespace ksana_llm
