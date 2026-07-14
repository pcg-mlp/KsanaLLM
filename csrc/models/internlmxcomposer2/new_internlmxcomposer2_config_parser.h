/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include "models/base/base_model_config_parser.h"
#include "models/internlmxcomposer2/internlmxcomposer2_config.h"

namespace ksana_llm {

class InternlmxComposer2ConfigParser : public BaseModelConfigParser {
 public:
  InternlmxComposer2ConfigParser();
  virtual ~InternlmxComposer2ConfigParser() override;

  virtual Status ParseModelConfig(const nlohmann::json &config_json,
                                  const ParallelismBasicConfig &parallel_basic_config, const std::string &model_dir,
                                  std::shared_ptr<BaseModelConfig> &model_config) override;

 private:
  Status ParseQuantConfig(const nlohmann::json &config_json, std::shared_ptr<InternlmxComposer2Config> config);
};

}  // namespace ksana_llm

