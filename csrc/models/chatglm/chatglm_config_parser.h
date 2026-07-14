/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include "models/base/base_model_config_parser.h"
#include "models/chatglm/chatglm_model_config.h"

namespace ksana_llm {

// Model config parser for ChatGLM loader.
class ChatglmConfigParser : public BaseModelConfigParser {
 public:
  ChatglmConfigParser();
  virtual ~ChatglmConfigParser() override;

  // Parse config from config.json
  virtual Status ParseModelConfig(const nlohmann::json &config_json,
                                  const ParallelismBasicConfig &parallel_basic_config, const std::string &model_dir,
                                  std::shared_ptr<BaseModelConfig> &model_config) override;
};

}  // namespace ksana_llm
