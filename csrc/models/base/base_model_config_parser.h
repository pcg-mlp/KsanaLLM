/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "configure/schedule_config_parser.h"
#include "models/base/base_model_config.h"
#include "nlohmann/json.hpp"
#include "utils/gguf_file_utils.h"
#include "utils/status.h"

namespace ksana_llm {

// Parse model config from pytorch or gguf format.
class BaseModelConfigParser {
 public:
  virtual ~BaseModelConfigParser();

  // Parse config from config.json
  virtual Status ParseModelConfig(const nlohmann::json &config_json,
                                  const ParallelismBasicConfig &parallel_basic_config, const std::string &model_dir,
                                  std::shared_ptr<BaseModelConfig> &model_config);

  // Parse config from gguf files.
  virtual Status ParseModelConfig(const std::unordered_map<std::string, GGUFMetaValue> &gguf_meta,
                                  std::shared_ptr<BaseModelConfig> &model_config);
};

}  // namespace ksana_llm
