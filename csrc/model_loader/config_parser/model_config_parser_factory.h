/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "models/base/base_model_config_parser.h"
#include "models/base/model_arch.h"

namespace ksana_llm {

// Used to create model parser for every xxx model.
class ModelConfigParserFactory {
 public:
  static Status CreateModelConfigParser(ModelArchitecture model_arch,
                                        std::shared_ptr<BaseModelConfigParser>& model_config_parser);
};

}  // namespace ksana_llm
