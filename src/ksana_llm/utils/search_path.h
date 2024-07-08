/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include <string>
#include <vector>

namespace ksana_llm {

std::vector<std::string> SearchLocalPath(const std::string& model_path, bool& is_safetensors);

}  // namespace ksana_llm