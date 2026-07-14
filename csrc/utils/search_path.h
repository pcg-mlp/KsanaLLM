/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include <string>
#include <vector>

namespace ksana_llm {

enum ModelFileFormat { BIN, SAFETENSORS, GGUF };
std::vector<std::string> SearchLocalPath(const std::string& model_path, ModelFileFormat& is_safetensors);

}  // namespace ksana_llm