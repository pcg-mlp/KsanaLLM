/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include <string>
#include <vector>
#include "utils/search_path.h"
#include "utils/status.h"

namespace ksana_llm {

Status GetCustomNameList(const std::string& model_path, const std::string& model_type,
                         const std::vector<std::string>& weight_name_list, std::vector<std::string>& custom_name_list,
                         ModelFileFormat model_file_format = ModelFileFormat::SAFETENSORS);

std::string GetWeightMapPath(const std::string& model_path, const std::string& model_type,
                             ModelFileFormat model_file_format);

}  // namespace ksana_llm
