/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include <string>
#include <vector>
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

Status GetCustomNameList(std::vector<std::string>& weight_name_list, std::vector<std::string>& custom_name_list,
                         std::string& model_path, std::string& model_type);

}  // namespace ksana_llm
