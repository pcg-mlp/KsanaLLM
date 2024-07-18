/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <fstream>
#include <regex>

#include "ksana_llm/utils/get_custom_weight_name.h"
#include "ksana_llm/utils/optional_file.h"

#include "nlohmann/json.hpp"

namespace ksana_llm {

Status GetCustomNameList(std::vector<std::string>& weight_name_list, std::vector<std::string>& custom_name_list,
                         std::string& model_path, std::string& model_type) {
  // In the default case, the tensor name is consistent with the weight name.
  custom_name_list.assign(weight_name_list.begin(), weight_name_list.end());

  // Search for the optional_weight_map.json file
  auto optional_file = Singleton<OptionalFile>::GetInstance();
  std::string& weight_path =
      optional_file->GetOptionalFile(model_path, "weight_map", model_type + "_weight_map.json");
  if (weight_path == "") {
    return Status();
  }

  nlohmann::json weight_map_json;
  std::ifstream file(weight_path);
  if (!file.is_open()) {
    KLLM_LOG_ERROR << fmt::format("Load weight map json: {} error.", weight_path) << std::endl;
    return Status(RetCode::RET_INVALID_ARGUMENT, fmt::format("Load weight map json: {} error.", weight_path));
  } else {
    file >> weight_map_json;
    file.close();
  }
  for (size_t idx = 0; idx < weight_name_list.size(); ++idx) {
    std::string weight_name = weight_name_list[idx];
    for (auto it = weight_map_json.begin(); it != weight_map_json.end(); ++it) {
      std::regex pattern(it.key());
      std::string format = it.value();
      if (std::regex_search(weight_name, pattern)) {
        custom_name_list[idx] = std::regex_replace(weight_name, pattern, format);
        break;
      }
    }
  }
  return Status();
}

}  // namespace ksana_llm
