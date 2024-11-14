/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/utils/search_path.h"
#include <filesystem>
#include "ksana_llm/utils/gguf_file_tensor_loader.h"

namespace ksana_llm {

std::vector<std::string> SearchLocalPath(const std::string& model_path, ModelFileFormat& model_file_format) {
  std::vector<std::string> bin_file_list;
  std::vector<std::string> safetensors_list;
  std::vector<std::string> gguf_list;
  std::vector<std::string> black_list = {"training_args.bin", "optimizer.bin"};
  for (const auto& entry : std::filesystem::directory_iterator(model_path)) {
    if (entry.is_regular_file()) {
      std::string file_name = entry.path().filename().string();
      std::string extension = entry.path().extension().string();
      if (file_name.length() >= 6 && file_name.compare(0, 6, ".etag.") == 0) {
        // skip etag file
        continue;
      } else if (extension == ".bin") {
        bool is_black_file = false;
        for (std::string& black_file_name : black_list) {
          if (entry.path().filename().string() == black_file_name) {
            is_black_file = true;
            break;
          }
        }
        if (!is_black_file) {
          bin_file_list.emplace_back(entry.path().string());
        }
      } else if (extension == ".safetensors") {
        safetensors_list.emplace_back(entry.path().string());
      } else if (extension == ".gguf") {
        gguf_list = GGUFFileTensorLoader::FindModelFiles(model_path);
      }
    }
  }
  if (safetensors_list.size() > 0) {
    model_file_format = SAFETENSORS;
    return safetensors_list;
  } else if (gguf_list.size() > 0) {
    model_file_format = GGUF;
    return gguf_list;
  } else {
    model_file_format = BIN;
    return bin_file_list;
  }
}

}  // namespace ksana_llm