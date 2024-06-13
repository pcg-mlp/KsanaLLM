#include "ksana_llm/utils/search_path.h"
#include <filesystem>

namespace ksana_llm {

std::vector<std::string> SearchLocalPath(const std::string& model_path, bool& is_safetensors) {
  std::vector<std::string> bin_file_list;
  std::vector<std::string> safetensors_list;
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
      }
    }
  }
  if (safetensors_list.size() > 0) {
    is_safetensors = true;
    return safetensors_list;
  }
  return bin_file_list;
}

}  // namespace ksana_llm