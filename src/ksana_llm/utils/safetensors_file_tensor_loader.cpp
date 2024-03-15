#include "safetensors_file_tensor_loader.h"
#include "logger.h"
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace ksana_llm {
// Constructor of SafeTensorsLoader that takes a file name as input
SafeTensorsLoader::SafeTensorsLoader(const std::string& file_name) : BaseFileTensorLoader(file_name) {
  // Check if the file name has a ".bin" extension
  if (file_name_.length() > 12) {
    if (file_name_.substr(file_name_.length() - 12) == ".safetensors") {
      // Load the PyTorch binary file
      LoadSafeTensors();
    }
  }
}

SafeTensorsLoader::~SafeTensorsLoader() {
  delete[] weights_buffer_;
}

// Function to load the SafeTensors binary file
void SafeTensorsLoader::LoadSafeTensors() {
  std::ifstream safetensors_file(file_name_, std::ios::binary | std::ios::ate);
  if (!safetensors_file.is_open()) {
    NLLM_LOG_ERROR << fmt::format("Can't open safetensors file: {}", file_name_);
  }
  size_t file_size = safetensors_file.tellg();
  if (file_size == -1) {
    NLLM_LOG_ERROR << fmt::format("Invalid safetensors file size: -1, filename: {}", file_name_);
  }
  safetensors_file.seekg(0, std::ios::beg);

  // get the tensor list(string)
  size_t header_size;
  safetensors_file.read(reinterpret_cast<char*>(&header_size), sizeof(size_t));
  std::string tensor_dict_str;
  tensor_dict_str.resize(header_size);
  safetensors_file.read(&tensor_dict_str[0], header_size);

  size_t data_size = file_size - header_size - sizeof(size_t);
  weights_buffer_ = new char[data_size];
  safetensors_file.read(weights_buffer_, data_size);
  safetensors_file.close();

  // Parsing JSON to retrieve tensor information.
  json tensor_dict = json::parse(tensor_dict_str);
  for (const auto& tensor_iter : tensor_dict.items()) {
    // tensor name
    const std::string& tensor_name = tensor_iter.key();
    tensor_name_list_.emplace_back(tensor_name);
    json tensor_data = tensor_iter.value();
    if (!tensor_data.contains("data_offsets")) {
      continue;
    }
    // tensor ptr
    size_t tensor_begin_index = tensor_data["data_offsets"][0];
    size_t tensor_end_index = tensor_data["data_offsets"][1];
    tensor_offset_map_[tensor_name] = tensor_begin_index;
    tensor_size_map_[tensor_name] = tensor_end_index - tensor_begin_index;
  }
}

// Function to get a tensor by its name
void* SafeTensorsLoader::GetTensor(const std::string& tensor_name) {
  // Check if the tensor name exists in the index map
  if (!tensor_ptr_map_.count(tensor_name)) {
    if (!tensor_offset_map_.count(tensor_name) || !tensor_size_map_.count(tensor_name)) {
      return nullptr;
    }
    std::ifstream safetensors_file(file_name_, std::ios::binary | std::ios::ate);
    if (!safetensors_file.is_open()) {
      NLLM_LOG_ERROR << fmt::format("Can't open safetensors file: {}", file_name_);
      return nullptr;
    }
    safetensors_file.seekg(tensor_offset_map_["base_index"], std::ios::cur);
    safetensors_file.read(weights_buffer_ + tensor_offset_map_[tensor_name], tensor_size_map_[tensor_name]);
    safetensors_file.close();
    tensor_ptr_map_[tensor_name] = weights_buffer_ + tensor_offset_map_[tensor_name];
  }
  return tensor_ptr_map_[tensor_name];
}

}  // namespace ksana_llm
