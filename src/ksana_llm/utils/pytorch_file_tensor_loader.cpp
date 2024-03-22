// Copyright 2024 Tencent Inc.  All rights reserved.
#include "pytorch_file_tensor_loader.h"
#include "logger.h"
// #include "ksana_llm/utils/nvidia/cuda_utils.h"

namespace ksana_llm {
// Constructor of PytorchFileTensorLoader that takes a file name as input
PytorchFileTensorLoader::PytorchFileTensorLoader(const std::string& file_name) : BaseFileTensorLoader(file_name) {
  // Check if the file name has a ".bin" extension
  if (file_name_.length() > 4) {
    if (file_name_.substr(file_name_.length() - 4) == ".bin") {
      LoadPytorchBin();
    }
  }
}

// Function to load the PyTorch binary file
void PytorchFileTensorLoader::LoadPytorchBin() {
  // Create a PyTorchStreamReader object to read the model file
  pytorch_reader_ = std::make_unique<caffe2::serialize::PyTorchStreamReader>(file_name_);

  auto records = pytorch_reader_->getAllRecords();

  char* storage_indexs = nullptr;
  size_t max_tensor_size = 80 * 1024 * 1024;
  std::vector<char> storage_indexs_vector(max_tensor_size * records.size());
  storage_indexs = storage_indexs_vector.data();

  // When storage_context is nullptr, it indicates that the actual data of the torch tensor should be read directly.
  // Otherwise, it temporarily skips the reading process and waits until it is actually used before reading.
  std::shared_ptr<torch::jit::DeserializationStorageContext> storage_context = nullptr;
  if (fast_load_) {
    storage_context = std::make_shared<torch::jit::DeserializationStorageContext>();
    // Add fictional storage context
    for (int i = 0; i < records.size(); i++) {
      storage_indexs[i * max_tensor_size] = i;
      auto storage =
          at::Storage(c10::Storage::use_byte_size_t(), max_tensor_size,
                      at::DataPtr((void*)(storage_indexs + i * max_tensor_size), c10::DeviceType::CPU), nullptr, false);
      storage_context->addStorage(std::to_string(i), storage);
    }
  }

  auto pytorch_value =
      torch::jit::readArchiveAndTensors("data", "", "", c10::nullopt, c10::nullopt, c10::DeviceType::CPU,
                                        *pytorch_reader_, torch::jit::Unpickler::defaultTypeParser, storage_context);

  // If the value is a generic dictionary, process the tensors in the dictionary
  if (pytorch_value.isGenericDict()) {
    auto value_dict = pytorch_value.toGenericDict();
    for (auto& it : value_dict) {
      std::string tensor_name = it.key().toStringRef();
      tensor_name_list_.push_back(tensor_name);
      if (it.value().isTensor()) {
        if (fast_load_) {
          pytorch_tensor_index_map_[tensor_name] = *static_cast<int64_t*>(it.value().toTensor().data_ptr());
        } else {
          pytorch_tensor_map_[tensor_name] = it.value().toTensor();
        }
      }
    }
  }
}

DataType PytorchFileTensorLoader::GetDataType(const std::string& tensor_name) {
  DataType data_type = TYPE_INVALID;
  if (fast_load_) {
    data_type = TYPE_FP16;  // TODO
  } else {
    c10::ScalarType dtype = pytorch_tensor_map_[tensor_name].scalar_type();
    switch (dtype) {
      case c10::kBFloat16:
        data_type = TYPE_BF16;
        break;
      case torch::kFloat16:
        data_type = TYPE_FP16;
        break;
      case torch::kFloat32:
        data_type = TYPE_FP32;
        break;
      default:
        break;
    }
  }
  return data_type;
}

void* PytorchFileTensorLoader::GetTensor(const std::string& tensor_name) {
  if (fast_load_) {
    if (pytorch_tensor_index_map_.find(tensor_name) == pytorch_tensor_index_map_.end()) {
      return nullptr;
    }
    int64_t index = pytorch_tensor_index_map_[tensor_name];
    auto data_pair = pytorch_reader_->getRecord("data/" + std::to_string(index));
    // Get the data pointer and size of the tensor
    at::DataPtr at_data_ptr = std::move(std::get<0>(data_pair));
    void* data_ptr = std::get<0>(data_pair).get();
    pytorch_tensor_list_.push_back(std::move(at_data_ptr));
    int64_t data_size = std::get<1>(data_pair);
    return data_ptr;
  }
  if (!pytorch_tensor_map_.count(tensor_name)) {
    return nullptr;
  }
  return pytorch_tensor_map_[tensor_name].data_ptr();
}

}  // namespace ksana_llm
