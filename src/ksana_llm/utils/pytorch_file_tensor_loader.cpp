// Copyright 2024 Tencent Inc.  All rights reserved.
#include "pytorch_file_tensor_loader.h"
#include "logger.h"

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

  std::vector<int64_t> storage_indexs(records.size());

  auto storage_context = std::make_shared<torch::jit::DeserializationStorageContext>();

  // Add fictional storage context
  for (int i = 0; i < records.size(); i++) {
    storage_indexs[i] = i;
    auto storage = at::Storage(c10::Storage::use_byte_size_t(), sizeof(int64_t),
                               at::DataPtr((void*)(storage_indexs.data() + i), c10::DeviceType::CPU), nullptr, false);
    storage_context->addStorage(std::to_string(i), storage);
  }

  auto pytorch_value =
      torch::jit::readArchiveAndTensors("data", "", "", c10::nullopt, c10::nullopt, c10::nullopt, *pytorch_reader_,
                                        torch::jit::Unpickler::defaultTypeParser, storage_context);

  // If the value is a generic dictionary, process the tensors in the dictionary
  if (pytorch_value.isGenericDict()) {
    auto value_dict = pytorch_value.toGenericDict();
    for (auto& it : value_dict) {
      std::string tensor_name = it.key().toStringRef();
      tensor_name_list_.push_back(tensor_name);
      if (it.value().isTensor()) {
        pytorch_tensor_index_map_[tensor_name] = *static_cast<int64_t*>(it.value().toTensor().data_ptr());
      }
    }
  }
}

void* PytorchFileTensorLoader::GetTensor(const std::string& tensor_name) {
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

  NLLM_LOG_DEBUG << tensor_name << " " << data_ptr << " data_size= " << data_size;
  return data_ptr;
}

}  // namespace ksana_llm
