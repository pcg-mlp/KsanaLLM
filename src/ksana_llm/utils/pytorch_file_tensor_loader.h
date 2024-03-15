// Copyright 2024 Tencent Inc.  All rights reserved.
#pragma once

#include "base_file_tensor_loader.h"

#include <caffe2/serialize/inline_container.h>
#include <torch/csrc/jit/serialization/import_read.h>
#include <torch/csrc/jit/serialization/storage_context.h>
#include <torch/script.h>

namespace ksana_llm {
// Define a class named PytorchFileTensorLoader that inherits from BaseFileTensorLoader
class PytorchFileTensorLoader : public BaseFileTensorLoader {
 public:
  // Constructor that takes a file name as input
  PytorchFileTensorLoader(const std::string& file_name);

  // Get the list of tensor names
  const std::vector<std::string>& GetTensorNameList() { return tensor_name_list_; }

  // Get a tensor by its name
  void* GetTensor(const std::string& tensor_name);

 private:
  // Load the PyTorch binary file
  void LoadPytorchBin();

 private:
  // Use unique_ptr to manage the PyTorchStreamReader object for reading PyTorch model files
  std::unique_ptr<caffe2::serialize::PyTorchStreamReader> pytorch_reader_;

  // Use unordered_map to store the tensor names and their corresponding indices for easy lookup
  std::unordered_map<std::string, int64_t> pytorch_tensor_index_map_;

  // Use vector to store the DataPtr of tensors for easy management and access
  std::vector<at::DataPtr> pytorch_tensor_list_;
};

}  // namespace ksana_llm
