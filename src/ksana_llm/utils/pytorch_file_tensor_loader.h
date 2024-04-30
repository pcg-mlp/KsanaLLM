// Copyright 2024 Tencent Inc.  All rights reserved.
#pragma once

#include "base_file_tensor_loader.h"

#include <Python.h>
#include <caffe2/serialize/inline_container.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/jit/serialization/import_read.h>
#include <torch/csrc/jit/serialization/storage_context.h>
#include <torch/script.h>

namespace py = pybind11;

namespace ksana_llm {
// Define a class named PytorchFileTensorLoader that inherits from BaseFileTensorLoader
class PytorchFileTensorLoader : public BaseFileTensorLoader {
  public:
    // Constructor that takes a file name as input
    PytorchFileTensorLoader(const std::string& file_name);

    // Get the list of tensor names
    const std::vector<std::string>& GetTensorNameList() { return tensor_name_list_; }

    // Get a tensor by its name
    std::tuple<void*, size_t> GetTensor(const std::string& tensor_name);

    DataType GetTensorDataType(const std::string& tensor_name);

    std::vector<std::size_t> GetTensorShape(const std::string& tensor_name);

  private:
    // Load the PyTorch binary file
    void LoadPytorchBin();

  private:
    py::object model_;

    // Use unordered_map to store the tensor names and their corresponding indices for easy lookup
    std::unordered_map<std::string, torch::Tensor> pytorch_tensor_map_;
};

}  // namespace ksana_llm
