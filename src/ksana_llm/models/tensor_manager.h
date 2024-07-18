/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/tensor.h"
#include "ksana_llm/utils/utils.h"

namespace ksana_llm {

class TensorManager {
 public:
  TensorManager(int rank, std::unordered_map<std::string, Tensor>& weights_map)
      : rank_(rank), weights_map_(weights_map) {}
  ~TensorManager() {}

  Status AddWeightTensor(std::string weight_name, std::vector<size_t> shapes, DataType dtype) {
    if (weights_map_.count(weight_name)) {
      KLLM_LOG_WARNING << fmt::format("The weight named {} has already been created. Skip creating the weight tensor.",
                                      weight_name);
      return Status();
    }
    size_t length = GetTypeSize(dtype);
    for (auto& dim : shapes) {
      length *= dim;
    }
    int block_id;
    GetBlockManager()->SetDeviceId(rank_);
    GetBlockManager()->AllocateContiguous(length, block_id);

    weights_map_.emplace(weight_name, Tensor(MemoryDevice::MEMORY_DEVICE, dtype, shapes, block_id));
    return Status();
  }

  Status CreateTensorWithSameShape(const std::string& origin_tensor_name, const std::string& copy_tensor_name) {
    if (!weights_map_.count(origin_tensor_name)) {
      KLLM_LOG_ERROR << fmt::format("Create tensor {} faild: tensor {} not in weights map", copy_tensor_name,
                                    origin_tensor_name);
      exit(-1);
    }
    Tensor& origin_tensor = weights_map_[origin_tensor_name];
    AddWeightTensor(copy_tensor_name, origin_tensor.shape, origin_tensor.dtype);
    return Status();
  }

 private:
  int rank_ = 0;
  std::unordered_map<std::string, Tensor>& weights_map_;
};

}  // namespace ksana_llm