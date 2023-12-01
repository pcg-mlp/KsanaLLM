/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/utils/tensor.h"

namespace numerous_llm {

TensorMap::TensorMap(
    const std::unordered_map<std::string, Tensor> &tensor_map) {
  for (auto &kv : tensor_map) {
    insert(kv.first, kv.second);
  }
}

TensorMap::TensorMap(const std::vector<Tensor> &tensor_map) {
  for (size_t i = 0; i < tensor_map.size(); i++) {
    insert(std::to_string(i), tensor_map[i]);
  }
}

TensorMap::TensorMap(
    std::initializer_list<std::pair<std::string, Tensor>> tensor_map) {
  for (auto &pair : tensor_map) {
    insert(pair.first, pair.second);
  }
}

TensorMap::~TensorMap() { tensor_map_.clear(); }

void TensorMap::insert(const std::string &key, const Tensor &value) {
  tensor_map_.insert({key, value});
}

} // namespace numerous_llm
