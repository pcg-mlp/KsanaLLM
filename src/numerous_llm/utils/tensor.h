/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace numerous_llm {

class Tensor {};

class TensorMap {
public:
  TensorMap() = default;
  TensorMap(const std::unordered_map<std::string, Tensor> &tensor_map);
  TensorMap(const std::vector<Tensor> &tensor_map);
  TensorMap(std::initializer_list<std::pair<std::string, Tensor>> tensor_map);
  ~TensorMap();

  void insert(const std::string &key, const Tensor &value);

private:
  std::unordered_map<std::string, Tensor> tensor_map_;
};

} // namespace numerous_llm
