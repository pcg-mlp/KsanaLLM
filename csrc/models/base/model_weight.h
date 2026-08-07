/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <string>
#include <unordered_map>

#include "models/base/base_weight.h"
#include "utils/tensor.h"

namespace ksana_llm {

class ModelWeightLoader;

// class for model weights.
class ModelWeight : public BaseWeight {
 public:
  ModelWeight();
  virtual ~ModelWeight() override;

  friend class ModelWeightLoader;

  // Get a weight tensor by name.
  const Tensor& GetWeightTensor(const std::string& weight_name) const;

  // Get the weight name list.
  std::vector<std::string> GetWeightNames() const;

  // //////////////////////////////////////////////////////////////////////////////////////////
  // For compatible with common_model, improve it later
  // //////////////////////////////////////////////////////////////////////////////////////////
  virtual Tensor GetModelWeights(const std::string& weight_name) override { return GetWeightTensor(weight_name); }
  // //////////////////////////////////////////////////////////////////////////////////////////

 private:
  using BaseWeight::weights_map_;
  Tensor empty_tensor_;
};

}  // namespace ksana_llm
