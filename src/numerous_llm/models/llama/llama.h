/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "numerous_llm/models/base/base_model.h"
#include "numerous_llm/models/llama/llama_weight.h"
#include "numerous_llm/utils/status.h"
#include "numerous_llm/utils/utils.h"

namespace numerous_llm {

class Llama : public BaseModel {
 public:
  Llama() {}
  ~Llama() {}

  // The prefill stage.
  Status ContextDecode(std::shared_ptr<numerous_llm::BaseWeight>& base_weight,
                       const std::vector<TensorMap*>& input_tensor_maps, std::vector<TensorMap*>& output_tensor_maps);

  // The decode stage.
  Status Decode(std::shared_ptr<numerous_llm::BaseWeight>& base_weight,
                const std::vector<TensorMap*>& input_tensor_maps, std::vector<TensorMap*>& output_tensor_maps);
};

}  // namespace numerous_llm
