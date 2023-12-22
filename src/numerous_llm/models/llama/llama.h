/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "numerous_llm/models/base/base_model.h"
#include "numerous_llm/models/llama/llama_weight.h"
#include "numerous_llm/utils/environment.h"
#include "numerous_llm/utils/status.h"
#include "numerous_llm/utils/utils.h"

namespace numerous_llm {

template <typename T>
class Llama : public BaseModel {
 public:
  Llama(const ModelConfig& model_config, int rank) : model_config_(model_config), rank_(rank) {}
  ~Llama() {}

  float* GetLogitsPtr();

  // The prefill stage.
  Status ContextDecode(std::shared_ptr<numerous_llm::BaseWeight>& base_weight,
                       std::vector<ForwardRequest>& forward_reqs);

  // The decode stage.
  Status Decode(std::shared_ptr<numerous_llm::BaseWeight>& base_weight, std::vector<ForwardRequest>& forward_reqs);

 private:
  ModelConfig model_config_;

  int rank_;
};

}  // namespace numerous_llm
