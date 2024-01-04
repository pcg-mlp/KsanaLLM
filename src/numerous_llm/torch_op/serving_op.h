/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <torch/script.h>

#include "numerous_llm/endpoints/local/local_endpoint.h"
#include "numerous_llm/service/inference_engine.h"
#include "numerous_llm/torch_op/serving_impl.h"
#include "numerous_llm/utils/channel.h"

namespace numerous_llm {

// The torch OP for inference.
class ServingOp : public torch::jit::CustomClassHolder {
 public:
  ServingOp();
  ~ServingOp();

  // Initialize the service implement.
  void InitServing(const std::string &mode_dir);

  // Generate a response.
  Status Generate(const std::string &model_name, const std::vector<std::vector<int>> &tokens,
                  const std::vector<SamplingConfig> &sampling_configs, std::vector<std::vector<int>> &output_tokens);

 private:
  // The inference implement.
  std::shared_ptr<ServingImpl> serving_impl_ = nullptr;
};

}  // namespace numerous_llm
