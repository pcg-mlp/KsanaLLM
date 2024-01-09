/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <torch/script.h>

#include "numerous_llm/endpoints/local/local_endpoint.h"
#include "numerous_llm/endpoints/streaming/streaming_iterator.h"
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
  Status Generate(const std::string &model_name, const std::vector<int> &input_tokens,
                  const SamplingConfig &sampling_config, std::vector<int> &output_tokens);

  // Generate a response, in streaming mode.
  Status GenerateStreaming(const std::string &model_name, const std::vector<int> &input_tokens,
                           const SamplingConfig &sampling_config,
                           std::shared_ptr<StreamingIterator> &streaming_iterator);

 private:
  // The inference implement.
  std::shared_ptr<ServingImpl> serving_impl_ = nullptr;
};

}  // namespace numerous_llm
