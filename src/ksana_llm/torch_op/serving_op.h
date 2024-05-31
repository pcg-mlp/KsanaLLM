/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <torch/script.h>

#include "ksana_llm/endpoints/local/local_endpoint.h"
#include "ksana_llm/endpoints/streaming/streaming_iterator.h"
#include "ksana_llm/service/inference_engine.h"
#include "ksana_llm/torch_op/serving_impl.h"
#include "ksana_llm/utils/channel.h"

namespace ksana_llm {

// The torch OP for inference.
class ServingOp : public torch::jit::CustomClassHolder {
 public:
  ServingOp();
  ~ServingOp();

  // Initialize the service implement.
  void InitServing(const std::string &mode_dir);

  // Generate a response.
  Status Generate(const ksana_llm::KsanaPythonInput &ksana_python_input,
                  ksana_llm::KsanaPythonOutput &ksana_python_output);

  // Generate a response, in streaming mode.
  Status GenerateStreaming(const ksana_llm::KsanaPythonInput &ksana_python_input,
                           std::shared_ptr<StreamingIterator> &streaming_iterator);

 public:
  std::string plugin_path_;

 private:
  // The inference implement.
  std::shared_ptr<ServingImpl> serving_impl_ = nullptr;
};

}  // namespace ksana_llm
