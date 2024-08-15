/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <torch/script.h>

#include "ksana_llm/endpoints/local/local_endpoint.h"
#include "ksana_llm/endpoints/streaming/streaming_iterator.h"
#include "ksana_llm/service/inference_engine.h"
#include "ksana_llm/torch_op/serving_impl.h"
#include "ksana_llm/utils/channel.h"
#include "ksana_llm/utils/request_packer.h"

namespace ksana_llm {

// The torch OP for inference.
class ServingOp : public torch::jit::CustomClassHolder {
 public:
  ServingOp();
  ~ServingOp();

  // Initialize the service implement.
  void InitServing(const std::string &mode_dir);

  // Generate a response.
  Status Generate(const std::shared_ptr<KsanaPythonInput> &ksana_python_input,
                  ksana_llm::KsanaPythonOutput &ksana_python_output);

  // Generate a response, in streaming mode.
  Status GenerateStreaming(const std::shared_ptr<KsanaPythonInput> &ksana_python_input,
                           std::shared_ptr<StreamingIterator> &streaming_iterator);

  /** 
   * Generate responses for requests toward the forward interface.
   * This interface performs a single-step inference, i.e., the context phase.
   * It allows a batch of requests to be submitted at once, and each request
   * can specify multiple targets to obtain model outputs such as transformer,
   * final layer norm, logits, etc.
   * Refer to serving_forward_client for more details.
   */
  Status Forward(const std::string &request_bytes, std::string &response_bytes);

 public:
  std::string plugin_path_;

 private:
  // The inference implement.
  std::shared_ptr<ServingImpl> serving_impl_ = nullptr;

  // Used in the forward interface to unpack requests and pack responses.
  RequestPacker request_packer_;
};

}  // namespace ksana_llm
