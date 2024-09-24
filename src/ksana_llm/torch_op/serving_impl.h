/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <torch/script.h>

#include "ksana_llm/endpoints/local/local_endpoint.h"
#include "ksana_llm/endpoints/streaming/streaming_iterator.h"
#include "ksana_llm/service/inference_engine.h"
#include "ksana_llm/utils/channel.h"

namespace ksana_llm {

// The serving implementation.
class ServingImpl {
 public:
  ServingImpl();
  ~ServingImpl() {}

  // Start the inference server.
  Status Start();

  // Stop the inference server.
  Status Stop();

  // Handle serving request.
  Status Handle(const std::shared_ptr<KsanaPythonInput> &ksana_python_input,
                ksana_llm::KsanaPythonOutput &ksana_python_output);

  // Handle serving request, in streaming mode.
  Status HandleStreaming(const std::shared_ptr<KsanaPythonInput> &ksana_python_input,
                         std::shared_ptr<StreamingIterator> &streaming_iterator);

  // Handle serving batch request.
  Status HandleBatch(const std::vector<std::shared_ptr<KsanaPythonInput>> &ksana_python_inputs,
                     std::vector<KsanaPythonOutput> &ksana_python_outputs);

 private:
  // The inference engine.
  std::shared_ptr<InferenceEngine> inference_engine_ = nullptr;

  // The rpc endpoint of this service.
  std::shared_ptr<LocalEndpoint> endpoint_ = nullptr;

  // channel for endpoint and inference server
  Channel<std::pair<Status, std::shared_ptr<Request>>> request_queue_;
};

}  // namespace ksana_llm
