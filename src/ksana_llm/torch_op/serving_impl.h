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
    Status Handle(const std::string &model_name, const std::vector<int> &input_tokens,
                  const SamplingConfig &sampling_config, std::vector<std::vector<int>> &output_tokens,
                  std::vector<std::vector<std::vector<std::pair<int, float>>>> &logprobs);

    // Handle serving request, in streaming mode.
    Status HandleStreaming(const std::string &model_name, const std::vector<int> &input_tokens,
                           const SamplingConfig &sampling_config,
                           std::shared_ptr<StreamingIterator> &streaming_iterator);

  private:
    // The inference engine.
    std::shared_ptr<InferenceEngine> inference_engine_ = nullptr;

    // The rpc endpoint of this service.
    std::shared_ptr<LocalEndpoint> endpoint_ = nullptr;

    // channel for endpoint and inference server
    Channel<std::pair<Status, std::shared_ptr<Request>>> request_queue_;
};

}  // namespace ksana_llm
