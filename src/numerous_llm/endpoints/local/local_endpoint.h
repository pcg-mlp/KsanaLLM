/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "numerous_llm/endpoints/base/base_endpoint.h"

#include "numerous_llm/endpoints/streaming/streaming_iterator.h"
#include "numerous_llm/utils/channel.h"
#include "numerous_llm/utils/request.h"
#include "numerous_llm/utils/status.h"

namespace numerous_llm {

class LocalEndpoint : public BaseEndpoint {
 public:
  LocalEndpoint(const EndpointConfig &endpoint_config,
                Channel<std::pair<Status, std::shared_ptr<Request>>> &request_queue);

  virtual ~LocalEndpoint() override {}

  // Handle a request.
  virtual Status Handle(const std::string &model_name, const std::vector<int> &input_tokens,
                        const SamplingConfig &sampling_config, std::vector<int> &output_tokens);

  // handle a streaming request.
  virtual Status HandleStreaming(const std::string &model_name, const std::vector<int> &input_tokens,
                                 const SamplingConfig &sampling_config,
                                 std::shared_ptr<StreamingIterator> &streaming_iterator);
};

}  // namespace numerous_llm
