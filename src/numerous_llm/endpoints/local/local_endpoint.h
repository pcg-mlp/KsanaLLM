/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "numerous_llm/endpoints/base/base_endpoint.h"

#include "numerous_llm/utils/channel.h"
#include "numerous_llm/utils/request.h"
#include "numerous_llm/utils/status.h"

namespace numerous_llm {

class LocalEndpoint : public BaseEndpoint {
 public:
  LocalEndpoint(const EndpointConfig &endpoint_config, std::function<Status(int64_t, std::vector<int> &)> fetch_func,
                Channel<std::pair<Status, Request>> &request_queue);

  virtual ~LocalEndpoint() override {}

  // Handle a request.
  virtual Status Handle(const std::string &model_name, const std::vector<int> &input_tokens,
                        const SamplingConfig &sampling_config, std::vector<int> &output_tokens);
};

}  // namespace numerous_llm
