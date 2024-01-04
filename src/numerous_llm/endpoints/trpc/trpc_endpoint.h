/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "numerous_llm/endpoints/base/base_endpoint.h"

namespace numerous_llm {

class TrpcEndpoint : public RpcEndpoint {
 public:
  TrpcEndpoint(const EndpointConfig &endpoint_config,
               std::function<Status(int64_t, std::vector<std::vector<int>> &)> fetch_func,
               Channel<std::pair<Status, Request>> &request_queue);

  virtual ~TrpcEndpoint() override {}

  // Listen at specific socket.
  virtual Status Start() override;

  // Close the listening socket.
  virtual Status Stop() override;
};

}  // namespace numerous_llm
