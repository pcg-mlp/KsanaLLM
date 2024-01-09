/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "numerous_llm/endpoints/base/base_endpoint.h"

namespace numerous_llm {

class TrpcEndpoint : public RpcEndpoint {
 public:
  TrpcEndpoint(const EndpointConfig &endpoint_config,
               Channel<std::pair<Status, std::shared_ptr<Request>>> &request_queue);

  virtual ~TrpcEndpoint() override {}

  // Listen at specific socket.
  virtual Status Start() override;

  // Close the listening socket.
  virtual Status Stop() override;
};

}  // namespace numerous_llm
