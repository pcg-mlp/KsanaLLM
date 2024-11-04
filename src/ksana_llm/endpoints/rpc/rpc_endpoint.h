/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "ksana_llm/utils/environment.h"

namespace ksana_llm {

class LocalEndpoint;

// Abstract base class for rpc endpoint.
class RpcEndpoint {
 public:
  RpcEndpoint(const EndpointConfig& endpoint_config, const std::shared_ptr<LocalEndpoint>& local_endpoint)
      : endpoint_config_(endpoint_config), local_endpoint_(local_endpoint) {}

  virtual ~RpcEndpoint() {}

  // Start the rpc endpoint.
  virtual Status Start() = 0;

  // Stop the rpc endpoint.
  virtual Status Stop() = 0;

 protected:
  // The endpoint config.
  EndpointConfig endpoint_config_;

  // The local endpoint that contains the main logic for handling requests.
  std::shared_ptr<LocalEndpoint> local_endpoint_;
};

}  // namespace ksana_llm
