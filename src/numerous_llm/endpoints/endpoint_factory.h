/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <atomic>

#include "numerous_llm/batch_manager/batch_manager.h"
#include "numerous_llm/endpoints/base/base_endpoint.h"
#include "numerous_llm/endpoints/local/local_endpoint.h"
#include "numerous_llm/utils/environment.h"
#include "numerous_llm/utils/request.h"
#include "numerous_llm/utils/status.h"
#include "src/numerous_llm/utils/channel.h"

namespace numerous_llm {

class EndpointFactory {
 public:
  // Create a rpc endpoint instance via the input config.
  static std::shared_ptr<RpcEndpoint> CreateRpcEndpoint(
      const EndpointConfig &endpoint_config, std::function<Status(int64_t, std::vector<std::vector<int>> &)> fetch_func,
      Channel<std::pair<Status, Request>> &request_queue);

  // Create a local endpoint instance via the input config.
  static std::shared_ptr<LocalEndpoint> CreateLocalEndpoint(
      const EndpointConfig &endpoint_config, std::function<Status(int64_t, std::vector<std::vector<int>> &)> fetch_func,
      Channel<std::pair<Status, Request>> &request_queue);
};

}  // namespace numerous_llm
