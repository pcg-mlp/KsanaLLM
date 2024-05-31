/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <atomic>

#include "ksana_llm/batch_manager/batch_manager.h"
#include "ksana_llm/endpoints/base/base_endpoint.h"
#include "ksana_llm/endpoints/local/local_endpoint.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/request.h"
#include "ksana_llm/utils/status.h"
#include "src/ksana_llm/utils/channel.h"

namespace ksana_llm {

class EndpointFactory {
 public:
  // Create a rpc endpoint instance via the input config.
  static std::shared_ptr<RpcEndpoint> CreateRpcEndpoint(
      const EndpointConfig &endpoint_config, Channel<std::pair<Status, std::shared_ptr<Request>>> &request_queue);

  // Create a local endpoint instance via the input config.
  static std::shared_ptr<LocalEndpoint> CreateLocalEndpoint(
      const EndpointConfig &endpoint_config, Channel<std::pair<Status, std::shared_ptr<Request>>> &request_queue);
};

}  // namespace ksana_llm
