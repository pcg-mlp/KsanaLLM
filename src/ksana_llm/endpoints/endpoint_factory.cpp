/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/endpoints/endpoint_factory.h"

#include <chrono>
#include <memory>
#include <string>
#include <thread>

#include "nlohmann/json.hpp"

#include "ksana_llm/endpoints/http/http_endpoint.h"
#include "ksana_llm/endpoints/local/local_endpoint.h"
#include "ksana_llm/endpoints/trpc/trpc_endpoint.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/waiter.h"

namespace ksana_llm {

std::shared_ptr<RpcEndpoint> EndpointFactory::CreateRpcEndpoint(
    const EndpointConfig &endpoint_config, Channel<std::pair<Status, std::shared_ptr<Request>>> &request_queue) {
  switch (endpoint_config.type) {
    case EndpointType::ENDPOINT_HTTP:
      return std::make_shared<HttpEndpoint>(endpoint_config, request_queue);
    case EndpointType::ENDPOINT_TRPC:
      return std::make_shared<TrpcEndpoint>(endpoint_config, request_queue);
    default:
      KLLM_LOG_ERROR << "Rpc endpoint type " << endpoint_config.type << " is not supported.";
      break;
  }
  return nullptr;
}

std::shared_ptr<LocalEndpoint> EndpointFactory::CreateLocalEndpoint(
    const EndpointConfig &endpoint_config, Channel<std::pair<Status, std::shared_ptr<Request>>> &request_queue) {
  switch (endpoint_config.type) {
    case EndpointType::ENDPOINT_LOCAL:
      return std::make_shared<LocalEndpoint>(endpoint_config, request_queue);
    default:
      KLLM_LOG_ERROR << "Local endpoint type " << endpoint_config.type << " is not supported.";
      break;
  }
  return nullptr;
}

}  // namespace ksana_llm
