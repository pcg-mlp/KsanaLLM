/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/endpoints/endpoint_factory.h"

#include <chrono>
#include <memory>
#include <string>
#include <thread>

#include "nlohmann/json.hpp"

#include "numerous_llm/endpoints/http/http_endpoint.h"
#include "numerous_llm/endpoints/local/local_endpoint.h"
#include "numerous_llm/endpoints/trpc/trpc_endpoint.h"
#include "numerous_llm/utils/logger.h"
#include "numerous_llm/utils/waiter.h"

namespace numerous_llm {

std::shared_ptr<RpcEndpoint> EndpointFactory::CreateRpcEndpoint(
    const EndpointConfig &endpoint_config, std::function<Status(int64_t, std::vector<int> &)> fetch_func,
    Channel<std::pair<Status, Request>> &request_queue) {
  switch (endpoint_config.type) {
    case EndpointType::ENDPOINT_HTTP:
      return std::make_shared<HttpEndpoint>(endpoint_config, fetch_func, request_queue);
    case EndpointType::ENDPOINT_TRPC:
      return std::make_shared<TrpcEndpoint>(endpoint_config, fetch_func, request_queue);
    default:
      NLLM_LOG_ERROR << "Rpc endpoint type " << endpoint_config.type << " is not supported.";
      break;
  }
  return nullptr;
}

std::shared_ptr<LocalEndpoint> EndpointFactory::CreateLocalEndpoint(
    const EndpointConfig &endpoint_config, std::function<Status(int64_t, std::vector<int> &)> fetch_func,
    Channel<std::pair<Status, Request>> &request_queue) {
  switch (endpoint_config.type) {
    case EndpointType::ENDPOINT_LOCAL:
      return std::make_shared<LocalEndpoint>(endpoint_config, fetch_func, request_queue);
    default:
      NLLM_LOG_ERROR << "Local endpoint type " << endpoint_config.type << " is not supported.";
      break;
  }
  return nullptr;
}

}  // namespace numerous_llm
