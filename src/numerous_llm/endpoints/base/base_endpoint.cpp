/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/endpoints/base/base_endpoint.h"

namespace numerous_llm {

BaseEndpoint::BaseEndpoint(const EndpointConfig &endpoint_config,
                           std::function<Status(int64_t, std::vector<int> &)> fetch_func,
                           Channel<std::pair<Status, Request>> &request_queue)
    : request_queue_(request_queue), fetch_func_(fetch_func), endpoint_config_(endpoint_config) {}

RpcEndpoint::RpcEndpoint(const EndpointConfig &endpoint_config,
                         std::function<Status(int64_t, std::vector<int> &)> fetch_func,
                         Channel<std::pair<Status, Request>> &request_queue)
    : BaseEndpoint(endpoint_config, fetch_func, request_queue) {}

}  // namespace numerous_llm
