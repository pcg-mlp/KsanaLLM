/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/endpoints/trpc/trpc_endpoint.h"

namespace numerous_llm {

TrpcEndpoint::TrpcEndpoint(const EndpointConfig &endpoint_config,
                           std::function<Status(int64_t, std::vector<std::vector<int>> &)> fetch_func,
                           Channel<std::pair<Status, Request>> &request_queue)
    : RpcEndpoint(endpoint_config, fetch_func, request_queue) {}

Status TrpcEndpoint::Start() { return Status(); }

// Close the listening socket.
Status TrpcEndpoint::Stop() { return Status(); }

}  // namespace numerous_llm
