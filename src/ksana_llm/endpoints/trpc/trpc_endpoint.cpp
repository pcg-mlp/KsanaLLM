/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/endpoints/trpc/trpc_endpoint.h"

namespace ksana_llm {

TrpcEndpoint::TrpcEndpoint(const EndpointConfig &endpoint_config,
                           Channel<std::pair<Status, std::shared_ptr<Request>>> &request_queue)
    : RpcEndpoint(endpoint_config, request_queue) {}

Status TrpcEndpoint::Start() { return Status(); }

// Close the listening socket.
Status TrpcEndpoint::Stop() { return Status(); }

}  // namespace ksana_llm
