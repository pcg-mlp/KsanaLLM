/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/endpoints/base/base_endpoint.h"

namespace ksana_llm {

BaseEndpoint::BaseEndpoint(const EndpointConfig &endpoint_config,
                           Channel<std::pair<Status, std::shared_ptr<Request>>> &request_queue)
    : request_queue_(request_queue), endpoint_config_(endpoint_config) {}

RpcEndpoint::RpcEndpoint(const EndpointConfig &endpoint_config,
                         Channel<std::pair<Status, std::shared_ptr<Request>>> &request_queue)
    : BaseEndpoint(endpoint_config, request_queue) {}

}  // namespace ksana_llm
