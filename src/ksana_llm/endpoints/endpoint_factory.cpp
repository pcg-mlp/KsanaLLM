/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/endpoints/endpoint_factory.h"

#include <dlfcn.h>
#include <cctype>

#include "ksana_llm/utils/logger.h"

namespace ksana_llm {

std::shared_ptr<LocalEndpoint> EndpointFactory::CreateLocalEndpoint(
    const EndpointConfig &endpoint_config, Channel<std::pair<Status, std::shared_ptr<Request>>> &request_queue) {
  return std::make_shared<LocalEndpoint>(endpoint_config, request_queue);
}

std::shared_ptr<RpcEndpoint> EndpointFactory::CreateRpcEndpoint(const EndpointConfig &endpoint_config,
                                                                const std::shared_ptr<LocalEndpoint> &local_endpoint) {
  void *handle = nullptr;
  // Load the endpoint shared library: trpc/grpc -> libtrpc_endpoint.so/libgrpc_endpoint.so.
  const std::string endpoint_lib = fmt::format("lib{}_endpoint.so", endpoint_config.rpc_plugin_name);
  handle = dlopen(endpoint_lib.c_str(), RTLD_LAZY);
  if (handle) {
    using create_rpc_endpoint_t =
        std::shared_ptr<RpcEndpoint>(const EndpointConfig &, const std::shared_ptr<LocalEndpoint> &);
    // Get the create rpc function name: trpc/grpc -> CreateTrpcEndpoint/CreateGrpcEndpoint.
    std::string create_api = fmt::format("Create{}Endpoint", endpoint_config.rpc_plugin_name);
    create_api[6] = std::toupper(create_api[6]);
    // Extract the create rpc function.
    auto create_rpc_endpoint = reinterpret_cast<create_rpc_endpoint_t *>(dlsym(handle, create_api.c_str()));
    if (create_rpc_endpoint) {
      return create_rpc_endpoint(endpoint_config, local_endpoint);
    }
  }

  KLLM_THROW(fmt::format("Load rpc endpoint failed: {}", dlerror()));
}

}  // namespace ksana_llm
