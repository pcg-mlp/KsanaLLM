/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/endpoints/endpoint.h"

#include <chrono>
#include <thread>

#include "numerous_llm/utils/logger.h"

namespace numerous_llm {

Endpoint::Endpoint(const EndpointConfig &endpoint_config) {}

Status Endpoint::Listen() {
  NLLM_LOG_INFO << "Listen on port 8080." << std::endl;
  return Status();
}

Status Endpoint::Close() {
  NLLM_LOG_INFO << "Close endpoint." << std::endl;
  terminated_ = true;
  return Status();
}

Status Endpoint::Accept(Request &req) {
  static int count = 0;
  while (count > 0 && !terminated_) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    continue;
  }

  if (terminated_) {
    return Status(RET_TERMINATED);
  }

  ++count;
  NLLM_LOG_INFO << "Accept a req." << std::endl;

  SamplingConfig sampling_config;
  std::vector<SamplingConfig> sampling_configs;
  sampling_configs.push_back(sampling_config);

  TensorMap tensor_map;
  std::vector<TensorMap> tensor_maps;
  tensor_maps.push_back(tensor_map);

  req.req_id = 1;
  req.tensor_maps = tensor_maps;
  req.sampling_configs = sampling_configs;

  return Status();
}

Status Endpoint::Send(const Response &rsp) {
  NLLM_LOG_INFO << "Send a rsp." << std::endl;
  return Status();
}

} // namespace numerous_llm
