/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/endpoints/local/local_endpoint.h"
#include "numerous_llm/utils/logger.h"
#include "numerous_llm/utils/status.h"

namespace numerous_llm {

LocalEndpoint::LocalEndpoint(const EndpointConfig &endpoint_config,
                             std::function<Status(int64_t, std::vector<std::vector<int>> &)> fetch_func,
                             Channel<std::pair<Status, Request>> &request_queue)
    : BaseEndpoint(endpoint_config, fetch_func, request_queue) {}

Status LocalEndpoint::Handle(const std::string &model_name, const std::vector<std::vector<int>> &tokens,
                             const std::vector<SamplingConfig> &sampling_configs,
                             std::vector<std::vector<int>> &output_tokens) {
  Request req;
  req.model_name = model_name;
  req.tokens = tokens;
  req.sampling_configs = sampling_configs;

  req.waiter = std::make_shared<Waiter>(req.tokens.size());

  Status status = Status();
  std::shared_ptr<Waiter> waiter = req.waiter;
  request_queue_.Write(std::make_pair<Status, Request>(std::move(status), std::move(req)));

  // Get inference result
  NLLM_LOG_INFO << "LocalEndpoint::Handle start Wait:" << req.tokens.size();
  waiter->Wait();
  NLLM_LOG_INFO << "LocalEndpoint::Handle Wait finished.";
  Status infer_status = fetch_func_(req.req_id, output_tokens);
  NLLM_LOG_INFO << "LocalEndpoint::Handle Fetch result.";
  return infer_status;
}

}  // namespace numerous_llm
