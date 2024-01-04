/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/endpoints/http/http_endpoint.h"

#include "nlohmann/json.hpp"
#include "numerous_llm/utils/ret_code.h"
#include "numerous_llm/utils/status.h"

namespace numerous_llm {

HttpEndpoint::HttpEndpoint(const EndpointConfig &endpoint_config,
                           std::function<Status(int64_t, std::vector<std::vector<int>> &)> fetch_func,
                           Channel<std::pair<Status, Request>> &request_queue)
    : RpcEndpoint(endpoint_config, fetch_func, request_queue) {}

Status HttpEndpoint::Accept(Request &req) {
  if (terminated_) {
    return Status(RET_TERMINATED);
  }
  NLLM_LOG_INFO << "Accept a req.";

  std::vector<SamplingConfig> sampling_configs(req.tokens.size());
  req.sampling_configs = sampling_configs;
  req.waiter = std::make_shared<Waiter>(req.tokens.size());

  return Status();
}

Status HttpEndpoint::Send(const Status infer_status, const Response &rsp, httplib::Response &http_rsp) {
  nlohmann::json_abi_v3_11_2::json result_json;

  if (infer_status.OK()) {
    result_json["tokens"] = rsp.tokens;
    std::vector<size_t> token_lens;
    for (const auto &tokens : rsp.tokens) {
      token_lens.push_back(tokens.size());
    }
    result_json["tokens_len"] = token_lens;
    http_rsp.set_content(result_json.dump(), "text/plain");
  } else {
    http_rsp.status = httplib::StatusCode::InternalServerError_500;
  }

  return Status();
}

Status HttpEndpoint::HandleRequest(const httplib::Request &http_req, httplib::Response &http_rsp) {
  if (http_req.has_param("tokens")) {
    Request req;
    req.model_name = http_req.get_param_value("model_name");
    uint32_t batch_size = static_cast<uint32_t>(http_req.get_param_value_count("tokens_len"));
    uint32_t offset = 0;
    for (size_t t_l_id = 0; t_l_id < batch_size; ++t_l_id) {
      int input_tokens_length = std::stoi(http_req.get_param_value("tokens_len", t_l_id));
      std::vector<int> tokens_vec(input_tokens_length);
      for (int v_id = 0; v_id < input_tokens_length; ++v_id) {
        tokens_vec[v_id] = std::stoi(http_req.get_param_value("tokens", v_id + offset));
      }
      offset += input_tokens_length;
      req.tokens.emplace_back(std::move(tokens_vec));
    }

    // TODO(yancyliu): Make sure size of tokens and sampling_configs are equal.

    // At this moment the shape of tokens is [batch_size, each prompt's tokens number] for example:
    // examples/llama13b/llama13b_simple_client.py requests 2 tokens list [[1,2,3],[4,5,6,7,8]],
    // tokens[0] shape is [3]
    // tokens[1] shape is [5]
    Status req_prepare_status = Accept(req);
    std::shared_ptr<Waiter> waiter = req.waiter;
    request_queue_.Write(std::make_pair<Status, Request>(std::move(req_prepare_status), std::move(req)));

    // Get inference result
    Response rsp;
    rsp.req_id = req.req_id;
    waiter->Wait();
    Status infer_status = fetch_func_(req.req_id, rsp.tokens);
    Send(infer_status, rsp, http_rsp);

    return Status();
  }
  return Status(RET_INVALID_ARGUMENT, "Invalid http request.");
}

Status HttpEndpoint::Start() {
  NLLM_LOG_INFO << "Listen on port " << endpoint_config_.port;

  // define generate
  // TODO(karlluo): should also support stream mode
  http_server_.Post("/generate", [&](const httplib::Request &http_req, httplib::Response &http_rsp) {
    HandleRequest(http_req, http_rsp);
  });

  // define logger
  http_server_.set_logger([](const auto &http_req, const auto &http_rsp) {});

  http_server_thread_ = std::thread([&]() { http_server_.listen(endpoint_config_.host, endpoint_config_.port); });
  http_server_thread_.detach();

  return Status();
}

Status HttpEndpoint::Stop() {
  NLLM_LOG_INFO << "Close http endpoint.";
  http_server_.stop();
  terminated_ = true;
  return Status();
}

}  // namespace numerous_llm
