/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/endpoints/http/http_endpoint.h"
#include <memory>

#include "ksana_llm/utils/ret_code.h"
#include "ksana_llm/utils/status.h"
#include "nlohmann/json.hpp"

namespace ksana_llm {

HttpEndpoint::HttpEndpoint(const EndpointConfig &endpoint_config,
                           Channel<std::pair<Status, std::shared_ptr<Request>>> &request_queue)
    : RpcEndpoint(endpoint_config, request_queue) {}

Status HttpEndpoint::Accept(std::shared_ptr<Request> &req) {
  if (terminated_) {
    return Status(RET_TERMINATED);
  }
  NLLM_LOG_DEBUG << "Accept a req.";

  SamplingConfig sampling_config;
  req->sampling_config = sampling_config;
  req->waiter = std::make_shared<Waiter>(1);

  return Status();
}

Status HttpEndpoint::Send(const Status infer_status, const std::shared_ptr<Request> &req, httplib::Response &http_rsp) {
  nlohmann::json_abi_v3_11_2::json result_json;

  if (infer_status.OK()) {
    std::vector<int> output_tokens = {req->output_tokens.begin() + req->input_tokens.size(), req->output_tokens.end()};
    result_json["output_tokens"] = output_tokens;
    result_json["tokens_len"] = output_tokens.size();
    http_rsp.set_content(result_json.dump(), "text/plain");
  } else {
    http_rsp.status = httplib::StatusCode::InternalServerError_500;
  }

  return Status();
}

Status HttpEndpoint::HandleRequest(const httplib::Request &http_req, httplib::Response &http_rsp) {
  if (http_req.has_param("input_tokens")) {
    std::shared_ptr<Request> req = std::make_shared<Request>();
    req->model_name = http_req.get_param_value("model_name");

    int input_tokens_length = std::stoi(http_req.get_param_value("tokens_len", 0));
    std::vector<int> tokens_vec(input_tokens_length);
    for (int v_id = 0; v_id < input_tokens_length; ++v_id) {
      tokens_vec[v_id] = std::stoi(http_req.get_param_value("input_tokens", v_id));
    }
    req->input_tokens = tokens_vec;
    req->output_tokens = tokens_vec;

    Status req_prepare_status = Accept(req);
    std::shared_ptr<Waiter> waiter = req->waiter;
    request_queue_.Write(std::pair<Status, std::shared_ptr<Request>>(req_prepare_status, req));

    // Get inference result
    waiter->Wait();
    Send(req->finish_status, req, http_rsp);

    return Status();
  }
  NLLM_LOG_ERROR << "Invalid http request [input_tokens not exists].";
  return Status(RET_INVALID_ARGUMENT, "Invalid http request.");
}

Status HttpEndpoint::Start() {
  NLLM_LOG_DEBUG << "Listen on port " << endpoint_config_.port;

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
  NLLM_LOG_DEBUG << "Close http endpoint.";
  http_server_.stop();
  terminated_ = true;
  return Status();
}

}  // namespace ksana_llm
