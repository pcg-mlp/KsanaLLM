/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/endpoints/endpoint.h"

#include <chrono>
#include <thread>

#include "numerous_llm/utils/logger.h"

namespace numerous_llm {

Endpoint::Endpoint(const EndpointConfig &endpoint_config) : endpoint_config_(endpoint_config) {}

Status Endpoint::Listen(Channel<std::pair<Status, Request>> &requests_queue, Channel<std::pair<Status, Response>>& response_queue) {
  NLLM_LOG_INFO << "Listen on port " << endpoint_config_.port;

  // define generate
  // TODO(karlluo): should also support stream mode
  http_server_.Post("/generate", [&](const httplib::Request &req, httplib::Response &res) {
    if (req.has_param("tokens")) {
      Request infer_req;
      std::vector<std::vector<int>> tokens;
      uint32_t batch_size = static_cast<uint32_t>(req.get_param_value_count("tokens_len"));
      for (size_t t_l_id = 0; t_l_id < batch_size; ++t_l_id) {
        int input_tokens_length = std::stoi(req.get_param_value("tokens_len", t_l_id));
        std::vector<int> tokens_vec(input_tokens_length);
        for (int v_id = 0; v_id < input_tokens_length; ++v_id) {
          tokens_vec[v_id] = std::stoi(req.get_param_value("tokens", v_id));
        }
        tokens.emplace_back(std::move(tokens_vec));
      }
      // At this moment the shape of tokens is [batch_size, each prompt's tokens number] for example:
      // examples/llama13b/llama13b_simple_client.py requests 2 tokens list [[1,2,3],[4,5,6,7,8]],
      // tokens[0] shape is [3]
      // tokens[1] shape is [5]
      // TODO(karlluo): Convert token to tensor
      Status req_prepare_status = Accept(infer_req);
      requests_queue.Write(std::make_pair<Status, Request>(std::move(req_prepare_status), std::move(infer_req)));

      // Get inference result
      std::pair<Status, Response> rsp_pair;
      response_queue.Read(&rsp_pair);
      Status rsp_prepare_status = Send(rsp_pair.second);
    }
  });

  // define logger
  http_server_.set_logger([](const auto &req, const auto &res) {});

  http_server_thread_ = std::thread([&]() { http_server_.listen(endpoint_config_.host, endpoint_config_.port); });
  http_server_thread_.detach();

  return Status();
}

Status Endpoint::Close() {
  NLLM_LOG_INFO << "Close endpoint.";
  http_server_.stop();
  http_server_thread_.join();
  terminated_ = true;
  return Status();
}

Status Endpoint::Accept(Request &req) {
  static int count = 0;

  if (terminated_) {
    return Status(RET_TERMINATED);
  }

  ++count;
  NLLM_LOG_INFO << "Accept a req.";

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
  NLLM_LOG_INFO << "Send a rsp.";
  return Status();
}

}  // namespace numerous_llm
