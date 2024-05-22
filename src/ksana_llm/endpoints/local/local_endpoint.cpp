/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/endpoints/local/local_endpoint.h"
#include <memory>
#include "ATen/core/interned_strings.h"
#include "ksana_llm/endpoints/streaming/streaming_iterator.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

LocalEndpoint::LocalEndpoint(const EndpointConfig &endpoint_config,
                             Channel<std::pair<Status, std::shared_ptr<Request>>> &request_queue)
    : BaseEndpoint(endpoint_config, request_queue) {}

Status LocalEndpoint::Handle(const std::string &model_name, const std::vector<int> &input_tokens,
                             const SamplingConfig &sampling_config, 
                             const std::vector<int> &subinput_pos, const std::vector<std::vector<float>> &subinput_embedding,
                             std::vector<std::vector<int>> &output_tokens,
                             std::vector<std::vector<std::vector<std::pair<int, float>>>> &logprobs) {
  bool return_logprobs = sampling_config.logprobs_num > 0;
  std::shared_ptr<Request> req = std::make_shared<Request>(sampling_config, subinput_embedding);
  req->model_name = model_name;
  req->input_tokens = input_tokens;
  for (auto &[output, req_logprobs, total_score] : req->output_group) {
    output = input_tokens;
  }
  req->sampling_config = sampling_config;

  req->subinput_pos = subinput_pos;

  req->waiter = std::make_shared<Waiter>(1);

  Status status = Status();
  std::shared_ptr<Waiter> waiter = req->waiter;
  request_queue_.Write(std::pair<Status, std::shared_ptr<Request>>(status, req));

  // Get inference result
  NLLM_LOG_DEBUG << "LocalEndpoint::Handle start Wait.";
  waiter->Wait();

  NLLM_LOG_DEBUG << "LocalEndpoint::Handle Wait finished.";
  for (auto &[output, req_logprobs, total_score] : req->output_group) {
    std::vector<int> req_output = {output.begin() + req->input_tokens.size() + req->padded_size, output.end()};
    output_tokens.emplace_back(req_output);
    if (return_logprobs) logprobs.emplace_back(req_logprobs);
  }
  NLLM_LOG_DEBUG << "LocalEndpoint::Handle Fetch result.";
  return req->finish_status;
}

Status LocalEndpoint::HandleStreaming(const std::string &model_name, const std::vector<int> &input_tokens,
                                      const SamplingConfig &sampling_config,
                                      const std::vector<int> &subinput_pos, const std::vector<std::vector<float>> &subinput_embedding,
                                      std::shared_ptr<StreamingIterator> &streaming_iterator) {
  bool return_logprobs = sampling_config.logprobs_num > 0;
  std::shared_ptr<Request> req = std::make_shared<Request>(sampling_config, subinput_embedding);
  req->model_name = model_name;
  req->input_tokens = input_tokens;
  for (auto &[output, req_logprobs, total_score] : req->output_group) {
    output = input_tokens;
  }
  req->sampling_config = sampling_config;

  req->subinput_pos = subinput_pos;

  req->step_waiter = std::make_shared<Waiter>(1);

  streaming_iterator = std::make_shared<StreamingIterator>(req, return_logprobs);

  Status status = Status();
  request_queue_.Write(std::pair<Status, std::shared_ptr<Request>>(status, req));
  return status;
}

}  // namespace ksana_llm
