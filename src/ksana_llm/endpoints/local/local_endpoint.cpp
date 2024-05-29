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

Status LocalEndpoint::Handle(const ksana_llm::KsanaPythonInput &ksana_python_input,
                             ksana_llm::KsanaPythonOutput &ksana_python_output) {
  std::shared_ptr<Request> req = std::make_shared<Request>(ksana_python_input);
  for (auto &[output, req_logprobs, total_score] : req->output_group) {
    output = ksana_python_input.input_tokens;
  }
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
    ksana_python_output.output_tokens.emplace_back(req_output);
    if (ksana_python_input.sampling_config.logprobs_num > 0) {
      ksana_python_output.logprobs.emplace_back(req_logprobs);
    }
  }
  ksana_python_output.prompt_probs = std::move(req->prompt_probs);
  NLLM_LOG_DEBUG << "LocalEndpoint::Handle Fetch result.";
  return req->finish_status;
}

Status LocalEndpoint::HandleStreaming(const ksana_llm::KsanaPythonInput &ksana_python_input,
                                      std::shared_ptr<StreamingIterator> &streaming_iterator) {
  std::shared_ptr<Request> req = std::make_shared<Request>(ksana_python_input);
  for (auto &[output, req_logprobs, total_score] : req->output_group) {
    output = ksana_python_input.input_tokens;
  }
  req->step_waiter = std::make_shared<Waiter>(1);

  streaming_iterator = std::make_shared<StreamingIterator>(req, ksana_python_input.sampling_config.logprobs_num > 0);

  Status status = Status();
  request_queue_.Write(std::pair<Status, std::shared_ptr<Request>>(status, req));
  return status;
}

}  // namespace ksana_llm
