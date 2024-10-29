/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/endpoints/local/local_endpoint.h"
#include <memory>
#include "ATen/core/interned_strings.h"
#include "ksana_llm/endpoints/streaming/streaming_iterator.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

LocalEndpoint::LocalEndpoint(const EndpointConfig &endpoint_config,
                             Channel<std::pair<Status, std::shared_ptr<Request>>> &request_queue)
    : BaseEndpoint(endpoint_config, request_queue) {}

Status LocalEndpoint::Handle(const std::shared_ptr<KsanaPythonInput> &ksana_python_input,
                             const std::shared_ptr<std::unordered_map<std::string, std::string>> &req_ctx,
                             ksana_llm::KsanaPythonOutput &ksana_python_output) {
  std::shared_ptr<Request> req = std::make_shared<Request>(ksana_python_input, req_ctx);
  req->waiter = std::make_shared<Waiter>(1);
  opentelemetry::trace::StartSpanOptions options;
  auto span = REPORT_TRACE(local_endpoint_handle, options);
  opentelemetry::trace::Scope scope(span);
  req->span_context = span->GetContext();

  Status status = Status();
  std::shared_ptr<Waiter> waiter = req->waiter;
  request_queue_.Write(std::pair<Status, std::shared_ptr<Request>>(status, req));

  // Get inference result
  KLLM_LOG_DEBUG << "LocalEndpoint::Handle start Wait.";
  waiter->Wait();
  KLLM_LOG_DEBUG << "LocalEndpoint::Handle Wait finished.";

  ksana_python_output = KsanaPythonOutput(req);
  KLLM_LOG_DEBUG << "LocalEndpoint::Handle Fetch result.";
  STATUS_CHECK_AND_REPORT(req->finish_status, span);
}

Status LocalEndpoint::HandleStreaming(const std::shared_ptr<KsanaPythonInput> &ksana_python_input,
                                      const std::shared_ptr<std::unordered_map<std::string, std::string>> &req_ctx,
                                      std::shared_ptr<StreamingIterator> &streaming_iterator) {
  std::shared_ptr<Request> req = std::make_shared<Request>(ksana_python_input, req_ctx);
  req->step_waiter = std::make_shared<Waiter>(1);
  req->abort_waiter = std::make_shared<Waiter>(1);

  opentelemetry::trace::StartSpanOptions options;
  auto span = REPORT_TRACE(local_endpoint_handle_streaming, options);
  opentelemetry::trace::Scope scope(span);
  req->span_context = span->GetContext();

  streaming_iterator = std::make_shared<StreamingIterator>(req, ksana_python_input);

  Status status = Status();
  request_queue_.Write(std::pair<Status, std::shared_ptr<Request>>(status, req));
  STATUS_CHECK_AND_REPORT(status, span);
}

Status LocalEndpoint::HandleBatch(const std::vector<std::shared_ptr<KsanaPythonInput>> &ksana_python_inputs,
                                  const std::shared_ptr<std::unordered_map<std::string, std::string>> &req_ctx,
                                  std::vector<KsanaPythonOutput> &ksana_python_outputs) {
  const size_t batch_size = ksana_python_inputs.size();
  auto waiter = std::make_shared<Waiter>(batch_size);
  std::vector<std::pair<Status, std::shared_ptr<Request>>> reqs;
  reqs.reserve(batch_size);

  opentelemetry::trace::StartSpanOptions options;
  auto span = REPORT_TRACE(local_endpoint_handle_batch, options);
  opentelemetry::trace::Scope scope(span);
  for (size_t i = 0; i < batch_size; i++) {
    const auto &ksana_python_input = ksana_python_inputs[i];
    auto req = std::make_shared<Request>(ksana_python_input, req_ctx);
    req->waiter = waiter;
    req->last_in_batch = (i == batch_size - 1);
    req->span_context = span->GetContext();
    reqs.emplace_back(Status(), std::move(req));
  }
  // Write the batch of requests once
  request_queue_.Write(reqs.data(), batch_size);

  // Get inference result
  KLLM_LOG_DEBUG << "LocalEndpoint::HandleBatch start Wait.";
  waiter->Wait();
  KLLM_LOG_DEBUG << "LocalEndpoint::HandleBatch Wait finished.";

  ksana_python_outputs.reserve(batch_size);
  for (const auto &[_, req] : reqs) {
    ksana_python_outputs.emplace_back(req);
  }
  KLLM_LOG_DEBUG << "LocalEndpoint::HandleBatch Fetch result.";
  return Status();
}

}  // namespace ksana_llm
