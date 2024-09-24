/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/torch_op/serving_impl.h"

#include "ksana_llm/endpoints/endpoint_factory.h"
#include "ksana_llm/profiler/reporter.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/singleton.h"

namespace ksana_llm {

ServingImpl::ServingImpl() {
  inference_engine_ = std::make_shared<InferenceEngine>(request_queue_);

  std::shared_ptr<Environment> env = Singleton<Environment>::GetInstance();
  if (!env) {
    KLLM_THROW("The Environment is nullptr.");
  }

  EndpointConfig endpoint_config;
  Status status = env->GetEndpointConfig(endpoint_config);
  if (!status.OK()) {
    KLLM_THROW("Get endpoint config error:" + status.ToString());
  }

  // Create local endpoint.
  endpoint_config.type = EndpointType::ENDPOINT_LOCAL;
  endpoint_ = EndpointFactory::CreateLocalEndpoint(endpoint_config, request_queue_);
}

Status ServingImpl::Handle(const std::shared_ptr<KsanaPythonInput> &ksana_python_input,
                           const std::shared_ptr<std::unordered_map<std::string, std::string>> &req_ctx,
                           ksana_llm::KsanaPythonOutput &ksana_python_output) {
  opentelemetry::trace::StartSpanOptions options;
  auto span = REPORT_TRACE(serving_impl_handle, options);
  opentelemetry::trace::Scope scope(span);
  STATUS_CHECK_AND_REPORT(endpoint_->Handle(ksana_python_input, req_ctx, ksana_python_output), span);
}

Status ServingImpl::HandleStreaming(const std::shared_ptr<KsanaPythonInput> &ksana_python_input,
                                    const std::shared_ptr<std::unordered_map<std::string, std::string>> &req_ctx,
                                    std::shared_ptr<StreamingIterator> &streaming_iterator) {
  opentelemetry::trace::StartSpanOptions options;
  auto span = REPORT_TRACE(serving_impl_handle_streaming, options);
  opentelemetry::trace::Scope scope(span);
  STATUS_CHECK_AND_REPORT(endpoint_->HandleStreaming(ksana_python_input, req_ctx, streaming_iterator), span);
}

Status ServingImpl::HandleBatch(const std::vector<std::shared_ptr<KsanaPythonInput>> &ksana_python_inputs,
                                const std::shared_ptr<std::unordered_map<std::string, std::string>> &req_ctx,
                                std::vector<KsanaPythonOutput> &ksana_python_outputs) {
  KLLM_LOG_DEBUG << "Processing Of ServingImpl::HandleBatch";
  opentelemetry::trace::StartSpanOptions options;
  auto span = REPORT_TRACE(serving_impl_handle_batch, options);
  opentelemetry::trace::Scope scope(span);
  STATUS_CHECK_AND_REPORT(endpoint_->HandleBatch(ksana_python_inputs, req_ctx, ksana_python_outputs), span);
}

Status ServingImpl::Start() {
  inference_engine_->Start();
  return Status();
}

Status ServingImpl::Stop() {
  KLLM_LOG_DEBUG << "Recive stop signal, ready to quit.";

  request_queue_.Close();
  inference_engine_->Stop();

  // Force exit here.
  KLLM_LOG_DEBUG << "Exit now.";
  _exit(0);

  return Status();
}

}  // namespace ksana_llm
