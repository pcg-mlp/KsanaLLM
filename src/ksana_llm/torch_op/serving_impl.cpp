/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/torch_op/serving_impl.h"

#include "ksana_llm/endpoints/endpoint_factory.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/singleton.h"

namespace ksana_llm {

ServingImpl::ServingImpl() {
  inference_engine_ = std::make_shared<InferenceEngine>(request_queue_);

  std::shared_ptr<Environment> env = Singleton<Environment>::GetInstance();
  if (!env) {
    throw std::runtime_error("The Environment is nullptr.");
  }

  EndpointConfig endpoint_config;
  Status status = env->GetEndpointConfig(endpoint_config);
  if (!status.OK()) {
    throw std::runtime_error("Get endpoint config error:" + status.ToString());
  }

  // Create local endpoint.
  endpoint_config.type = EndpointType::ENDPOINT_LOCAL;
  endpoint_ = EndpointFactory::CreateLocalEndpoint(endpoint_config, request_queue_);
}

Status ServingImpl::Handle(const std::shared_ptr<KsanaPythonInput> &ksana_python_input,
                           ksana_llm::KsanaPythonOutput &ksana_python_output) {
  return endpoint_->Handle(ksana_python_input, ksana_python_output);
}

Status ServingImpl::HandleStreaming(const std::shared_ptr<KsanaPythonInput> &ksana_python_input,
                                    std::shared_ptr<StreamingIterator> &streaming_iterator) {
  return endpoint_->HandleStreaming(ksana_python_input, streaming_iterator);
}

Status ServingImpl::HandleBatch(const std::vector<std::shared_ptr<KsanaPythonInput>> &ksana_python_inputs,
                                std::vector<KsanaPythonOutput> &ksana_python_outputs) {
  return endpoint_->HandleBatch(ksana_python_inputs, ksana_python_outputs);
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
