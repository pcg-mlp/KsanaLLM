/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/torch_op/serving_impl.h"

#include "numerous_llm/endpoints/endpoint_factory.h"
#include "numerous_llm/utils/logger.h"

namespace numerous_llm {

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
  endpoint_ = EndpointFactory::CreateLocalEndpoint(
      endpoint_config,
      [&](int64_t req_id, std::vector<int> &output_tokens) -> Status {
        return inference_engine_->FetchResult(req_id, output_tokens);
      },
      request_queue_);
}

Status ServingImpl::Handle(const std::string &model_name, const std::vector<int> &input_tokens,
                           const SamplingConfig &sampling_config, std::vector<int> &output_tokens) {
  return endpoint_->Handle(model_name, input_tokens, sampling_config, output_tokens);
}

Status ServingImpl::Start() {
  inference_engine_->Start();
  return Status();
}

Status ServingImpl::Stop() {
  NLLM_LOG_INFO << "Recive stop signal, ready to quit.";

  request_queue_.Close();
  inference_engine_->Stop();

  // Force exit here.
  NLLM_LOG_INFO << "Exit now.";
  _exit(0);

  return Status();
}

}  // namespace numerous_llm
