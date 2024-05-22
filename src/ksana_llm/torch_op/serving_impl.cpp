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

Status ServingImpl::Handle(const std::string &model_name, const std::vector<int> &input_tokens,
                           const SamplingConfig &sampling_config, 
                           const std::vector<int> &subinput_pos, const std::vector<std::vector<float>> &subinput_embedding,
                           std::vector<std::vector<int>> &output_tokens,
                           std::vector<std::vector<std::vector<std::pair<int, float>>>> &logprobs) {
  // TODO(jinxcwu): 
  // combine input_tokens, subinput_pos, subinput_embedding for Struct input
  // combine output_tokens, logprobs for Struct output
  return endpoint_->Handle(model_name, input_tokens, sampling_config, subinput_pos, subinput_embedding, output_tokens, logprobs);
}

Status ServingImpl::HandleStreaming(const std::string &model_name, const std::vector<int> &input_tokens,
                                    const SamplingConfig &sampling_config,
                                    const std::vector<int> &subinput_pos, const std::vector<std::vector<float>> &subinput_embedding,
                                    std::shared_ptr<StreamingIterator> &streaming_iterator) {
  return endpoint_->HandleStreaming(model_name, input_tokens, sampling_config, subinput_pos, subinput_embedding, streaming_iterator);
}

Status ServingImpl::Start() {
  inference_engine_->Start();
  return Status();
}

Status ServingImpl::Stop() {
  NLLM_LOG_DEBUG << "Recive stop signal, ready to quit.";

  request_queue_.Close();
  inference_engine_->Stop();

  // Force exit here.
  NLLM_LOG_DEBUG << "Exit now.";
  _exit(0);

  return Status();
}

}  // namespace ksana_llm
