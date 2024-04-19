/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/service/inference_server.h"

#include <iostream>
#include <memory>
#include <stdexcept>

#include "ksana_llm/endpoints/endpoint_factory.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/waiter.h"

namespace ksana_llm {

InferenceServer::InferenceServer() {
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

  // Create rpc endpoint.
  endpoint_config.type = EndpointType::ENDPOINT_HTTP;
  endpoint_ = EndpointFactory::CreateRpcEndpoint(endpoint_config, request_queue_);

  waiter_ = std::make_shared<Waiter>(1);
}

Status InferenceServer::WaitUntilStop() {
  waiter_->Wait();
  return Status();
}

Status InferenceServer::Start() {
  inference_engine_->Start();
  endpoint_->Start();

  WaitUntilStop();
  return Status();
}

Status InferenceServer::Stop() {
  NLLM_LOG_DEBUG << "Recive stop signal, ready to quit.";

  request_queue_.Close();

  endpoint_->Stop();
  inference_engine_->Stop();

  waiter_->Notify();

  // Force exit here.
  NLLM_LOG_DEBUG << "Exit now.";
  _exit(0);

  return Status();
}

}  // namespace ksana_llm.
