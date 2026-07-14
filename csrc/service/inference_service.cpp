/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "service/inference_service.h"

#include <cstdlib>
#include <stdexcept>
#include <string>

#include "endpoints/endpoint_factory.h"
#include "utils/logger.h"
#include "utils/singleton.h"

namespace ksana_llm {

InferenceService::InferenceService(const EndpointConfig &endpoint_config, std::shared_ptr<Environment> env,
                                   std::shared_ptr<Context> context) {
  KLLM_LOG_INFO << "Init inference service.";

  // Set distributed config.
  PipelineConfig pipeline_config;
  env->GetPipelineConfig(pipeline_config);
  pipeline_config.SetDistributeRelatedConfig();
  env->SetPipelineConfig(pipeline_config);

  Singleton<Environment>::GetInstance()->InitializeExpertParallelConfig();

  context->global->InitDistributedConfig();

  // Init inference engine.
  inference_engine_ = std::make_shared<InferenceEngine>(request_queue_, env, context);

  // Init local endpoint.
  local_endpoint_ = EndpointFactory::CreateLocalEndpoint(endpoint_config, request_queue_);

  // Init rpc endpoint if specified.
  if (endpoint_config.type == EndpointType::RPC) {
    rpc_endpoint_ = EndpointFactory::CreateRpcEndpoint(endpoint_config, local_endpoint_);
  }

  KLLM_LOG_INFO << "Inference service is initialized.";
}

Status InferenceService::Initialize() { return inference_engine_->Initialize(); }

void InferenceService::SetIpcFuncWrapper(std::shared_ptr<IpcFuncWrapper> ipc_func_wrapper) {
  inference_engine_->SetIpcFuncWrapper(ipc_func_wrapper);
}

InferenceService::~InferenceService() { KLLM_LOG_DEBUG << "InferenceService destroyed."; }

Status InferenceService::Handle(const std::shared_ptr<KsanaPythonInput> &ksana_python_input,
                                const std::shared_ptr<std::unordered_map<std::string, std::string>> &req_ctx,
                                ksana_llm::KsanaPythonOutput &ksana_python_output) {
  STATUS_CHECK_AND_REPORT(local_endpoint_->Handle(ksana_python_input, req_ctx, ksana_python_output));
}

Status InferenceService::HandleStreaming(const std::shared_ptr<KsanaPythonInput> &ksana_python_input,
                                         const std::shared_ptr<std::unordered_map<std::string, std::string>> &req_ctx,
                                         std::shared_ptr<StreamingIterator> &streaming_iterator) {
  STATUS_CHECK_AND_REPORT(local_endpoint_->HandleStreaming(ksana_python_input, req_ctx, streaming_iterator));
}

Status InferenceService::HandleForward(const std::string &request_bytes,
                                       const std::shared_ptr<std::unordered_map<std::string, std::string>> &req_ctx,
                                       std::string &response_bytes) {
  STATUS_CHECK_AND_REPORT(local_endpoint_->HandleForward(request_bytes, req_ctx, response_bytes));
}

Status InferenceService::Start() {
  KLLM_LOG_INFO << "Inference service start.";

  inference_engine_->Start();
  if (rpc_endpoint_) {
    rpc_endpoint_->Start();
  }

  return Status();
}

Status InferenceService::Stop() {
  KLLM_LOG_INFO << "Inference service stop.";

  inference_engine_->Stop();
  if (rpc_endpoint_) {
    rpc_endpoint_->Stop();
  }

  // Force exit here.
  KLLM_LOG_INFO << "Inference service exit.";
  return Status();
}

}  // namespace ksana_llm
