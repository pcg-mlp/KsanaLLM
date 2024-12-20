/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/service/inference_server.h"
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "ksana_llm/endpoints/endpoint_factory.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/singleton.h"

namespace ksana_llm {

InferenceServer::InferenceServer(const std::string &config_file, const EndpointConfig &endpoint_config) {
  InitLoguru();
  KLLM_LOG_INFO << "Log INFO level: " << GetLevelName(GetLogLevel());

  KLLM_LOG_INFO << "Init inference server with config file: " << config_file;
  auto env = Singleton<Environment>::GetInstance();
  if (!env) {
    KLLM_THROW("The Environment is nullptr.");
  }
  STATUS_CHECK_FAILURE(env->ParseConfig(config_file));
  InitializePipelineConfig();

  // Init inference engine.
  inference_engine_ = std::make_shared<InferenceEngine>(request_queue_);

  // Init local endpoint.
  local_endpoint_ = EndpointFactory::CreateLocalEndpoint(endpoint_config, request_queue_);
  // Init rpc endpoint if specified.
  if (endpoint_config.type == EndpointType::RPC) {
    rpc_endpoint_ = EndpointFactory::CreateRpcEndpoint(endpoint_config, local_endpoint_);
  }

  KLLM_LOG_INFO << "Inference server is initialized.";
}

InferenceServer::~InferenceServer() { KLLM_LOG_DEBUG << "InferenceServer destroyed."; }

void InferenceServer::InitializePipelineConfig() {
  const char *master_host = std::getenv("MASTER_HOST");
  const char *master_port = std::getenv("MASTER_PORT");
  const char *world_size = std::getenv("WORLD_SIZE");
  const char *node_rank = std::getenv("NODE_RANK");

  PipelineConfig pipeline_config;
  Singleton<Environment>::GetInstance()->GetPipelineConfig(pipeline_config);

  pipeline_config.world_size = world_size ? std::stoi(world_size) : 1;
  pipeline_config.node_rank = node_rank ? std::stoi(node_rank) : 0;
  if (pipeline_config.world_size > 1) {
    if (!master_host || !master_port) {
      throw std::runtime_error("The environment variable MASTER_HOST and MASTER_PORT must be set in distributed mode.");
    }
  }

  pipeline_config.master_host = master_host ? master_host : "";
  pipeline_config.master_port = master_port ? std::stoi(master_port) : 0;

  KLLM_LOG_INFO << "InferenceServer initialize pipeline config, master_host:" << pipeline_config.master_host
                << ", master_port:" << pipeline_config.master_port << ", world_size:" << pipeline_config.world_size
                << ", node_rank:" << pipeline_config.node_rank;
  Singleton<Environment>::GetInstance()->SetPipelineConfig(pipeline_config);
}

Status InferenceServer::Handle(const std::shared_ptr<KsanaPythonInput> &ksana_python_input,
                               const std::shared_ptr<std::unordered_map<std::string, std::string>> &req_ctx,
                               ksana_llm::KsanaPythonOutput &ksana_python_output) {
  const HttpTextMapCarrier<const std::unordered_map<std::string, std::string>> carrier(*req_ctx);
  auto span = REPORT_TRACE(inference_server_handle_span, carrier);
  opentelemetry::trace::Scope scope(span);
  STATUS_CHECK_AND_REPORT(local_endpoint_->Handle(ksana_python_input, req_ctx, ksana_python_output), span);
}

Status InferenceServer::HandleStreaming(const std::shared_ptr<KsanaPythonInput> &ksana_python_input,
                                        const std::shared_ptr<std::unordered_map<std::string, std::string>> &req_ctx,
                                        std::shared_ptr<StreamingIterator> &streaming_iterator) {
  const HttpTextMapCarrier<const std::unordered_map<std::string, std::string>> carrier(*req_ctx);
  auto span = REPORT_TRACE(inference_server_handle_streaming_span, carrier);
  opentelemetry::trace::Scope scope(span);
  STATUS_CHECK_AND_REPORT(local_endpoint_->HandleStreaming(ksana_python_input, req_ctx, streaming_iterator), span);
}

Status InferenceServer::HandleForward(const std::string &request_bytes,
                                      const std::shared_ptr<std::unordered_map<std::string, std::string>> &req_ctx,
                                      std::string &response_bytes) {
  const HttpTextMapCarrier<const std::unordered_map<std::string, std::string>> carrier(*req_ctx);
  auto span = REPORT_TRACE(inference_server_handle_forward_span, carrier);
  opentelemetry::trace::Scope scope(span);
  STATUS_CHECK_AND_REPORT(local_endpoint_->HandleForward(request_bytes, req_ctx, response_bytes), span);
}

Status InferenceServer::Start() {
  KLLM_LOG_INFO << "Inference server start.";

  inference_engine_->Start();
  if (rpc_endpoint_) {
    rpc_endpoint_->Start();
  }

  return Status();
}

Status InferenceServer::Stop() {
  KLLM_LOG_INFO << "Inference server stop.";

  inference_engine_->Stop();
  if (rpc_endpoint_) {
    rpc_endpoint_->Stop();
  }

  // Force exit here.
  KLLM_LOG_INFO << "Inference server exit.";
  return Status();
}

}  // namespace ksana_llm
