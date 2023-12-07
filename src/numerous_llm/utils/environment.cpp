/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/utils/environment.h"

#include <fstream>
#include <iostream>
#include <stdexcept>

#include "fmt/core.h"
#include "gflags/gflags.h"

#include "3rdparty/ini_reader.h"
#include "numerous_llm/utils/logger.h"

DEFINE_string(model_config, "./config.ini", "Get the model config file path");
DEFINE_string(host, "localhost", "HTTP service hostname, default is localhost");
DEFINE_int32(port, 8080, "HTTP service port, default is 8080");

namespace numerous_llm {

inline bool IsFileExists(const std::string &file_path) {
  std::ifstream f(file_path.c_str());
  return f.good();
}

Status Environment::ParseOptions(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (!IsFileExists(FLAGS_model_config)) {
    NLLM_LOG_ERROR << fmt::format("Model config file: {} is not exists.", FLAGS_model_config) << std::endl;
    return Status(RetCode::RET_SEGMENT_FAULT);
  }

  INIReader ini_reader = INIReader(FLAGS_model_config);
  if (ini_reader.ParseError() < 0) {
    NLLM_LOG_ERROR << fmt::format("Load model config file: {} error.", FLAGS_model_config) << std::endl;
    return Status(RetCode::RET_SEGMENT_FAULT);
  }

  ModelConfig model_config;
  model_config.name = ini_reader.Get("ft_instance_hyperparameter", "model_name");
  model_config.path = FLAGS_model_config;
  model_configs_.push_back(model_config);

  NLLM_LOG_INFO << fmt::format("Load model {} from config file: {} success.", model_config.name, model_config.path);

  endpoint_config_.host = FLAGS_host;
  endpoint_config_.port = static_cast<uint32_t>(FLAGS_port);

  endpoint_config_.host = FLAGS_host;
  endpoint_config_.port = static_cast<uint32_t>(FLAGS_port);

  return Status();
}

Status Environment::GetModelList(std::vector<ModelConfig> &model_configs) {
  model_configs = model_configs_;
  return Status();
}

Status Environment::GetBatchManagerConfig(BatchManagerConfig &batch_manager_config) {
  batch_manager_config = batch_manager_config_;
  return Status();
}

Status Environment::GetEndpointConfig(EndpointConfig &endpoint_config) {
  endpoint_config = endpoint_config_;
  return Status();
}

}  // namespace numerous_llm
