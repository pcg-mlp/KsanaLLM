/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <csignal>
#include <iostream>

#include "numerous_llm/service/inference_server.h"
#include "numerous_llm/utils/environment.h"
#include "numerous_llm/utils/logger.h"
#include "numerous_llm/utils/status.h"

using namespace numerous_llm;

static std::shared_ptr<InferenceServer> server = nullptr;

void SignalHandler(int signum) {
  if (server) {
    server->StopServer();
    server = nullptr;
  }
}

int main(int argc, char **argv) {
  // Install signal handler.
  signal(SIGINT, SignalHandler);
  signal(SIGQUIT, SignalHandler);
  signal(SIGTERM, SignalHandler);

  // Parse command line options.
  std::shared_ptr<Environment> env = std::make_shared<Environment>();
  Status status = env->ParseOptions(argc, argv);
  if (!status.OK()) {
    std::cerr << status.ToString() << std::endl;
    return 1;
  }

  // Initialize inference server
  server = std::make_shared<InferenceServer>();
  status = server->Initialize(env);
  if (!status.OK()) {
    NLLM_LOG_ERROR << "Init inference server error: " << status.ToString()
                   << std::endl;
    return 1;
  }

  status = server->StartServer();
  if (!status.OK()) {
    NLLM_LOG_ERROR << "Start inference server error: " << status.ToString()
                   << std::endl;
    return 1;
  }

  return 0;
}
