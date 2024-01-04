/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <csignal>
#include <iostream>

#include "numerous_llm/service/inference_server.h"
#include "numerous_llm/utils/environment.h"
#include "numerous_llm/utils/logger.h"
#include "numerous_llm/utils/singleton.h"
#include "numerous_llm/utils/status.h"

using namespace numerous_llm;

static std::shared_ptr<InferenceServer> server = nullptr;

void SignalHandler(int signum) {
  if (server) {
    // Skip dup signals.
    signal(SIGINT, SIG_IGN);
    signal(SIGQUIT, SIG_IGN);
    signal(SIGTERM, SIG_IGN);

    server->Stop();
    server = nullptr;
  }
}

int main(int argc, char** argv) {
  // Initialize logger.
  InitLoguru();
  NLLM_LOG_INFO << "Log level: " << GetLevelName(GetLogLevel());

  // Install signal handler.
  signal(SIGINT, SignalHandler);
  signal(SIGQUIT, SignalHandler);
  signal(SIGTERM, SignalHandler);

  // Parse command line options.
  Status status = Singleton<Environment>::GetInstance()->ParseOptions(argc, argv);
  if (!status.OK()) {
    std::cerr << status.ToString() << std::endl;
    return 1;
  }

  // Initialize inference server
  server = std::make_shared<InferenceServer>();
  status = server->Start();
  if (!status.OK()) {
    NLLM_LOG_ERROR << "Start inference server error: " << status.ToString() << std::endl;
    return 1;
  }

  return 0;
}
