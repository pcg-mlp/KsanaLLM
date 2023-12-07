/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <atomic>

#include "numerous_llm/endpoints/base/base_endpoint.h"
#include "numerous_llm/utils/environment.h"
#include "numerous_llm/utils/request.h"
#include "numerous_llm/utils/status.h"

namespace numerous_llm {

class Endpoint : public BaseEndpoint {
public:
  explicit Endpoint(const EndpointConfig &endpoint_config);

  // Listen at specific socket.
  Status Listen();

  // Close the listening socket.
  Status Close();

  // Wait until a request arrived.
  Status Accept(Request &req);

  // Send rsp to client.
  Status Send(const Response &rsp);

private:
  std::atomic<bool> terminated_;

  httplib::Server http_server_;
  std::thread     http_server_thread_;

  EndpointConfig endpoint_config_;
};

} // namespace numerous_llm
