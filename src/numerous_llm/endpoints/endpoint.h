/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <atomic>

#include "numerous_llm/endpoints/base/base_endpoint.h"
#include "numerous_llm/utils/environment.h"
#include "numerous_llm/utils/request.h"
#include "numerous_llm/utils/status.h"
#include "src/numerous_llm/utils/channel.h"

namespace numerous_llm {

class Endpoint : public BaseEndpoint {
 public:
  explicit Endpoint(const EndpointConfig &endpoint_config);

  // Listen at specific socket.
  Status Listen(Channel<std::pair<Status, Request>> &requests_queue, std::mutex &response_container_mutex,
                std::unordered_map<int64_t, std::pair<Status, Response>> &response_container);

  // Close the listening socket.
  Status Close();

  // Wait until a request arrived.
  Status Accept(Request &req);

  // Send rsp to client.
  Status Send(const Status infer_status, const Response &rsp, httplib::Response &res);

 private:
  std::atomic<bool> terminated_{false};

  httplib::Server http_server_;
  std::thread http_server_thread_;

  EndpointConfig endpoint_config_;
};

}  // namespace numerous_llm
