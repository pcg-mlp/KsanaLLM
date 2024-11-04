/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "httplib.h"
#include "ksana_llm/endpoints/rpc/rpc_endpoint.h"

namespace ksana_llm {

class Request;

// The HTTP endpoint, used to receive request from http client.
class HttpEndpoint : public RpcEndpoint {
 public:
  HttpEndpoint(const EndpointConfig &endpoint_config, const std::shared_ptr<LocalEndpoint> &local_endpoint);

  virtual ~HttpEndpoint() override {}

  // Listen at specific socket.
  virtual Status Start() override;

  // Close the listening socket.
  virtual Status Stop() override;

 private:
  // Wait until a request arrived.
  Status Accept(std::shared_ptr<Request> &req);

  // Send rsp to client.
  Status Send(const Status infer_status, const std::shared_ptr<Request> &req, httplib::Response &http_rsp);

  // Handle the http request.
  Status HandleRequest(const httplib::Request &http_req, httplib::Response &http_rsp);

 private:
  // The terminate flag to control processing loop.
  std::atomic<bool> terminated_{false};

  // The http server instance.
  httplib::Server http_server_;

  // The http server thread.
  std::thread http_server_thread_;
};

}  // namespace ksana_llm
