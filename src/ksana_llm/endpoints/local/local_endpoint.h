/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/endpoints/base/base_endpoint.h"

#include "ksana_llm/endpoints/streaming/streaming_iterator.h"
#include "ksana_llm/profiler/reporter.h"
#include "ksana_llm/utils/channel.h"
#include "ksana_llm/utils/request.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

class LocalEndpoint : public BaseEndpoint {
 public:
  LocalEndpoint(const EndpointConfig& endpoint_config,
                Channel<std::pair<Status, std::shared_ptr<Request>>>& request_queue);

  virtual ~LocalEndpoint() override {}

  // Handle a request.
  virtual Status Handle(const std::shared_ptr<KsanaPythonInput>& ksana_python_input,
                        const std::shared_ptr<std::unordered_map<std::string, std::string>>& req_ctx,
                        ksana_llm::KsanaPythonOutput& ksana_python_output);

  // Handle a streaming request.
  virtual Status HandleStreaming(const std::shared_ptr<KsanaPythonInput>& ksana_python_input,
                                 const std::shared_ptr<std::unordered_map<std::string, std::string>>& req_ctx,
                                 std::shared_ptr<StreamingIterator>& streaming_iterator);

  // Handle a batch request.
  virtual Status HandleBatch(const std::vector<std::shared_ptr<KsanaPythonInput>>& ksana_python_inputs,
                             const std::shared_ptr<std::unordered_map<std::string, std::string>>& req_ctx,
                             std::vector<KsanaPythonOutput>& ksana_python_outputs);
};

}  // namespace ksana_llm
