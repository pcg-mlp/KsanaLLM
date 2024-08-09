/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/endpoints/base/base_endpoint.h"

#include "ksana_llm/endpoints/streaming/streaming_iterator.h"
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
                        ksana_llm::KsanaPythonOutput& ksana_python_output);

  // handle a streaming request.
  virtual Status HandleStreaming(const std::shared_ptr<KsanaPythonInput>& ksana_python_input,
                                 std::shared_ptr<StreamingIterator>& streaming_iterator);
};

}  // namespace ksana_llm
