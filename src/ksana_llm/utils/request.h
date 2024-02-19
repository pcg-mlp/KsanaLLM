/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <sys/stat.h>
#include <vector>

#include "ksana_llm/utils/id_generator.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/tensor.h"
#include "ksana_llm/utils/waiter.h"

namespace ksana_llm {

struct SamplingConfig {
  int beam_width = 1;
  int topk = 1;
  float topp = 0.0f;
  float temperature = 0.0f;
};

class Request {
 public:
  Request();

  // The unique id of a request.
  int64_t req_id;

  // The requested model name.
  std::string model_name;

  // The tokens of this request.
  std::vector<int> input_tokens;

  // The output tokens of this request.
  std::vector<int> output_tokens;

  // The config of sampling.
  SamplingConfig sampling_config;

  // The waiter notified when request finished.
  std::shared_ptr<Waiter> waiter = nullptr;

  // The waiter notified when step finished.
  std::shared_ptr<Waiter> step_waiter = nullptr;

  // Whether the request is finished.
  bool finished = false;

  // The finish statu of this request.
  Status finish_status;

  // Protect parallel access for output token.
  std::mutex output_mutex;

 private:
  // The id generator
  static IdGenerator id_generator_;
};

}  // namespace ksana_llm
