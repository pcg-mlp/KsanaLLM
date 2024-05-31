/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstddef>
#include <vector>
#include "ksana_llm/runtime/forward_request.h"
#include "ksana_llm/utils/tensor.h"

namespace ksana_llm {

// Copy model output to samplers.
class ModelOutput {
 public:
  ModelOutput(size_t max_batch_size, size_t vocab_size, int rank, std::shared_ptr<Context> context);
  ~ModelOutput();

  void CopyToLogistBuffer(const size_t batch_size, std::vector<ForwardRequest>& forward_reqs,
                          std::vector<Tensor>& logits_float);

 public:
  // Whether the compute is ready for output.
  Event compute_ready_event;

  // Store logist result, shape: [max_batch_size, vocab_size], dtype: float
  Tensor logits_tensor;

 private:
  int rank_;
  std::shared_ptr<Context> context_;

  size_t max_batch_size_;
  size_t vocab_size_;
};

}  // namespace ksana_llm
