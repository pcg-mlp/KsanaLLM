/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/models/base/base_weight.h"
#include "ksana_llm/runtime/cuda_graph_runner.h"
#include "ksana_llm/runtime/forward_request.h"
#include "ksana_llm/utils/context.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/tensor.h"

namespace ksana_llm {

class BaseModel {
 public:
  // Disable a default constructor
  BaseModel() {}

  ~BaseModel();

  // The output logits pointer on device, used by sampler to avoid memory copy.
  virtual float* GetLogitsPtr() = 0;

  // The prefill stage.
  virtual Status ContextDecode(std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                               std::vector<ForwardRequest>& forward_reqs) = 0;

  // The decode stage.
  virtual Status Decode(std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                        std::vector<ForwardRequest>& forward_reqs) = 0;

  // Implement this method if cuda graph is used.
  virtual Status WarmUpCudaGraph() { return Status(); }

 protected:
  std::shared_ptr<Context> context_{nullptr};

  int rank_{0};
  bool use_custom_all_reduce_{true};
  size_t total_buffer_size_{0ul};

  // Store logist result, shape: [max_batch_size, vocab_size], dtype: float
  Tensor logits_tensor_;

  // Record all buffer used
  std::vector<Tensor*> buffer_tensor_heap_;

  // Whether cuda graph is enabled.
  bool enable_cuda_graph_ = true;

  // The cuda graph runner.
  CudaGraphRunner cuda_graph_runner_;

 protected:
  // Create Buffer tensor
  Status CreateBufferTensor(Tensor& buf_tensor, const std::vector<size_t> shape, const DataType dtype,
                            const MemoryDevice memory_device = MEMORY_GPU);

  // Release all buffer tensors
  Status ReleaseBufferTensors();

  // Log total buffer tensors memory used
  const size_t GetBufferTensorsMemoryUsed();
};

}  // namespace ksana_llm
