/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "layers/base_layer.h"

namespace ksana_llm {

class CastLayer : public BaseLayer {
 public:
  // Typed interface: cast input to float32 and write at byte_offset into output.
  // vocab_size and vocab_size_pad are passed to the kernel for row padding.
  Status Forward(const Tensor* input, size_t byte_offset, size_t vocab_size, size_t vocab_size_pad, Tensor* output,
                 const LayerExecContext& exec_ctx = {});

 private:
  template <typename SRC_DTYPE>
  Status ForwardT(const Tensor* input, size_t byte_offset, size_t vocab_size, size_t vocab_size_pad, Tensor* output);
};

}  // namespace ksana_llm
