/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "kernels/permute.h"

namespace ksana_llm {

Status Permute(Tensor& input_tensor, Tensor& output_tensor, const std::vector<size_t>& permutation, Stream& stream,
               void* device_ctx, void* workspace_ptr) {
  return Status(RET_UNDEFINED_REFERENCE, "Permute not supported.");
}

}  // namespace ksana_llm
