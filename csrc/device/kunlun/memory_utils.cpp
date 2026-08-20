/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
// Kunlun XPU implementation of memory helpers declared in utils/memory_utils.h
// that have no generic (device-agnostic) definition.

#include <cstdint>

#include "utils/logger.h"
#include "utils/memory_utils.h"
#include "utils/tensor.h"

namespace ksana_llm {

// BatchMemoryD2D is only exercised by the DeepSeek-V4 attention path, which is
// not supported on Kunlun (see kunlun/unsupported_layers_stub.cpp). Provide a
// definition so libtorch_serving.so links, honouring the empty-list no-op
// contract and throwing only if it is actually invoked with work to do.
void BatchMemoryD2D(const Tensor& src_addr_list, const Tensor& dst_addr_list, size_t src_offset, size_t dst_offset,
                    size_t block_size, Stream& stream) {
  const int N = src_addr_list.shape.empty() ? 0 : static_cast<int>(src_addr_list.shape[0]);
  if (N <= 0 || block_size == 0) {
    return;
  }
  KLLM_THROW("BatchMemoryD2D not supported on Kunlun.");
}

}  // namespace ksana_llm
