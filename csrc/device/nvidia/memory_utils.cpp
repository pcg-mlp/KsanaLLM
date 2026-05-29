/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "utils/memory_utils.h"

#include <cuda_runtime.h>

#include <cstdint>

#include "device/device_utils.h"
#include "utils/tensor.h"
namespace ksana_llm {

// Defined in memory_utils.cu in this directory.
void LaunchBatchMemoryD2DKernel(const uint64_t* src_addr_list, const uint64_t* dst_addr_list, size_t src_offset,
                                size_t dst_offset, size_t block_size, int N, cudaStream_t stream);

void BatchMemoryD2D(const Tensor& src_addr_list, const Tensor& dst_addr_list, size_t src_offset, size_t dst_offset,
                    size_t block_size, Stream& stream) {
  const int N = static_cast<int>(src_addr_list.shape[0]);
  if (N <= 0 || block_size == 0) {
    return;
  }
  LaunchBatchMemoryD2DKernel(src_addr_list.GetPtr<uint64_t>(), dst_addr_list.GetPtr<uint64_t>(), src_offset, dst_offset,
                             block_size, N, stream.Get());
}

}  // namespace ksana_llm
