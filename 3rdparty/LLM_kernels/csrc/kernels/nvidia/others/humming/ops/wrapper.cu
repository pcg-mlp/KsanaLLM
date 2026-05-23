/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

#include "csrc/kernels/nvidia/others/humming/launcher/kernel_meta.h"
#include "csrc/kernels/nvidia/others/humming/ops/wrapper.h"

#include <cuda_runtime.h>
#include <algorithm>

namespace llm_kernels {
namespace nvidia {
namespace humming {

int64_t get_locks_size(int64_t size_m, int64_t size_n, int64_t size_k, ComputeDtype compute_dtype) {
  int dev = -1;
  cudaGetDevice(&dev);
  int sm_count = -1;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);

  int num_ctas = (compute_dtype == ComputeDtype::FP8) ? 3 : 2;
  return static_cast<int64_t>(sm_count) * num_ctas * 4;  // 4x over-provision
}

int64_t get_workspace_size(int64_t size_m, int64_t size_n, int64_t size_k, int64_t num_experts, int64_t top_k,
                           ComputeDtype compute_dtype) {
  int dev = -1;
  cudaGetDevice(&dev);
  int sm_count = -1;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);

  int num_ctas = (compute_dtype == ComputeDtype::FP8) ? 3 : 2;
  int64_t total_ctas = static_cast<int64_t>(sm_count) * num_ctas;

  // tensor_map_buffer: total_ctas * 128 bytes (device-side TMA storage)
  int64_t tma_buffer_size = total_ctas * 128;

  // Total workspace (256-byte aligned)
  int64_t workspace = tma_buffer_size;
  workspace = (workspace + 255) & ~255;
  return workspace;
}

}  // namespace humming
}  // namespace nvidia
}  // namespace llm_kernels
