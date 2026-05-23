/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

#include "marlin_moe_wna16_wrapper.h"

#include <algorithm>

namespace llm_kernels {
namespace nvidia {
namespace marlin_moe_wna16 {

int64_t get_workspace_size(int64_t size_n, int64_t sorted_token_ids_len, int64_t moe_block_size) {
  int sms = -1;
  int dev = -1;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);

  int64_t max_n_tiles = size_n / kMinThreadN;
  int64_t min_workspace_size = std::min(
      max_n_tiles * (sorted_token_ids_len / moe_block_size),
      (int64_t)sms * 4);
  return min_workspace_size;
}

int64_t get_c_tmp_size(int64_t size_n, int64_t sorted_token_ids_len,
                       int64_t moe_block_size, ::marlin_moe_wna16::ScalarTypeId b_q_type_id) {
  int sms = -1;
  int dev = -1;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);

  int64_t max_c_tmp_size = std::min(
      size_n * sorted_token_ids_len,
      (int64_t)sms * 4 * moe_block_size * kMaxThreadN);
  if (moe_block_size == 8) max_c_tmp_size *= 2;
  return max_c_tmp_size;
}

int64_t get_a_tmp_size(int64_t size_m, int64_t size_k, int64_t top_k) {
  return size_m * top_k * size_k;
}

}  // namespace marlin_moe_wna16
}  // namespace nvidia
}  // namespace llm_kernels
