/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * E8M0 scale processing for humming WGMMA layout.
 * Performs: transpose [E, N, K/gs] -> [E, K/gs, N] then index permutation.
 * Uses libtorch for tensor operations (matches humming's prepare_humming_weight_scale).
 */

#include <torch/torch.h>
#include <cuda_runtime.h>

#include "humming_preprocess.h"

namespace llm_kernels {
namespace nvidia {
namespace humming {

void process_scales(cudaStream_t stream, const void* input, void* output,
                    int64_t num_experts, int64_t size_n, int64_t size_k,
                    int64_t group_size, ComputeDtype compute_dtype) {
  // Input scale shape: [E, N, K/group_size] uint8 (E8M0 format)
  // Output: [E, K/group_size, N] uint8 with permuted indices within groups of 8

  int64_t num_groups_k = size_k / group_size;
  auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);

  // Wrap input as tensor (no copy)
  auto scale_tensor = torch::from_blob(const_cast<void*>(input),
                                       {num_experts, size_n, num_groups_k}, options);

  // Step 1: Transpose [E, N, K/gs] -> [E, K/gs, N]
  auto transposed = scale_tensor.transpose(1, 2).contiguous();

  // Step 2: Permutation — matches humming's prepare_humming_weight_scale(to_apply_on_c=false)
  // Base pattern: perm = [0, 8, 16, 24, 32, 40, 48, 56]
  // count = number of elements < 8 = 1
  // Expanded: for i in range(8/1=8): perm_new += [v + 1*i for v in base_perm]
  // Result: 64-element permutation
  std::vector<int64_t> base_perm = {0, 8, 16, 24, 32, 40, 48, 56};
  int64_t count = 0;
  for (auto v : base_perm) {
    if (v < 8) count++;
  }
  // count = 1

  std::vector<int64_t> perm_new;
  for (int64_t i = 0; i < 8 / count; i++) {
    for (auto v : base_perm) {
      perm_new.push_back(v + count * i);
    }
  }
  // perm_new = [0,8,16,24,32,40,48,56, 1,9,17,25,33,41,49,57, ..., 7,15,23,31,39,47,55,63]

  auto perm_tensor = torch::tensor(perm_new,
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));

  auto orig_shape = transposed.sizes().vec();
  int64_t perm_size = static_cast<int64_t>(perm_new.size());

  // Reshape to [..., perm_size], apply permutation on last dim, reshape back
  auto permuted = transposed.reshape({-1, perm_size})
                            .index_select(1, perm_tensor)
                            .contiguous()
                            .reshape(orig_shape);

  // Copy result to output buffer
  cudaMemcpyAsync(output, permuted.data_ptr(), permuted.nbytes(),
                  cudaMemcpyDeviceToDevice, stream);
}

}  // namespace humming
}  // namespace nvidia
}  // namespace llm_kernels
