/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <vector>

#include "acl/acl.h"
#include "acl/acl_op_compiler.h"
#include "aclnn/acl_meta.h"

#include "csrc/kernels/ascend/permute/permute_tiling.h"

namespace llm_kernels {
namespace ascend {

// NOTE(karlluo): perform the same process as Pytorch, just change shape and stride
void Permute(const aclTensor* permute_input, void** permute_input_tensor_addr_ptr, aclTensor** permute_output,
             const std::vector<int64_t>& dims, aclrtStream& stream, void (*ws_func)(size_t, void**));

template <typename T>
class PermuteKernelWrapper {
 public:
  PermuteKernelWrapper();
  ~PermuteKernelWrapper();

  // Permute.
  void Forward(void* output, void* input, const std::vector<uint64_t>& shape, const std::vector<uint64_t> new_indexes,
               aclrtStream stream);

 private:
  // Copy the tiling data from host to global memory.
  void CopyTilingToDevice(aclrtStream stream);

  // The tiling data for current request.
  PermuteTilingData tiling_data_;

  // The tiling buffer on global memory
  void* tiling_buffer_gm_;

  // The size of tiling data.
  size_t tiling_size_;

  // The worksapce.
  void* workspace_gm_;
};

}  // namespace ascend
}  // namespace llm_kernels
