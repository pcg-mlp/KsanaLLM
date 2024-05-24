/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#pragma once

#include "acl/acl.h"
#include "acl/acl_op_compiler.h"
#include "aclnn/acl_meta.h"

#include "csrc/kernels/ascend/slice/slice_tiling.h"

namespace llm_kernels {
namespace ascend {

void Slice(const aclTensor* input, const int sliceDim, const int sliceStart, const int sliceEnd, const int sliceStep,
           aclTensor** output, aclrtStream& stream, void (*ws_func)(size_t, void**));

template <typename T>
class Slice2 {
 public:
  Slice2();
  ~Slice2();

  // Slice fragments from input, and concat it to output.
  void Forward(void* output, void* input, int start, int length, int step, int times, aclrtStream stream);

 private:
  // Copy the tiling data from host to global memory.
  void CopyTilingToDevice(aclrtStream stream);

  // The tiling data for current request.
  SliceTilingData tiling_data_;

  // The tiling buffer on global memory
  void* tiling_buffer_gm_;

  // The size of tiling data.
  size_t tiling_size_;

  // The worksapce.
  void* workspace_gm_;
};

}  // namespace ascend
}  // namespace llm_kernels
