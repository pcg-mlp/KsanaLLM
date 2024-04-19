/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include "slice.h"

#include "aclnnop/aclnn_slice.h"
#include "csrc/utils/ascend/common.h"

namespace llm_kernels {
namespace ascend {

void Slice(const aclTensor* input, const int sliceDim, const int sliceStart, const int sliceEnd, const int sliceStep,
           aclTensor** output, aclrtStream& stream, void (*ws_func)(size_t, void**)) {
  uint64_t ws_size = 0ull;
  void* workspace = nullptr;
  aclOpExecutor* executor = nullptr;
  ACL_CHECK_RET(
      aclnnSliceGetWorkspaceSize(input, sliceDim, sliceStart, sliceEnd, sliceStep, *output, &ws_size, &executor));
  ws_func(ws_size, &workspace);
  ACL_CHECK_RET(aclnnSlice(workspace, ws_size, executor, stream));
  ACL_CHECK_RET(aclrtSynchronizeStream(stream));
}

}  // namespace ascend
}  // namespace llm_kernels
