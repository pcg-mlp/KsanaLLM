/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include "gather.h"

#include "aclnnop/aclnn_gather_v2.h"
#include "csrc/utils/ascend/common.h"

namespace llm_kernels {
namespace ascend {

void Gather(const aclTensor* gatherInput, const int gatherDim, const aclTensor* gatherIndex, aclTensor** gatherOutput,
            aclrtStream& stream, void (*ws_func)(size_t, void**)) {
  uint64_t ws_size = 0ull;
  void* workspace = nullptr;
  aclOpExecutor* executor = nullptr;

  ACL_CHECK_RET(aclnnGatherV2GetWorkspaceSize(gatherInput, gatherDim, gatherIndex, *gatherOutput, &ws_size, &executor));
  ws_func(ws_size, &workspace);
  ACL_CHECK_RET(aclnnGatherV2(workspace, ws_size, executor, stream));
}

}  // namespace ascend
}  // namespace llm_kernels
