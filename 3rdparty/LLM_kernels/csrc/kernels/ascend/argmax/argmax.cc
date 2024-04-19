/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include "argmax.h"

#include "aclnnop/aclnn_argmax.h"
#include "csrc/utils/ascend/common.h"

namespace llm_kernels {
namespace ascend {

void ArgMax(const aclTensor* argMaxInput, const int64_t argMaxDim, const bool argMaxKeepdim, aclTensor** argMaxOutput,
            aclrtStream& stream, void (*ws_func)(size_t, void**)) {
  uint64_t ws_size = 0ull;
  void* workspace = nullptr;
  aclOpExecutor* executor = nullptr;
  ACL_CHECK_RET(aclnnArgMaxGetWorkspaceSize(argMaxInput, argMaxDim, argMaxKeepdim, *argMaxOutput, &ws_size, &executor));
  ws_func(ws_size, &workspace);
  ACL_CHECK_RET(aclnnArgMax(workspace, ws_size, executor, stream));
  ACL_CHECK_RET(aclrtSynchronizeStream(stream));
}

}  // namespace ascend
}  // namespace llm_kernels
