/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include "transpose.h"

#include <vector>

#include "aclnnop/aclnn_copy.h"
#include "csrc/utils/ascend/common.h"

namespace llm_kernels {
namespace ascend {

void Copy(const aclTensor* input, aclTensor** output, aclrtStream& stream, void (*ws_func)(size_t, void**)) {
  uint64_t ws_size = 0ull;
  void* workspace = nullptr;
  aclOpExecutor* executor = nullptr;
  ACL_CHECK_RET(aclnnInplaceCopyGetWorkspaceSize(*output, input, &ws_size, &executor));
  ws_func(ws_size, &workspace);
  ACL_CHECK_RET(aclnnInplaceCopy(workspace, ws_size, executor, stream));
  ACL_CHECK_RET(aclrtSynchronizeStream(stream));
}

void Transpose(const aclTensor* input, aclTensor** output, aclrtStream& stream, void (*ws_func)(size_t, void**)) {
  Copy(input, output, stream, ws_func);
}

}  // namespace ascend
}  // namespace llm_kernels
