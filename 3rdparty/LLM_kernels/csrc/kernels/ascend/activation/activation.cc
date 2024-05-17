/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include "activation.h"

#include "aclnnop/aclnn_silu.h"
#include "csrc/utils/ascend/common.h"

namespace llm_kernels {
namespace ascend {

void Silu(const aclTensor* input, aclTensor** output, aclrtStream& stream, void (*ws_func)(size_t, void**)) {
  uint64_t ws_size = 0ull;
  void* workspace = nullptr;
  aclOpExecutor* executor = nullptr;
  ACL_CHECK_RET(aclnnSiluGetWorkspaceSize(input, *output, &ws_size, &executor));
  ws_func(ws_size, &workspace);
  ACL_CHECK_RET(aclnnSilu(workspace, ws_size, executor, stream));
}

}  // namespace ascend
}  // namespace llm_kernels
