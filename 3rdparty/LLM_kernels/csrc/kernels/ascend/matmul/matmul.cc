/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include "matmul.h"

#include "aclnnop/aclnn_matmul.h"
#include "csrc/utils/ascend/common.h"

namespace llm_kernels {
namespace ascend {

aclError MatMul(const aclTensor* input, const aclTensor* weight, const int8_t matmulCubeMathType, aclTensor** output,
                aclrtStream& stream, void (*ws_func)(size_t, void**)) {
  uint64_t ws_size = 0ull;
  void* workspace = nullptr;
  aclOpExecutor* executor = nullptr;

  ACL_CHECK_RET(aclnnMatmulGetWorkspaceSize(input, weight, *output, matmulCubeMathType, &ws_size, &executor));
  ws_func(ws_size, &workspace);
  ACL_CHECK_RET(aclnnMatmul(workspace, ws_size, executor, stream));
  ACL_CHECK_RET(aclrtSynchronizeStream(stream));
  return ACL_SUCCESS;
}

}  // namespace ascend
}  // namespace llm_kernels
