/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include "pointwise.h"

#include "aclnnop/aclnn_cast.h"
#include "aclnnop/aclnn_div.h"
#include "aclnnop/aclnn_mean.h"
#include "aclnnop/aclnn_neg.h"
#include "aclnnop/aclnn_pow.h"
#include "aclnnop/aclnn_sqrt.h"

#include "csrc/utils/ascend/common.h"

namespace llm_kernels {
namespace ascend {

void Cast(const aclTensor* castInput, const aclDataType castToType, aclTensor** castOutput, aclrtStream& stream,
          void (*ws_func)(size_t, void**)) {
  uint64_t ws_size = 0ull;
  void* workspace = nullptr;
  aclOpExecutor* executor = nullptr;

  ACL_CHECK_RET(aclnnCastGetWorkspaceSize(castInput, castToType, *castOutput, &ws_size, &executor));
  ws_func(ws_size, &workspace);
  ACL_CHECK_RET(aclnnCast(workspace, ws_size, executor, stream));
  ACL_CHECK_RET(aclrtSynchronizeStream(stream));
}

void Neg(const aclTensor* input, aclTensor** output, aclrtStream& stream, void (*ws_func)(size_t, void**)) {
  uint64_t ws_size = 0ull;
  void* workspace = nullptr;
  aclOpExecutor* executor = nullptr;

  ACL_CHECK_RET(aclnnNegGetWorkspaceSize(input, *output, &ws_size, &executor));
  ws_func(ws_size, &workspace);
  ACL_CHECK_RET(aclnnNeg(workspace, ws_size, executor, stream));
}

}  // namespace ascend
}  // namespace llm_kernels
