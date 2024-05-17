/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include "elementwise.h"

#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_mul.h"

#include "csrc/utils/ascend/common.h"

namespace llm_kernels {
namespace ascend {

void Add(const aclTensor* input, const aclTensor* other, const aclScalar* alpha, aclTensor** addOutput,
         aclrtStream& stream, void (*ws_func)(size_t, void**)) {
  uint64_t ws_size = 0ull;
  void* workspace = nullptr;
  aclOpExecutor* executor = nullptr;

  ACL_CHECK_RET(aclnnAddGetWorkspaceSize(input, other, alpha, *addOutput, &ws_size, &executor));
  ws_func(ws_size, &workspace);
  ACL_CHECK_RET(aclnnAdd(workspace, ws_size, executor, stream));
}

void Adds(const aclTensor* input, const aclScalar* scalar1, const aclScalar* scalar2, aclTensor** addOutput,
          aclrtStream& stream, void (*ws_func)(size_t, void**)) {
  uint64_t ws_size = 0ull;
  void* workspace = nullptr;
  aclOpExecutor* executor = nullptr;

  ACL_CHECK_RET(aclnnAddsGetWorkspaceSize(input, scalar1, scalar2, *addOutput, &ws_size, &executor));
  ws_func(ws_size, &workspace);
  ACL_CHECK_RET(aclnnAdds(workspace, ws_size, executor, stream));
}

void Mul(const aclTensor* mulInput1, const aclTensor* mulInput2, aclTensor** mulOutput, aclrtStream& stream,
         void (*ws_func)(size_t, void**)) {
  uint64_t ws_size = 0ull;
  void* workspace = nullptr;
  aclOpExecutor* executor = nullptr;

  ACL_CHECK_RET(aclnnMulGetWorkspaceSize(mulInput1, mulInput2, *mulOutput, &ws_size, &executor));
  ws_func(ws_size, &workspace);
  ACL_CHECK_RET(aclnnMul(workspace, ws_size, executor, stream));
}

}  // namespace ascend
}  // namespace llm_kernels
