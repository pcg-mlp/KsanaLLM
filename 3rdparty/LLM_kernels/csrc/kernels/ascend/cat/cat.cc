/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include "cat.h"

#include "aclnnop/aclnn_cat.h"
#include "csrc/utils/ascend/common.h"

namespace llm_kernels {
namespace ascend {

void Cat(std::vector<const aclTensor*>& inputs, const int catDim, aclTensor** output, aclrtStream& stream,
         void (*ws_func)(size_t, void**)) {
  uint64_t ws_size = 0ull;
  void* workspace = nullptr;
  aclOpExecutor* executor = nullptr;

  aclTensorList* tensorList = aclCreateTensorList(inputs.data(), inputs.size());
  ACL_CHECK_RET(aclnnCatGetWorkspaceSize(tensorList, catDim, *output, &ws_size, &executor));
  ws_func(ws_size, &workspace);
  ACL_CHECK_RET(aclnnCat(workspace, ws_size, executor, stream));
  ACL_CHECK_RET(aclrtSynchronizeStream(stream));
}

}  // namespace ascend
}  // namespace llm_kernels
