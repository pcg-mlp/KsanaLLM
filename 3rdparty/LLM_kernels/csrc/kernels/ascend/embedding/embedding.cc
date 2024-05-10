/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include "embedding.h"
#include "csrc/utils/ascend/common.h"

namespace llm_kernels {
namespace ascend {

void LookupEmbedding(const aclTensor* input_ids, const aclTensor* embedding_table, const aclTensor* position_table,
                     aclTensor* output, aclrtStream stream, llm_kernels::utils::WorkSpaceFunc ws_func) {
  uint64_t ws_size = 0ull;
  void* workspace = nullptr;
  aclOpExecutor* executor = nullptr;
  ACL_CHECK_RET(aclnnEmbeddingGetWorkspaceSize(embedding_table, input_ids, output, &ws_size, &executor));
  ws_func(ws_size, &workspace);
  ACL_CHECK_RET(aclnnEmbedding(workspace, ws_size, executor, stream));
}

}  // namespace ascend
}  // namespace llm_kernels
