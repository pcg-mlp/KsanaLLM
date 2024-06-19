/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#pragma once

#include "acl/acl.h"
#include "acl/acl_op_compiler.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_embedding.h"

#include "csrc/utils/ascend/common.h"

namespace llm_kernels {
namespace ascend {

void LookupEmbedding(const aclTensor* input_ids, const aclTensor* embedding_table, const aclTensor* position_table,
                     aclTensor* output, aclrtStream stream, llm_kernels::utils::WorkSpaceFunc ws_func);

template <typename T>
void LookupFusedEmbedding(T* output_hidden_units, const T* embedding_table, const T* pos_table,
                          const int32_t* input_ids, const uint32_t total_seq_len, const int32_t start_step,
                          const int32_t batch_size, const uint32_t hidden_units, const size_t vocab_size,
                          const size_t vocab_id, aclrtStream stream, llm_kernels::utils::WorkSpaceFunc ws_func);

}  // namespace ascend
}  // namespace llm_kernels
