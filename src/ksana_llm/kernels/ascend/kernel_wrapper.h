/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/utils/ascend/acl_utils.h"
#include "ksana_llm/utils/memory_utils.h"

namespace ksana_llm {

// Invoke the lookup embedding.
void LookupEmbedding(const aclTensor* input_ids, const aclTensor* embedding_table, const aclTensor* position_table,
                     aclTensor* output, aclrtStream stream, WorkSpaceFunc ws_func);

}  // namespace ksana_llm
