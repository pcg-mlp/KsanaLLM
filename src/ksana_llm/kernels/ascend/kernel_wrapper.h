/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "3rdparty/LLM_kernels/csrc/kernels/ascend/atb_plugin_operations/argmax_operation.h"
#include "3rdparty/LLM_kernels/csrc/kernels/ascend/atb_plugin_operations/cast_operation.h"
#include "3rdparty/LLM_kernels/csrc/utils/ascend/atb_executor.h"
#include "3rdparty/LLM_kernels/csrc/utils/ascend/common.h"
#include "ksana_llm/utils/ascend/acl_utils.h"
#include "ksana_llm/utils/device_utils.h"
#include "ksana_llm/utils/memory_utils.h"
#include "ksana_llm/utils/tensor.h"

namespace ksana_llm {

// NOTE(karlluo): There is a bug "ATB plugin ops with ACLNN cant not run in graph mode".
template <typename T>
class ArgMaxATBExecutor {
 public:
  ArgMaxATBExecutor() {}
  ~ArgMaxATBExecutor() {}

  Status Init(const int rank, const size_t max_batch_size);

  Status Run(const int rank, const T* input, const int32_t batch_size, const int32_t vocab_size, uint32_t* result,
             Stream& stream);

 private:
  llm_kernels::utils::ATBOperationExecutor atb_argmax_op_executor_;
  llm_kernels::utils::ATBOperationExecutor atb_cast_op_executor_;

  Tensor internal_tensor_;
};

// Invoke the lookup embedding.
void LookupEmbedding(const aclTensor* input_ids, const aclTensor* embedding_table, const aclTensor* position_table,
                     aclTensor* output, aclrtStream stream, WorkSpaceFunc ws_func);

}  // namespace ksana_llm
