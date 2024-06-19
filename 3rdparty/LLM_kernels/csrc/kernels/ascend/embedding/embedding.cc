/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include "embedding.h"
#include "aclrtlaunch_InvokeLookupEmbeddingFloatKernel.h"
#include "aclrtlaunch_InvokeLookupEmbeddingHalfKernel.h"
#include "csrc/kernels/ascend/embedding/embedding_kernel.h"
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

template <typename T>
void LookupFusedEmbedding(T* output_hidden_units, const T* embedding_table, const T* pos_table,
                          const int32_t* input_ids, const uint32_t total_seq_len, const int32_t start_step,
                          const int32_t batch_size, const uint32_t hidden_units, const size_t vocab_size,
                          const size_t vocab_id, aclrtStream stream, llm_kernels::utils::WorkSpaceFunc ws_func) {
  uint64_t tiling_size = sizeof(EmbeddingConfigTiling);
  void* tiling_device = nullptr;
  ws_func(tiling_size, &tiling_device);
  EmbeddingConfigTiling emb_config_tiling;
  emb_config_tiling.total_seq_len = total_seq_len;
  emb_config_tiling.hidden_units = hidden_units;
  emb_config_tiling.vocab_size = vocab_size;
  emb_config_tiling.vocab_id = vocab_id;
  emb_config_tiling.batch_size = batch_size;
  EmbeddingConfigTiling* buf = &emb_config_tiling;

  ACL_CHECK_RET(
      aclrtMemcpyAsync(tiling_device, tiling_size, (void*)buf, tiling_size, ACL_MEMCPY_HOST_TO_DEVICE, stream));
  ACL_CHECK_RET(aclrtSynchronizeStream(stream));

  if (std::is_same<T, aclFloat16>::value) {
    ACL_CHECK_RET(ACLRT_LAUNCH_KERNEL(InvokeLookupEmbeddingHalfKernel)(
        emb_config_tiling.total_seq_len, stream, (uint8_t*)input_ids, (uint8_t*)embedding_table,
        (uint8_t*)output_hidden_units, (uint8_t*)tiling_device));
  } else if (std::is_same<T, float>::value) {
    ACL_CHECK_RET(ACLRT_LAUNCH_KERNEL(InvokeLookupEmbeddingFloatKernel)(
        emb_config_tiling.total_seq_len, stream, (uint8_t*)input_ids, (uint8_t*)embedding_table,
        (uint8_t*)output_hidden_units, (uint8_t*)tiling_device));
  } else {
    throw std::invalid_argument("Invalid embedding lookup type, only support float16 or float32.");
  }
}

template void LookupFusedEmbedding(aclFloat16* output_hidden_units, const aclFloat16* embedding_table,
                                   const aclFloat16* pos_table, const int32_t* input_ids, const uint32_t total_seq_len,
                                   const int32_t start_step, const int32_t batch_size, const uint32_t hidden_units,
                                   const size_t vocab_size, const size_t vocab_id, aclrtStream stream,
                                   llm_kernels::utils::WorkSpaceFunc ws_func);
template void LookupFusedEmbedding(float* output_hidden_units, const float* embedding_table, const float* pos_table,
                                   const int32_t* input_ids, const uint32_t total_seq_len, const int32_t start_step,
                                   const int32_t batch_size, const uint32_t hidden_units, const size_t vocab_size,
                                   const size_t vocab_id, aclrtStream stream,
                                   llm_kernels::utils::WorkSpaceFunc ws_func);

}  // namespace ascend
}  // namespace llm_kernels
