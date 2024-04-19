/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "acl/acl.h"
#include "acl/acl_op_compiler.h"
#include "aclnn/acl_meta.h"

#include "csrc/kernels/ascend/rotary_embedding/rotary_embedding.h"

namespace llm_kernels {
namespace ascend {

class FlashAttentionACL {
 public:
  void Init(const int max_position_embeddings, const int head_dims, const int q_heads, const int kv_heads,
            const float rope_theta, const float rope_scaling_factor, aclDataType dtype, aclrtStream& stream,
            void (*ws_func)(size_t, void**));
  ~FlashAttentionACL();

  void Forward(const aclTensor* matmulQKVOutput, const int64_t token_pos, void** key_cache, void** val_cache,
               std::vector<void*>& tmp_buffers, aclTensor** output, const bool is_context_stage, aclrtStream& stream,
               void (*ws_func)(size_t, void**));

 private:
  void InitAttnMask(int max_tokens_num, aclDataType dtype);

  void PrepareRopeIndex(const int bs, const int seq_len, const int64_t token_pos, const bool is_context_stage,
                        void** rope_index_dev, aclTensor** rope_index);

  void GetSliceAndPermute(const aclTensor* input, int one_of_qkv_index, const std::vector<int64_t>& output_shape,
                          void** tmp_buffer_dev, void** output_dev, aclTensor** output, aclrtStream& stream,
                          void (*ws_func)(size_t, void**));

  void PromptFlashAttention(const aclTensor* query, const aclTensor* key, const aclTensor* value, aclTensor** output,
                            aclrtStream& stream, void (*ws_func)(size_t, void**));

  void IncFlashAttention(const aclTensor* query, const aclTensor* key, const aclTensor* value, aclTensor** output,
                         aclrtStream& stream, void (*ws_func)(size_t, void**));

 private:
  std::unique_ptr<RotaryEmbeddingACL> rope_ptr_;
  int q_heads_;
  int kv_heads_;
  int head_dims_;
  aclDataType dtype_;

  void* attn_mask_dev_ = nullptr;
  aclTensor* attn_mask_ = nullptr;
};

}  // namespace ascend
}  // namespace llm_kernels
