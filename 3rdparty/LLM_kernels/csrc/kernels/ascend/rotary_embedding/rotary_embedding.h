/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <unordered_map>

#include "acl/acl.h"
#include "acl/acl_op_compiler.h"
#include "aclnn/acl_meta.h"

namespace llm_kernels {
namespace ascend {

class RotaryEmbeddingACL {
 public:
  void Init(const int max_position_embeddings, const int head_dims, const float rope_theta,
            const float rope_scaling_factor, aclDataType dtype, aclrtStream& stream, void (*ws_func)(size_t, void**));

  void Forward(const aclTensor* input, const aclTensor* pos_index, aclTensor** output, aclrtStream& stream,
               void (*ws_func)(size_t, void**) = nullptr, void* workspace_buf_ptr = nullptr);
  ~RotaryEmbeddingACL();

 private:
  void InitSinCos(const int max_position_embeddings, const int head_dims, const float rope_theta, aclDataType dtype,
                  aclrtStream& stream, void (*ws_func)(size_t, void**));
  void RotarySplit(const aclTensor* input, const int bs, const int64_t seq_len, const int num_heads, void** maxDev_a,
                   void** maxDev_b, void** maxDev_c, aclTensor** catOutput, aclrtStream& stream,
                   void (*ws_func)(size_t, void**));

 private:
  int max_position_embeddings_;
  int head_dims_;
  float rope_theta_;
  float rope_scaling_factor_;
  aclDataType dtype_;

  void* sin_dev_ = nullptr;
  void* cos_dev_ = nullptr;
  aclTensor* sin_ = nullptr;
  aclTensor* cos_ = nullptr;
};

}  // namespace ascend
}  // namespace llm_kernels
