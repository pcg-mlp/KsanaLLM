/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#ifdef ENABLE_CUDA
#  include <optional>

#  include "csrc/kernels/nvidia/rotary_embedding/rotary_embedding.h"
#  include "csrc/kernels/nvidia/alibi/alibi.h"
#endif

#ifdef ENABLE_ACL
#  include "csrc/kernels/ascend/attention/attention.h"
#endif

#include "ksana_llm/layers/base_layer.h"

namespace ksana_llm {

template <typename T>
class AttentionLayer : public BaseLayer {
  public:
    virtual Status Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context, int rank) override;

    virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) = 0;

  protected:
    int layer_index_;
    int block_size_;
    int block_token_num_;
    int num_heads_;
    int num_kv_heads_;
    int head_size_;
    int stride_size_;
    int tensor_para_size_;
    bool is_causal_{true};
#ifdef ENABLE_CUDA
    llm_kernels::nvidia::RotaryEmbeddingCuda<T> rotary_embedding_cuda_;
    std::optional<void*> alibi_slopes_ = {};
#endif

#ifdef ENABLE_ACL
    // NOTE(karlluo): only need by ascend
    int workspace_block_id_{-1};
    size_t workspace_size_{0ul};
    // The attention implementation for ascend device.
    std::shared_ptr<llm_kernels::ascend::FlashAttentionACL> ascend_flash_attn_= nullptr;

    void PrepareWorkspaceBuffer(const size_t workspace_needed, void* workspace_buf_ptr);
#endif
};

}  // namespace ksana_llm
