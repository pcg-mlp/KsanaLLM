/* Copyright 2023 Tencent Inc.  All rights reserved.
   modify from https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/flash_api.cpp

==============================================================================*/
#pragma once

// Include these 2 headers instead of torch/extension.h since we don't need all of the torch headers.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/nn/functional.h>
#include <torch/python.h>

// NOTE(karlluo): this function is wrapped in flash_attn_2_cuda.cpython-39-x86_64-linux-gnu.so
std::vector<at::Tensor> mha_varlen_fwd(
    at::Tensor &q,                    // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    const at::Tensor &k,              // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    const at::Tensor &v,              // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    c10::optional<at::Tensor> &out_,  // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
    const at::Tensor &cu_seqlens_q,   // b+1
    const at::Tensor &cu_seqlens_k,   // b+1
    c10::optional<at::Tensor>
        &seqused_k,  // b. If given, only this many elements of each batch element's keys are used.
    c10::optional<at::Tensor> &alibi_slopes_,  // num_heads or b x num_heads
    int max_seqlen_q, const int max_seqlen_k, const float p_dropout, const float softmax_scale, const bool zero_tensors,
    bool is_causal, int window_size_left, int window_size_right, const bool return_softmax,
    c10::optional<at::Generator> gen_);