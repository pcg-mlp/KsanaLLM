/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include <optional>

#include "numerous_llm/utils/kernel_registry.h"
#include "numerous_llm/utils/tensor.h"

namespace numerous_llm {

void paged_attention(
    Tensor& out,                       // [num_seqs, num_heads, head_size]
    const Tensor& query,               // [num_seqs, num_heads, head_size]
    std::vector<Tensor>& key_cache,    // num_seqs,[seq_blocks,num_kv_heads,head_size/x,block_size,x],x=16/sizeof(T)
    std::vector<Tensor>& value_cache,  // num_seqs,[seq_blocks, num_kv_heads, head_size, block_size]
    const Tensor& context_lens,        // [num_seqs]
    int max_context_len, cudaStream_t stream, void* workspace, size_t work_size,
    const std::optional<Tensor>& alibi_slopes);

void paged_attention(
    const int layer_index,
    Tensor& out,                       // [num_seqs, num_heads, head_size]
    const Tensor& query,                     // [num_seqs, num_heads, head_size]
    // kv_list [layer_num, num_seqs] k [num_seqs,seq_blocks,num_kv_heads,head_size/x,block_size,x],x=16/sizeof(T) v [num_seqs,seq_blocks, num_kv_heads, head_size, block_size]
    Tensor& kv_list,
    const int64_t block_size,
    const Tensor& context_lens,              // [num_seqs]
    int max_context_len, cudaStream_t stream, Tensor workspace,
    const std::optional<Tensor>& alibi_slopes){
        //.GetPtr(), input_tensors[2].GetTotalBytes()
    }


}  // namespace numerous_llm
