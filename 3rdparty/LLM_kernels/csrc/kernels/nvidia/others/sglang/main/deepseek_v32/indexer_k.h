/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace llm_kernels {
namespace nvidia {

// Fused LayerNorm + RoPE (leading 64 dims) + FP8 quantization + paged store.
//
// k_input:              [batch, 128] bf16, row stride = k_input_stride_batch (elements).
// weight, bias:         [128] bf16 (LayerNorm gamma/beta).
// freqs_cis:            [max_pos, 64] bf16, split [cos|sin].
// positions:            [batch] int64.
// cache:                paged FP8 KV cache, uint8 base for this layer's indexer-K region.
//                       Per page: [value: 128*page_size][scale: 4*page_size] = 132*page_size bytes.
//                       Page-to-page stride is the full KV block size, passed via page_stride.
// indices:              [batch] int32 slot ids. page = idx >> page_bits,
//                       offset = idx & ((1<<page_bits)-1); -1 skips the store.
// page_bits:            log2(page_size). Supported: {1, 6, 8} → page_size {2, 64, 256}
// page_stride:          bytes between pages in cache (= attn_backend_config.block_size).
// is_neox:              true  → NeoX split-half (first half real, second half imag);
//                       false → GPT-J interleaved [real, imag, real, imag, ...].
void fused_k_indexer_norm_rope_store(const void* k_input, void* cache, const void* indices, const void* weight,
                                     const void* bias, const void* freqs_cis, const void* positions,
                                     int64_t k_input_stride_batch, uint32_t batch_size, uint32_t page_bits,
                                     int64_t page_stride, float eps, cudaStream_t stream, bool is_neox = false);

}  // namespace nvidia
}  // namespace llm_kernels
