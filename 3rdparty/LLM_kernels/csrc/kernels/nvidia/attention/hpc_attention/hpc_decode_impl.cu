/* Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * hpc-ops decode attention implementation (smallm_splitk_dim128).
 * Source: https://github.com/Tencent/hpc-ops (submodule pinned to 8d300e1)
 * ==============================================================================*/

#include "csrc/kernels/nvidia/attention/hpc_attention/hpc_attention.h"

#include <cuda_runtime_api.h>

#include <cstdlib>

#include "src/attention/decode/smallm_dim128.h"

namespace llm_kernels {
namespace nvidia {

namespace {

struct DecodeScratch {
  void* lse = nullptr;
  void* split_out = nullptr;
  void* split_flag = nullptr;
  size_t lse_bytes = 0;
  size_t split_out_bytes = 0;
  size_t flag_bytes = 0;

  ~DecodeScratch() {
    if (lse) cudaFree(lse);
    if (split_out) cudaFree(split_out);
    if (split_flag) cudaFree(split_flag);
  }

  void Ensure(size_t need_lse, size_t need_so, size_t need_flag) {
    if (need_lse > lse_bytes) {
      if (lse) cudaFree(lse);
      cudaMalloc(&lse, need_lse);
      lse_bytes = need_lse;
    }
    if (need_so > split_out_bytes) {
      if (split_out) cudaFree(split_out);
      cudaMalloc(&split_out, need_so);
      split_out_bytes = need_so;
    }
    if (need_flag > flag_bytes) {
      if (split_flag) cudaFree(split_flag);
      cudaMalloc(&split_flag, need_flag);
      flag_bytes = need_flag;
    }
  }
};

thread_local DecodeScratch g_decode_scratch;

int GetSplitkOverride() {
  static int v = []() {
    const char* s = std::getenv("KLLM_HPC_DECODE_SPLITK");
    return s ? atoi(s) : 0;
  }();
  return v;
}

}  // namespace

void HpcPagedDecodeAttention(void* out, const void* q, void* kcache, void* vcache, const int* block_table,
                             const int* context_lens, int batch, int q_seq_len, int num_heads, int num_kv_heads,
                             int head_size, int block_size, int max_blocks_per_seq, int num_kvcache_blocks, int ldq,
                             cudaStream_t stream) {
  if (batch <= 0 || q_seq_len <= 0) return;

  const int heads_per_group = num_heads / num_kv_heads;
  int splitk = (batch <= 32) ? 16 : 4;
  if (int o = GetSplitkOverride()) splitk = o;

  const int pad_hpg = ((heads_per_group + 7) / 8) * 8;
  const size_t lse_count = (size_t)batch * splitk * num_kv_heads * q_seq_len * pad_hpg;
  const size_t so_count = (size_t)batch * splitk * q_seq_len * num_heads * head_size;
  const size_t flag_count = (size_t)batch * num_kv_heads;

  g_decode_scratch.Ensure(lse_count * sizeof(float), so_count * sizeof(float), flag_count * sizeof(int));
  cudaMemsetAsync(g_decode_scratch.split_flag, 0, flag_count * sizeof(int), stream);

  const int ldY = num_heads * head_size;
  const int ldK = block_size * num_kv_heads * head_size;
  const int ldV = block_size * num_kv_heads * head_size;

  hpc::attention::decode::smallm_splitk_dim128_async(
      out, g_decode_scratch.lse, g_decode_scratch.split_out, q, kcache, vcache, block_table, context_lens,
      reinterpret_cast<int*>(g_decode_scratch.split_flag), /*new_kv_included=*/true, splitk, batch, q_seq_len,
      num_heads, num_kv_heads, num_kv_heads, head_size, head_size, num_kvcache_blocks, block_size, max_blocks_per_seq,
      ldY, ldq, ldK, ldV, stream);
}

}  // namespace nvidia
}  // namespace llm_kernels
