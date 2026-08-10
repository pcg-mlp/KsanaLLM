/* Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * hpc-ops FP8 fused MoE implementation (blockwise + per-tensor).
 * Source: https://github.com/Tencent/hpc-ops (submodule pinned to 8d300e1)
 * ==============================================================================*/

#include "csrc/kernels/nvidia/moe/hpc_moe/hpc_moe.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_map>

#include "src/fuse_moe/fuse_moe.h"

namespace llm_kernels {
namespace nvidia {

namespace {

constexpr int kBlock = 128;

inline void CudaOrDie(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "[hpc_moe] CUDA error (%s): %s\n", what, cudaGetErrorString(e));
    std::fflush(stderr);
    std::abort();
  }
}

inline size_t Align256(size_t n) { return (n + 255) & ~static_cast<size_t>(255); }

inline int Pad4(int x) { return (x + 3) / 4 * 4; }

// hpc-ops aligned_size / num_padded_tokens rule (fuse_moe/entry.cc:295-306).
inline int AlignedSize(int num_tokens_per_group_avg) {
  if (num_tokens_per_group_avg <= 16) return 16;
  if (num_tokens_per_group_avg <= 32) return 32;
  if (num_tokens_per_group_avg <= 48) return 48;
  return 64;
}

// Per-device scratch arena, grown on demand and reused across every MoE call
// (calls are sequential on the compute stream, so one arena per device is safe).
struct Arena {
  void* base = nullptr;
  size_t cap = 0;
  void* Ensure(size_t bytes) {
    if (bytes > cap) {
      if (base) CudaOrDie(cudaFree(base), "arena free");
      CudaOrDie(cudaMalloc(&base, bytes), "arena alloc");
      cap = bytes;
    }
    return base;
  }
};

std::mutex g_mu;
Arena g_arena[64];
// down_weight_scale [E, hidden/128, inter/128] has an inter/128 last dim that is
// generally NOT a multiple of 4; hpc's TMA loads scales in float4 chunks, so we
// repad each layer's down scale to [E, hidden/128, pad4(inter/128)] exactly once
// and cache it keyed by the source pointer (weights are loaded once, ptrs stable).
std::unordered_map<const void*, void*> g_down_scale_pad;

int CurrentDevice() {
  int dev = 0;
  CudaOrDie(cudaGetDevice(&dev), "getDevice");
  return dev;
}

// Grow-on-demand scratch for the current device, big enough for `total` bytes.
uint8_t* AcquireArena(size_t total) {
  std::lock_guard<std::mutex> lock(g_mu);
  return static_cast<uint8_t*>(g_arena[CurrentDevice() & 63].Ensure(total));
}

// Returns a device buffer holding down_weight_scale with its last dim padded
// from (inter/128) to pad4(inter/128). Cached per source pointer.
const void* GetPaddedDownScale(const void* down_weight_scale, int num_experts, int hidden_size,
                               int moe_inter_size, cudaStream_t stream) {
  const int n_blk = hidden_size / kBlock;        // down N-blocks (e.g. 16)
  const int k_blk = moe_inter_size / kBlock;     // down K-blocks (e.g. 6)
  const int k_pad = Pad4(k_blk);                 // e.g. 8
  if (k_pad == k_blk) return down_weight_scale;  // already aligned, no copy

  std::lock_guard<std::mutex> lock(g_mu);
  auto it = g_down_scale_pad.find(down_weight_scale);
  if (it != g_down_scale_pad.end()) return it->second;

  const size_t rows = static_cast<size_t>(num_experts) * n_blk;  // total scale rows
  const size_t dst_bytes = rows * k_pad * sizeof(float);
  void* dst = nullptr;
  CudaOrDie(cudaMalloc(&dst, dst_bytes), "down-scale pad alloc");
  CudaOrDie(cudaMemsetAsync(dst, 0, dst_bytes, stream), "down-scale pad memset");
  // Strided copy: src rows of k_blk floats (contiguous) -> dst rows of k_pad floats.
  CudaOrDie(cudaMemcpy2DAsync(dst, k_pad * sizeof(float), down_weight_scale,
                              k_blk * sizeof(float), k_blk * sizeof(float), rows,
                              cudaMemcpyDeviceToDevice, stream),
            "down-scale pad copy");
  g_down_scale_pad.emplace(down_weight_scale, dst);
  return dst;
}

}  // namespace

void InvokeHpcBlockwiseFp8Moe(const void* x_q, const void* x_scale, const void* gate_up_weight,
                              const void* gate_up_weight_scale, const void* down_weight,
                              const void* down_weight_scale, const void* topk_ids,
                              const void* topk_scale, void* output, int num_tokens, int hidden_size,
                              int moe_inter_size, int num_experts_local, int num_experts_total,
                              int rank_ep, int topk, float routed_scaling_factor, cudaStream_t stream) {
  if (num_tokens <= 0) return;

  static std::once_flag active_flag;
  std::call_once(active_flag, [&] {
    std::fprintf(stderr,
                 "[hpc_moe] ACTIVE: hpc-ops blockwise-FP8 fused MoE engaged "
                 "(hidden=%d moe_inter=%d E_local=%d E_total=%d rank_ep=%d topk=%d)\n",
                 hidden_size, moe_inter_size, num_experts_local, num_experts_total, rank_ep, topk);
    std::fflush(stderr);
  });

  // hpc "intermediate_size" == gate_up output width == 2 * moe_inter.
  const int inter2 = 2 * moe_inter_size;  // gate_up N (e.g. 1536)
  const int inter_half = moe_inter_size;  // down K (e.g. 768)
  const long long tt = static_cast<long long>(num_tokens) * topk;

  // Following hpc's entry.cc: the per-group token average and the padded-token count key off
  // the GLOBAL expert pool (num_experts_total); the gather/gemm/weight buffers below key off
  // the experts resident on this rank (num_experts_local). They coincide on single card.
  const int avg = static_cast<int>(tt / num_experts_total);
  const int aligned = AlignedSize(avg);
  const int num_padded_tokens = static_cast<int>(
      (tt + static_cast<long long>(num_experts_total) * aligned + aligned - 1) / aligned * aligned);

  const int gate_up_pad4 = Pad4(hidden_size / kBlock);  // 16
  const int down_pad4 = Pad4(inter_half / kBlock);      // 8

  // ---- carve workspaces out of the per-device arena ----
  const size_t sz_gate_up_input = Align256(static_cast<size_t>(tt) * hidden_size);  // fp8
  const size_t sz_gate_up_input_scale =
      Align256(static_cast<size_t>(hidden_size / kBlock) * num_padded_tokens * sizeof(float));
  const size_t sz_gate_up_output = Align256(static_cast<size_t>(tt) * inter2 * sizeof(uint16_t));  // bf16
  const size_t sz_gate_up_tmas = Align256(static_cast<size_t>(num_experts_local) * 2 * 128);
  const size_t sz_down_input = Align256(static_cast<size_t>(tt) * inter_half);  // fp8
  const size_t sz_down_input_scale =
      Align256(static_cast<size_t>(inter_half / kBlock) * num_padded_tokens * sizeof(float));
  const size_t sz_down_output = Align256(static_cast<size_t>(tt) * hidden_size * sizeof(uint16_t));  // bf16
  const size_t sz_down_tmas = Align256(static_cast<size_t>(num_experts_local) * 2 * 128);
  const size_t sz_topk_pos = Align256(static_cast<size_t>(num_tokens) * topk * sizeof(int32_t));
  const size_t sz_ntpg = Align256(static_cast<size_t>(num_experts_local) * sizeof(int32_t));
  const size_t sz_cu_ntpg = Align256(static_cast<size_t>(num_experts_local + 1) * sizeof(int32_t));
  const size_t sz_tiles = Align256(static_cast<size_t>(num_experts_local) * sizeof(int32_t));
  const size_t sz_cu_tiles = Align256(static_cast<size_t>(num_experts_local + 1) * sizeof(int32_t));

  const size_t total = sz_gate_up_input + sz_gate_up_input_scale + sz_gate_up_output +
                       sz_gate_up_tmas + sz_down_input + sz_down_input_scale + sz_down_output +
                       sz_down_tmas + sz_topk_pos + sz_ntpg + sz_cu_ntpg + sz_tiles + sz_cu_tiles;

  uint8_t* p = AcquireArena(total);
  auto Take = [&p](size_t n) {
    void* q = p;
    p += n;
    return q;
  };
  void* gate_up_input = Take(sz_gate_up_input);
  void* gate_up_input_scale = Take(sz_gate_up_input_scale);
  void* gate_up_output = Take(sz_gate_up_output);
  void* gate_up_tmas = Take(sz_gate_up_tmas);
  void* down_input = Take(sz_down_input);
  void* down_input_scale = Take(sz_down_input_scale);
  void* down_output = Take(sz_down_output);
  void* down_tmas = Take(sz_down_tmas);
  void* topk_pos = Take(sz_topk_pos);
  void* num_tokens_per_group = Take(sz_ntpg);
  void* cu_num_tokens_per_group = Take(sz_cu_ntpg);
  void* tiles = Take(sz_tiles);
  void* cu_tiles = Take(sz_cu_tiles);

  // count kernel for num_tokens>128 atomicAdds into num_tokens_per_group without
  // pre-zeroing it -> zero it here (cheap; harmless for the small-batch path).
  CudaOrDie(cudaMemsetAsync(num_tokens_per_group, 0, sz_ntpg, stream), "ntpg memset");

  const void* down_scale =
      GetPaddedDownScale(down_weight_scale, num_experts_local, hidden_size, moe_inter_size, stream);

  // routed_scaling_factor is retained in the signature for forward-compat: today the upstream
  // grouped-topk / softplus-sqrt gate already folds it into topk_scale (see topk_gating_softmax.cu),
  // so callers pass 1.0f and the fused kernel just combines with topk_scale. If a future gate stops
  // pre-folding, the real factor can be routed and applied here instead.
  (void)routed_scaling_factor;

  hpc::fuse_moe::fuse_moe_blockwise_fp8_async(
      output, x_q, x_scale, gate_up_input, gate_up_input_scale, gate_up_output, gate_up_weight,
      gate_up_weight_scale, gate_up_tmas, down_input, down_input_scale, down_output, down_weight,
      down_scale, down_tmas, topk_ids, topk_scale, topk_pos, num_tokens_per_group,
      cu_num_tokens_per_group, tiles, cu_tiles, /*shared_output=*/nullptr, num_tokens,
      num_padded_tokens, hidden_size, inter2, topk, num_experts_total, num_experts_local, gate_up_pad4,
      down_pad4, rank_ep, stream);
}

void InvokeHpcPerTensorFp8Moe(const void* x_q, const void* gate_up_weight, const void* gate_up_scale,
                              const void* act_and_mul_scale, const void* down_weight,
                              const void* down_scale, const void* topk_ids, const void* topk_scale,
                              void* output, int num_tokens, int hidden_size, int moe_inter_size,
                              int num_experts_local, int num_experts_total, int rank_ep, int topk,
                              bool use_bf16_mul, cudaStream_t stream) {
  if (num_tokens <= 0) return;

  static std::once_flag active_flag;
  std::call_once(active_flag, [&] {
    std::fprintf(stderr,
                 "[hpc_moe] ACTIVE: hpc-ops per-tensor-FP8 fused MoE engaged "
                 "(hidden=%d moe_inter=%d E_local=%d E_total=%d rank_ep=%d topk=%d)\n",
                 hidden_size, moe_inter_size, num_experts_local, num_experts_total, rank_ep, topk);
    std::fflush(stderr);
  });

  // hpc "intermediate_size" == gate_up output width == 2 * moe_inter; the down GEMM's K is
  // intermediate_size / 2, which hpc derives internally.
  const int inter2 = 2 * moe_inter_size;
  const long long tt = static_cast<long long>(num_tokens) * topk;

  // ---- carve workspaces out of the per-device arena (mirrors hpc's entry.cc allocations) ----
  // The per-tensor path has no padded-token scale planes: unlike blockwise, every buffer is
  // sized by the real gathered token count (num_tokens * topk).
  const size_t sz_gate_up_input = Align256(static_cast<size_t>(tt) * hidden_size);                   // fp8
  const size_t sz_gate_up_output = Align256(static_cast<size_t>(tt) * inter2 * sizeof(uint16_t));    // bf16
  const size_t sz_gate_up_tmas = Align256(static_cast<size_t>(num_experts_local) * 2 * 128);
  const size_t sz_down_input = Align256(static_cast<size_t>(tt) * moe_inter_size);                   // fp8
  const size_t sz_down_output = Align256(static_cast<size_t>(tt) * hidden_size * sizeof(uint16_t));  // bf16
  const size_t sz_down_tmas = Align256(static_cast<size_t>(num_experts_local) * 2 * 128);
  const size_t sz_topk_pos = Align256(static_cast<size_t>(tt) * sizeof(int32_t));
  const size_t sz_seqlens = Align256(static_cast<size_t>(num_experts_local) * sizeof(int32_t));
  const size_t sz_cu_seqlens = Align256(static_cast<size_t>(num_experts_local + 1) * sizeof(int32_t));
  const size_t sz_tiles = Align256(static_cast<size_t>(num_experts_local) * sizeof(int32_t));
  const size_t sz_cu_tiles = Align256(static_cast<size_t>(num_experts_local + 1) * sizeof(int32_t));

  const size_t total = sz_gate_up_input + sz_gate_up_output + sz_gate_up_tmas + sz_down_input +
                       sz_down_output + sz_down_tmas + sz_topk_pos + sz_seqlens + sz_cu_seqlens +
                       sz_tiles + sz_cu_tiles;

  uint8_t* p = AcquireArena(total);
  auto Take = [&p](size_t n) {
    void* q = p;
    p += n;
    return q;
  };
  void* gate_up_input = Take(sz_gate_up_input);
  void* gate_up_output = Take(sz_gate_up_output);
  void* gate_up_tmas = Take(sz_gate_up_tmas);
  void* down_input = Take(sz_down_input);
  void* down_output = Take(sz_down_output);
  void* down_tmas = Take(sz_down_tmas);
  void* topk_pos = Take(sz_topk_pos);
  void* seqlens = Take(sz_seqlens);
  void* cu_seqlens = Take(sz_cu_seqlens);
  void* tiles = Take(sz_tiles);
  void* cu_tiles = Take(sz_cu_tiles);

  // The count kernel for num_tokens>128 atomicAdds into seqlens without pre-zeroing it
  // (hpc's entry.cc relies on torch::zeros) -> zero it here.
  CudaOrDie(cudaMemsetAsync(seqlens, 0, sz_seqlens, stream), "seqlens memset");

  hpc::fuse_moe::fuse_moe_pertensor_fp8_async(
      output, x_q, gate_up_input, gate_up_output, gate_up_weight, gate_up_scale, gate_up_tmas,
      act_and_mul_scale, down_input, down_output, down_weight, down_scale, down_tmas, topk_ids,
      topk_scale, topk_pos, seqlens, cu_seqlens, tiles, cu_tiles, /*shared_output=*/nullptr,
      num_tokens, hidden_size, inter2, topk, num_experts_total, num_experts_local, rank_ep,
      use_bf16_mul, stream);
}

}  // namespace nvidia
}  // namespace llm_kernels
