/* Copyright 2026 Tencent Inc.  All rights reserved.
==============================================================================*/
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "layers/layer_test_base.h"

#ifdef ENABLE_CUDA

#  include "configure/schedule_config_parser.h"
#  include "csrc/kernels/nvidia/others/humming/preprocess/humming_preprocess.h"
#  include "csrc/kernels/nvidia/others/sglang/main/marlin_moe_wna16/marlin_moe_wna16_preprocess.h"
#  include "csrc/utils/nvidia/workspace.h"
#  include "device/nvidia/cuda_utils.h"
#  include "layers/base_moe_layer.h"
#  include "layers/humming_fp4_moe_layer.h"
#  include "layers/marlin_fp4_moe_layer.h"
#  include "layers/moe_layer.h"
#  include "layers/moe_layer_factory.h"
#  include "layers/moe_perf_stats.h"
#  include "utils/tensor.h"

namespace ksana_llm {

// ===========================================================================
// Benchmark result for one (layer, token_count) pair
// ===========================================================================
struct BenchResult {
  float total_ms;
  float gemm1_ms;
  float gemm2_ms;
  float nogemm_ms;  // total - gemm1 - gemm2 (driver-computed)
  // Humming-only sub-phase breakdown (0 for marlin)
  float gating_ms = 0.f;
  float align_ms = 0.f;
  float quant_ms = 0.f;
  float silu_ms = 0.f;
  float mulsum_ms = 0.f;
  // Humming-only: selected JIT kernel meta (copied by value; the shared static
  // jit_compiler_'s kernel_table_ vector may reallocate when a later backend
  // compiles its kernels, invalidating raw pointers captured by an earlier backend).
  bool has_gemm1_meta = false;
  llm_kernels::nvidia::humming::KernelMeta gemm1_meta{};
  bool has_gemm2_meta = false;
  llm_kernels::nvidia::humming::KernelMeta gemm2_meta{};
  double gemm1_flops;
  double gemm2_flops;
  double total_flops;
  double gemm1_bytes;
  double gemm2_bytes;
  double total_bytes;
};

// ===========================================================================
// Backend enum
// ===========================================================================
enum class Fp4Backend { kTriton = 0, kMarlin = 1, kHummingW4A16 = 2, kHummingW4A8 = 3 };

// ===========================================================================
// Model configuration
// ===========================================================================
struct Fp4MoeTestConfig {
  std::string name;
  size_t hidden_size;
  size_t inter_size;
  size_t num_experts;
  size_t tp_size;
  size_t expert_para_size;
};

static const Fp4MoeTestConfig kFlashTP4 = {"Flash_TP4", 4096, 2048, 256, 4, 1};
static const Fp4MoeTestConfig kFlashEP4 = {"Flash_EP4", 4096, 2048, 256, 1, 4};
static const Fp4MoeTestConfig kProTP8 = {"Pro_TP8", 7168, 3072, 384, 8, 1};
static const Fp4MoeTestConfig kProEP8 = {"Pro_EP8", 7168, 3072, 384, 1, 8};

// ===========================================================================
// Token sizes for benchmarking
// NOTE(jinxcwu) 可以多加一点做完整的性能测试覆盖，目前为了节省测试时间设置的比较小
// ===========================================================================
static const std::vector<size_t> kTokenSizes = {1, 4, 16};
// static const std::vector<size_t> kTokenSizes = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};

static constexpr int kWarmupIters = 20;
static constexpr int kBenchIters = 50;

// ===========================================================================
// Derived dimensions
// ===========================================================================
inline size_t ComputeMoeTp(const Fp4MoeTestConfig& cfg) {
  return std::max(size_t(1), cfg.tp_size / cfg.expert_para_size);
}
inline size_t ComputeInterPerRank(const Fp4MoeTestConfig& cfg) {
  size_t moe_tp = ComputeMoeTp(cfg);
  return (cfg.inter_size + moe_tp - 1) / moe_tp;
}
inline size_t ComputeExpertsPerRank(const Fp4MoeTestConfig& cfg) {
  return (cfg.num_experts + cfg.expert_para_size - 1) / cfg.expert_para_size;
}

// ===========================================================================
// CUDA stream helper
// ===========================================================================
inline cudaStream_t GetStream(std::shared_ptr<Context> context) {
  return context->device->GetPrimaryComputeStream().Get();
}

// ===========================================================================
// Processed weights container (owns weight + scale tensors)
// ===========================================================================
struct ProcessedWeights {
  Tensor w1;  // processed weight (owns memory)
  Tensor w2;
  Tensor w1_scales;  // processed scales (owns memory)
  Tensor w2_scales;
  Tensor w1_global_scale;  // [E] float, only for W4A8
  Tensor w2_global_scale;
};

// ===========================================================================
// Forward reference for deep-copy helper (see helper below)
// ===========================================================================
void DeepCopyTensorInto(const Tensor& src, Tensor& dst, cudaStream_t stream);

// ===========================================================================
// Weight generation — base FP4 weights (all backends start from same data)
// ===========================================================================
ProcessedWeights GenerateBaseWeights(const Fp4MoeTestConfig& cfg, int scale_lo, int scale_hi) {
  ProcessedWeights base;
  size_t E = ComputeExpertsPerRank(cfg);
  size_t N = ComputeInterPerRank(cfg);  // inter_per_rank
  size_t K = cfg.hidden_size;

  auto uint8_opt = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kUInt8);

  // w1 (up_gate fused): [E, N*2, K/2] uint8 — FP4 packed (2 values per byte)
  base.w1 = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_UINT8, {E, N * 2, K / 2});
  torch::from_blob(base.w1.GetPtr<void>(),
                   {static_cast<int64_t>(E), static_cast<int64_t>(N * 2), static_cast<int64_t>(K / 2)}, uint8_opt)
      .random_(0, 256);

  // w1_scales: [E, N*2, K/32] uint8 — E8M0 format
  base.w1_scales = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_UINT8, {E, N * 2, K / 32});
  torch::from_blob(base.w1_scales.GetPtr<void>(),
                   {static_cast<int64_t>(E), static_cast<int64_t>(N * 2), static_cast<int64_t>(K / 32)}, uint8_opt)
      .random_(scale_lo, scale_hi);

  // w2 (down): [E, K, N/2] uint8 — FP4 packed
  base.w2 = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_UINT8, {E, K, N / 2});
  torch::from_blob(base.w2.GetPtr<void>(),
                   {static_cast<int64_t>(E), static_cast<int64_t>(K), static_cast<int64_t>(N / 2)}, uint8_opt)
      .random_(0, 256);

  // w2_scales: [E, K, N/32] uint8 — E8M0 format
  base.w2_scales = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_UINT8, {E, K, N / 32});
  torch::from_blob(base.w2_scales.GetPtr<void>(),
                   {static_cast<int64_t>(E), static_cast<int64_t>(K), static_cast<int64_t>(N / 32)}, uint8_opt)
      .random_(scale_lo, scale_hi);

  return base;
}

// ===========================================================================
// Backend-specific weight processing
// ===========================================================================

// Marlin: transpose + repack + permute_scales + mxfp4_process_scales
ProcessedWeights ProcessWeightsMarlin(const ProcessedWeights& base, const Fp4MoeTestConfig& cfg, cudaStream_t stream) {
  using namespace llm_kernels::nvidia::marlin_moe_wna16;
  size_t E = ComputeExpertsPerRank(cfg);
  size_t N = ComputeInterPerRank(cfg);
  size_t K = cfg.hidden_size;
  int64_t N1 = static_cast<int64_t>(N * 2);  // fused up+gate
  int64_t K1 = static_cast<int64_t>(K);

  ProcessedWeights out;

  // ---- w1 (up_gate): [E, N1, K/2] uint8 → repack ----
  {
    Tensor transposed(MemoryLocation::LOCATION_DEVICE, TYPE_UINT32,
                      {E, static_cast<size_t>(K1 / 8), static_cast<size_t>(N1)});
    transpose_2d_batched(stream, base.w1.GetPtr<void>(), transposed.GetPtr<void>(), static_cast<int64_t>(E), N1, K1 / 8,
                         sizeof(uint32_t));

    auto shape = marlin_moe_repack_meta(K1, N1, 4);
    out.w1 = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_UINT32,
                    {E, static_cast<size_t>(shape[0]), static_cast<size_t>(shape[1])});
    marlin_moe_repack(transposed.GetPtr<const uint32_t>(), out.w1.GetPtr<uint32_t>(), static_cast<int64_t>(E), K1, N1,
                      4, 0, stream);
  }

  // w1 scales: [E, N1, K/32] uint8 → transpose → permute → mxfp4_process_scales
  {
    Tensor transposed(MemoryLocation::LOCATION_DEVICE, TYPE_UINT8,
                      {E, static_cast<size_t>(K / 32), static_cast<size_t>(N1)});
    transpose_2d_batched(stream, base.w1_scales.GetPtr<void>(), transposed.GetPtr<void>(), static_cast<int64_t>(E), N1,
                         static_cast<int64_t>(K / 32), sizeof(uint8_t));

    out.w1_scales =
        Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_UINT8, {E, static_cast<size_t>(K / 32), static_cast<size_t>(N1)});
    marlin_moe_permute_scales<uint8_t>(stream, transposed.GetPtr<const uint8_t>(), out.w1_scales.GetPtr<uint8_t>(),
                                       static_cast<int64_t>(E), K, static_cast<size_t>(N1), 32);
    mxfp4_process_scales(stream, out.w1_scales.GetPtr<uint8_t>(), static_cast<int64_t>(E * (K / 32) * N1));
  }
  out.w1.weight_scales = &out.w1_scales;

  // ---- w2 (down): [E, K, N/2] uint8 → repack ----
  {
    int64_t N2 = K1;                       // hidden
    int64_t K2 = static_cast<int64_t>(N);  // inter
    Tensor transposed(MemoryLocation::LOCATION_DEVICE, TYPE_UINT32,
                      {E, static_cast<size_t>(K2 / 8), static_cast<size_t>(N2)});
    transpose_2d_batched(stream, base.w2.GetPtr<void>(), transposed.GetPtr<void>(), static_cast<int64_t>(E), N2, K2 / 8,
                         sizeof(uint32_t));

    auto shape = marlin_moe_repack_meta(K2, N2, 4);
    out.w2 = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_UINT32,
                    {E, static_cast<size_t>(shape[0]), static_cast<size_t>(shape[1])});
    marlin_moe_repack(transposed.GetPtr<const uint32_t>(), out.w2.GetPtr<uint32_t>(), static_cast<int64_t>(E), K2, N2,
                      4, 0, stream);
  }

  // w2 scales
  {
    int64_t N2 = K1;                       // hidden
    int64_t K2 = static_cast<int64_t>(N);  // inter
    Tensor transposed(MemoryLocation::LOCATION_DEVICE, TYPE_UINT8,
                      {E, static_cast<size_t>(K2 / 32), static_cast<size_t>(N2)});
    transpose_2d_batched(stream, base.w2_scales.GetPtr<void>(), transposed.GetPtr<void>(), static_cast<int64_t>(E), N2,
                         K2 / 32, sizeof(uint8_t));

    out.w2_scales =
        Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_UINT8, {E, static_cast<size_t>(K2 / 32), static_cast<size_t>(N2)});
    marlin_moe_permute_scales<uint8_t>(stream, transposed.GetPtr<const uint8_t>(), out.w2_scales.GetPtr<uint8_t>(),
                                       static_cast<int64_t>(E), K2, N2, 32);
    mxfp4_process_scales(stream, out.w2_scales.GetPtr<uint8_t>(), static_cast<int64_t>(E * (K2 / 32) * N2));
  }
  out.w2.weight_scales = &out.w2_scales;

  return out;
}

// Humming W4A16: repack_weight_wgmma + process_scales (BF16 compute)
ProcessedWeights ProcessWeightsHummingW4A16(const ProcessedWeights& base, const Fp4MoeTestConfig& cfg,
                                            cudaStream_t stream) {
  using namespace llm_kernels::nvidia::humming;
  size_t E = ComputeExpertsPerRank(cfg);
  size_t N = ComputeInterPerRank(cfg);
  size_t K = cfg.hidden_size;
  int64_t N1 = static_cast<int64_t>(N * 2);
  int64_t K1 = static_cast<int64_t>(K);

  ProcessedWeights out;

  // w1
  out.w1 =
      Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_UINT8, {E, static_cast<size_t>(N1), static_cast<size_t>(K / 2)});
  repack_weight_wgmma(stream, base.w1.GetPtr<void>(), out.w1.GetPtr<void>(), static_cast<int64_t>(E), N1, K1,
                      ComputeDtype::BF16);

  out.w1_scales =
      Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_UINT8, {E, static_cast<size_t>(K / 32), static_cast<size_t>(N1)});
  process_scales(stream, base.w1_scales.GetPtr<void>(), out.w1_scales.GetPtr<void>(), static_cast<int64_t>(E), N1, K1,
                 32, ComputeDtype::BF16);
  out.w1.weight_scales = &out.w1_scales;

  // w2
  out.w2 = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_UINT8, {E, K, static_cast<size_t>(N / 2)});
  repack_weight_wgmma(stream, base.w2.GetPtr<void>(), out.w2.GetPtr<void>(), static_cast<int64_t>(E), K1,
                      static_cast<int64_t>(N), ComputeDtype::BF16);

  out.w2_scales = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_UINT8, {E, static_cast<size_t>(N / 32), K});
  process_scales(stream, base.w2_scales.GetPtr<void>(), out.w2_scales.GetPtr<void>(), static_cast<int64_t>(E), K1,
                 static_cast<int64_t>(N), 32, ComputeDtype::BF16);
  out.w2.weight_scales = &out.w2_scales;

  return out;
}

// Humming W4A8: process_mxfp4_w4a8_weight + repack + process_scales (FP8 compute)
ProcessedWeights ProcessWeightsHummingW4A8(const ProcessedWeights& base, const Fp4MoeTestConfig& cfg,
                                           cudaStream_t stream) {
  using namespace llm_kernels::nvidia::humming;
  size_t E = ComputeExpertsPerRank(cfg);
  size_t N = ComputeInterPerRank(cfg);
  size_t K = cfg.hidden_size;
  int64_t N1 = static_cast<int64_t>(N * 2);
  int64_t K1 = static_cast<int64_t>(K);

  ProcessedWeights out;

  // ---- w1: copy → process_mxfp4_w4a8 (in-place) → repack → process_scales ----
  {
    // Deep copy base weights (process_mxfp4_w4a8 modifies in-place)
    Tensor w1_copy(MemoryLocation::LOCATION_DEVICE, TYPE_UINT8,
                   {E, static_cast<size_t>(N1), static_cast<size_t>(K / 2)});
    Tensor s1_copy(MemoryLocation::LOCATION_DEVICE, TYPE_UINT8,
                   {E, static_cast<size_t>(N1), static_cast<size_t>(K / 32)});
    cudaMemcpyAsync(w1_copy.GetPtr<void>(), base.w1.GetPtr<void>(), w1_copy.GetTotalBytes(), cudaMemcpyDeviceToDevice,
                    stream);
    cudaMemcpyAsync(s1_copy.GetPtr<void>(), base.w1_scales.GetPtr<void>(), s1_copy.GetTotalBytes(),
                    cudaMemcpyDeviceToDevice, stream);

    out.w1_global_scale = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {E});
    process_mxfp4_w4a8_weight(stream, w1_copy.GetPtr<void>(), s1_copy.GetPtr<void>(),
                              out.w1_global_scale.GetPtr<float>(), static_cast<int64_t>(E), N1, K1, 32);

    out.w1 =
        Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_UINT8, {E, static_cast<size_t>(N1), static_cast<size_t>(K / 2)});
    repack_weight_wgmma(stream, w1_copy.GetPtr<void>(), out.w1.GetPtr<void>(), static_cast<int64_t>(E), N1, K1,
                        ComputeDtype::FP8);

    out.w1_scales =
        Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_UINT8, {E, static_cast<size_t>(K / 32), static_cast<size_t>(N1)});
    process_scales(stream, s1_copy.GetPtr<void>(), out.w1_scales.GetPtr<void>(), static_cast<int64_t>(E), N1, K1, 32,
                   ComputeDtype::FP8);
    out.w1.weight_scales = &out.w1_scales;
  }

  // ---- w2: same pattern ----
  {
    Tensor w2_copy(MemoryLocation::LOCATION_DEVICE, TYPE_UINT8, {E, K, static_cast<size_t>(N / 2)});
    Tensor s2_copy(MemoryLocation::LOCATION_DEVICE, TYPE_UINT8, {E, K, static_cast<size_t>(N / 32)});
    cudaMemcpyAsync(w2_copy.GetPtr<void>(), base.w2.GetPtr<void>(), w2_copy.GetTotalBytes(), cudaMemcpyDeviceToDevice,
                    stream);
    cudaMemcpyAsync(s2_copy.GetPtr<void>(), base.w2_scales.GetPtr<void>(), s2_copy.GetTotalBytes(),
                    cudaMemcpyDeviceToDevice, stream);

    out.w2_global_scale = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {E});
    process_mxfp4_w4a8_weight(stream, w2_copy.GetPtr<void>(), s2_copy.GetPtr<void>(),
                              out.w2_global_scale.GetPtr<float>(), static_cast<int64_t>(E), K1, static_cast<int64_t>(N),
                              32);

    out.w2 = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_UINT8, {E, K, static_cast<size_t>(N / 2)});
    repack_weight_wgmma(stream, w2_copy.GetPtr<void>(), out.w2.GetPtr<void>(), static_cast<int64_t>(E), K1,
                        static_cast<int64_t>(N), ComputeDtype::FP8);

    out.w2_scales = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_UINT8, {E, static_cast<size_t>(N / 32), K});
    process_scales(stream, s2_copy.GetPtr<void>(), out.w2_scales.GetPtr<void>(), static_cast<int64_t>(E), K1,
                   static_cast<int64_t>(N), 32, ComputeDtype::FP8);
    out.w2.weight_scales = &out.w2_scales;
  }

  return out;
}

// ===========================================================================
// Deep-copy a Tensor (allocates new buffer + cudaMemcpy)
// ===========================================================================
void DeepCopyTensorInto(const Tensor& src, Tensor& dst, cudaStream_t stream) {
  dst = Tensor(MemoryLocation::LOCATION_DEVICE, src.dtype, src.shape);
  if (stream != nullptr) {
    cudaMemcpyAsync(dst.GetPtr<void>(), src.GetPtr<void>(), src.GetTotalBytes(), cudaMemcpyDeviceToDevice, stream);
  } else {
    cudaMemcpy(dst.GetPtr<void>(), src.GetPtr<void>(), src.GetTotalBytes(), cudaMemcpyDeviceToDevice);
  }
}

// ===========================================================================
// FP4 weight dequantization on GPU (torch vectorized ops)
//
// Input:  w_gpu [E, N, K/2] uint8 (FP4 packed, low nibble first)
//         s_gpu [E, N, K/32] uint8 (E8M0 scales)
// Output: [E, N, K] float32 GPU tensor
// Formula: value = Fp4Lut[nibble] * 2^(scale_byte - 127)  per group of 32
//
// Uses standard torch GPU ops (gather, pow, repeat_interleave) — independent
// code path from Humming/Marlin fused FP4 kernels, so still a valid reference.
// ===========================================================================
torch::Tensor DequantFp4WeightGpu(const Tensor& w_gpu, const Tensor& s_gpu, size_t E, size_t N, size_t K) {
  auto u8_opt = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kUInt8);
  auto f32_opt = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kFloat32);

  auto w = torch::from_blob(w_gpu.GetPtr<void>(),
                            {static_cast<int64_t>(E), static_cast<int64_t>(N), static_cast<int64_t>(K / 2)}, u8_opt);
  auto s = torch::from_blob(s_gpu.GetPtr<void>(),
                            {static_cast<int64_t>(E), static_cast<int64_t>(N), static_cast<int64_t>(K / 32)}, u8_opt);

  // FP4 E2M1 LUT
  static const float kFp4Lut[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                                    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
  auto lut = torch::from_blob(const_cast<float*>(kFp4Lut), {16}, torch::TensorOptions().dtype(torch::kFloat32))
                 .to(torch::kCUDA, /*non_blocking=*/true);

  // Split low/high nibbles: [E, N, K/2] -> gather LUT -> [E, N, K/2]
  auto w_i32 = w.to(torch::kInt32);
  auto low = w_i32 & 0xF;                                  // [E, N, K/2]
  auto high = torch::bitwise_right_shift(w_i32, 4) & 0xF;  // [E, N, K/2]

  // LUT lookup via gather, then interleave into [E, N, K]
  auto low_vals = lut.index_select(0, low.reshape(-1))
                      .reshape({static_cast<int64_t>(E), static_cast<int64_t>(N), static_cast<int64_t>(K / 2)});
  auto high_vals = lut.index_select(0, high.reshape(-1))
                       .reshape({static_cast<int64_t>(E), static_cast<int64_t>(N), static_cast<int64_t>(K / 2)});

  // Interleave: result[..., 2*i] = low_vals[..., i], result[..., 2*i+1] = high_vals[..., i]
  auto values = torch::empty({static_cast<int64_t>(E), static_cast<int64_t>(N), static_cast<int64_t>(K)}, f32_opt);
  auto view2 = values.view({static_cast<int64_t>(E), static_cast<int64_t>(N), static_cast<int64_t>(K / 2), 2});
  view2.select(-1, 0).copy_(low_vals);
  view2.select(-1, 1).copy_(high_vals);

  // Scale: 2^(byte - 127), per group of 32
  auto scales = torch::pow(2.0f, s.to(torch::kFloat32) - 127.0f);  // [E, N, K/32]
  scales = scales.repeat_interleave(32, /*dim=*/-1);               // [E, N, K]

  return values * scales;  // [E, N, K] float32 on GPU
}

// ===========================================================================
// FP8 E4M3 quantize-dequantize on GPU (matches Triton _quant_tensor_kernel)
//
// Per-token: scale = max(|x|, 1e-30) / 448
//            x_fp8 = cast_to_fp8_e4m3fn_rne_sat(x / scale)
//            x_dequant = fp8_to_float(x_fp8) * scale
// ===========================================================================
torch::Tensor Fp8QuantDequantGpu(torch::Tensor x) {
  auto absmax = std::get<0>(x.abs().max(-1)).clamp_min(1e-30);
  auto scale = absmax / 448.0f;
  auto x_scaled = x / scale.unsqueeze(-1);
  auto x_fp8 = x_scaled.to(torch::kFloat8_e4m3fn);
  return x_fp8.to(torch::kFloat32) * scale.unsqueeze(-1);
}

// ===========================================================================
// GPU reference FP4 MOE forward — computes from pre-dequantized FP32 weights
//
// Pipeline: TopK gating -> GEMM1 -> SwiGLU(limit) -> GEMM2 -> weighted sum
// For W4A8: activations are FP8-quantized before each GEMM
//
// All ops use standard torch GPU kernels (cuBLAS matmul, element-wise) — a
// completely independent code path from Humming/Marlin fused FP4 kernels.
// Accepts pre-dequantized weights to avoid redundant dequant across token sizes.
// ===========================================================================
torch::Tensor GpuRefFp4MoeForward(const torch::Tensor& w1_dequant, const torch::Tensor& w2_dequant,
                                  const Fp4MoeTestConfig& cfg, torch::Tensor hs_gpu, torch::Tensor gate_gpu,
                                  Fp4Backend backend, float swiglu_limit) {
  size_t E = ComputeExpertsPerRank(cfg);
  size_t N = ComputeInterPerRank(cfg);
  size_t K = cfg.hidden_size;
  size_t topk = 6;
  int64_t M = hs_gpu.size(0);

  // 1. TopK gating: scores = sqrt(softplus(gate))
  auto gate_f32 = gate_gpu.to(torch::kFloat32);
  auto scores = torch::sqrt(torch::nn::functional::softplus(gate_f32));
  auto topk_result = torch::topk(scores, static_cast<int64_t>(topk), /*dim=*/-1, /*largest=*/true, /*sorted=*/true);
  auto topk_weights = std::get<0>(topk_result).to(torch::kFloat32);  // [M, topk]
  auto topk_ids = std::get<1>(topk_result).to(torch::kInt64);        // [M, topk]

  // Flatten routing indices for per-expert grouping
  auto flat_ids = topk_ids.reshape(-1);  // [M*topk]
  auto flat_m = torch::arange(M, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt64))
                    .unsqueeze(1)
                    .expand({M, static_cast<int64_t>(topk)})
                    .reshape(-1);  // [M*topk]

  // 2. GEMM1: per-expert batched matmul
  auto x = hs_gpu.to(torch::kFloat32);  // [M, K]
  if (backend == Fp4Backend::kHummingW4A8) {
    x = Fp8QuantDequantGpu(x);
  }
  auto cache1 = torch::zeros({M * static_cast<int64_t>(topk), static_cast<int64_t>(N * 2)},
                             torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
  for (int64_t e = 0; e < static_cast<int64_t>(E); ++e) {
    auto mask = (flat_ids == e);
    if (!mask.any().item<bool>()) continue;
    auto row_idx = mask.nonzero().squeeze(1);      // [num_rows]
    auto m_idx = flat_m.index_select(0, row_idx);  // [num_rows]
    auto x_e = x.index_select(0, m_idx);           // [num_rows, K]
    cache1.index_copy_(0, row_idx, x_e.matmul(w1_dequant[e].t()));
  }

  // 3. SwiGLU with limit
  auto gate_part = cache1.slice(1, 0, static_cast<int64_t>(N));
  auto up_part = cache1.slice(1, static_cast<int64_t>(N), static_cast<int64_t>(N * 2));
  gate_part = torch::clamp_max(gate_part, swiglu_limit);
  up_part = torch::clamp(up_part, -swiglu_limit, swiglu_limit);
  auto cache2 = torch::nn::functional::silu(gate_part) * up_part;  // [M*topk, N]

  // 4. GEMM2: per-expert batched matmul
  auto x2 = cache2;
  if (backend == Fp4Backend::kHummingW4A8) {
    x2 = Fp8QuantDequantGpu(x2);
  }
  auto cache3 = torch::zeros({M * static_cast<int64_t>(topk), static_cast<int64_t>(K)},
                             torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
  for (int64_t e = 0; e < static_cast<int64_t>(E); ++e) {
    auto mask = (flat_ids == e);
    if (!mask.any().item<bool>()) continue;
    auto row_idx = mask.nonzero().squeeze(1);
    auto m_idx = flat_m.index_select(0, row_idx);
    auto x2_e = x2.index_select(0, row_idx);  // [num_rows, N]
    cache3.index_copy_(0, row_idx, x2_e.matmul(w2_dequant[e].t()));
  }

  // 5. Final reduce: output[m] = sum_k topk_weights[m,k] * cache3[m*topk+k]
  auto topk_w = topk_weights.reshape({-1, 1});                                               // [M*topk, 1]
  auto weighted = cache3 * topk_w;                                                           // [M*topk, K]
  return weighted.reshape({M, static_cast<int64_t>(topk), static_cast<int64_t>(K)}).sum(1);  // [M, K]
}

// ===========================================================================
// Build 24-parameter Init list
// ===========================================================================
std::vector<std::any> BuildMoeParams(const Fp4MoeTestConfig& cfg, size_t max_token_num, float swiglu_limit) {
  size_t inter_per_rank = ComputeInterPerRank(cfg);
  size_t experts_per_rank = ComputeExpertsPerRank(cfg);
  size_t moe_tp = ComputeMoeTp(cfg);

  std::vector<std::any> params;
  params.push_back(MoeScaleNormMode::NO_NORM);     // 0
  params.push_back(max_token_num);                 // 1
  params.push_back(0);                             // 2: layer_idx
  params.push_back(experts_per_rank);              // 3
  params.push_back(cfg.hidden_size);               // 4: expert_hidden_size
  params.push_back(inter_per_rank);                // 5: expert_inter_size
  params.push_back(size_t(6));                     // 6: expert_topk
  params.push_back(moe_tp);                        // 7: tp_size
  params.push_back(true);                          // 8: use_vllm_moe
  params.push_back(uint32_t(1));                   // 9
  params.push_back(uint32_t(1));                   // 10
  params.push_back(std::string("sqrtsoftplus"));   // 11
  params.push_back(std::string("greedy"));         // 12
  params.push_back(false);                         // 13
  params.push_back(1.0f);                          // 14
  params.push_back(false);                         // 15
  params.push_back(false);                         // 16
  params.push_back(DataType::TYPE_INVALID);        // 17
  params.push_back(DataType::TYPE_FP4_E2M1FN_x2);  // 18: int_weight_dtype
  params.push_back(128);                           // 19: group_size
  params.push_back(false);                         // 20: apply_weight
  params.push_back(swiglu_limit);                  // 21: swiglu_limit
  params.push_back(uint32_t(0));                   // 22: num_hash_layers
  params.push_back(uint32_t(0));                   // 23: vocab_size
  return params;
}

// ===========================================================================
// Create layer by backend type (Init + SetWorkspaceBuffer + Preprocess)
// ===========================================================================
std::unique_ptr<BaseLayer> CreateLayer(Fp4Backend backend, const std::vector<std::any>& params,
                                       RuntimeConfig& runtime_config, std::shared_ptr<Context> context, int rank) {
  switch (backend) {
    case Fp4Backend::kMarlin:
      runtime_config.fp4_moe_backend = FP4_MOE_BACKEND_TYPE::FP4Marlin;
      break;
    case Fp4Backend::kHummingW4A16:
      runtime_config.fp4_moe_backend = FP4_MOE_BACKEND_TYPE::FP4HummingW4A16;
      break;
    case Fp4Backend::kHummingW4A8:
      runtime_config.fp4_moe_backend = FP4_MOE_BACKEND_TYPE::FP4HummingW4A8;
      break;
    case Fp4Backend::kTriton:
      break;  // unused
  }

  std::unique_ptr<BaseLayer> layer;
  switch (backend) {
    case Fp4Backend::kMarlin:
      layer = std::make_unique<MarlinFp4MoeLayer>();
      break;
    case Fp4Backend::kHummingW4A16:
    case Fp4Backend::kHummingW4A8:
      layer = std::make_unique<HummingFp4MoeLayer>();
      break;
    case Fp4Backend::kTriton:
      break;  // unused
  }

  layer->Init(params, runtime_config, context, rank);

  auto workspace_mgr = std::make_shared<LayerWorkspaceManager>(rank);
  size_t ws_size = layer->GetWorkspaceSize();
  if (ws_size > 0) {
    layer->SetWorkspaceBuffer(workspace_mgr->GetWorkspace(ws_size));
  }

  ModelConfig mc;
  mc.weight_data_type = TYPE_BF16;
  mc.path = "/model/llama-hf/7B/";
  layer->Preprocess(mc, runtime_config);

  return layer;
}

// ===========================================================================
// Setup inputs vector with weight tensors (deep-copied)
//
// Returns persistent_inputs with weights in [2] and [3], scale pointers bound.
// Activation slots [0] and [1] are left empty — caller fills them per iteration.
// ===========================================================================
std::vector<Tensor> SetupPersistentInputs(const ProcessedWeights& pw, Fp4Backend backend, cudaStream_t stream) {
  std::vector<Tensor> inputs(4);

  // Deep-copy weight tensors into inputs so they have independent ownership
  DeepCopyTensorInto(pw.w1, inputs[2], stream);
  DeepCopyTensorInto(pw.w2, inputs[3], stream);

  // Bind scale pointers (point to ProcessedWeights' scale tensors, which must
  // remain alive for the lifetime of `inputs`)
  inputs[2].weight_scales = pw.w1.weight_scales;
  inputs[3].weight_scales = pw.w2.weight_scales;
  if (backend == Fp4Backend::kHummingW4A8) {
    inputs[2].weight_global_scale = const_cast<Tensor*>(&pw.w1_global_scale);
    inputs[3].weight_global_scale = const_cast<Tensor*>(&pw.w2_global_scale);
  }
  return inputs;
}
// ===========================================================================
// Compute FLOPs for the GEMM kernels in a single MoE Forward.
//
// NOTE: These formulas assume ALL topk-selected tokens are computed on the
// current rank, which holds for TP-only configurations.  Under Expert
// Parallelism (EP, expert_para_size > 1) each rank only processes tokens
// routed to its local expert subset, so the effective FLOPs depend on the
// actual token-expert distribution and cannot be computed from static config
// alone.  The driver therefore skips TFLOPS output for EP configs.
//
//   gemm1: 2 * M * topk * hidden * (inter * 2 / moe_tp)
//   gemm2: 2 * M * topk * hidden * (inter / moe_tp)
//   total: gemm1 + gemm2   (non-GEMM FLOPs are negligible)
// ===========================================================================
void ComputeFlops(const Fp4MoeTestConfig& cfg, size_t num_tokens, double& gemm1_flops, double& gemm2_flops,
                  double& total_flops) {
  size_t moe_tp = ComputeMoeTp(cfg);
  size_t inter_per_rank = ComputeInterPerRank(cfg);
  size_t K = cfg.hidden_size;
  double M = static_cast<double>(num_tokens);
  double topk = 6.0;  // expert_topk

  // GEMM1: [M, K] × [E, K, inter*2/moe_tp] → [M*topk, inter*2/moe_tp]
  double gemm1_m = M * topk;  // gates
  double gemm1_k = K;
  double gemm1_n = static_cast<double>(inter_per_rank) * 2.0;  // fused gate+up
  gemm1_flops = 2.0 * gemm1_m * gemm1_k * gemm1_n;             // multiply-add counted as 2 operations

  // GEMM2: [M*topk, inter/moe_tp] × [E, inter/moe_tp, K] → [M*topk, K]
  double gemm2_m = M * topk;
  double gemm2_k = static_cast<double>(inter_per_rank);
  double gemm2_n = K;
  gemm2_flops = 2.0 * gemm2_m * gemm2_k * gemm2_n;

  total_flops = gemm1_flops + gemm2_flops;
}

// ===========================================================================
// Compute memory bytes moved for the GEMM kernels in a single MoE Forward.
//
// Same TP-only assumption as ComputeFlops.  Bytes counted:
//   - FP4 weights: 0.5 bytes/element
//   - E8M0 scales: 1 byte per 32 weight elements
//   - BF16 activations: 2 bytes/element (input read + output write)
//
// EP configs are skipped (same reason as TFLOPS — actual per-rank work
// depends on token-expert distribution).
// ===========================================================================
void ComputeBytes(const Fp4MoeTestConfig& cfg, size_t num_tokens, double& gemm1_bytes, double& gemm2_bytes,
                  double& total_bytes) {
  size_t E = ComputeExpertsPerRank(cfg);
  size_t inter_per_rank = ComputeInterPerRank(cfg);
  size_t K = cfg.hidden_size;
  double M = static_cast<double>(num_tokens);
  double topk = 6.0;  // expert_topk

  constexpr double kFp4Bytes = 0.5;         // 4 bits per element
  constexpr double kScaleBytes = 1.0 / 32;  // 1 byte (E8M0) per 32 elements
  constexpr double kBF16Bytes = 2.0;

  // Only experts that actually receive tokens have their weights loaded.
  // With M tokens × topk selections, at most min(M*topk, E) distinct experts
  // are accessed; each accessed expert's weight is loaded once from HBM and
  // reused by all tokens routed to it.
  double experts_accessed = std::min(M * topk, static_cast<double>(E));

  // GEMM1: [M*topk, K] × [experts_accessed, K, N1] → [M*topk, N1],  N1 = inter*2
  double gemm1_m = M * topk;
  double gemm1_n = static_cast<double>(inter_per_rank) * 2.0;
  gemm1_bytes = experts_accessed * gemm1_n * K * (kFp4Bytes + kScaleBytes)  // weights + scales
                + gemm1_m * K * kBF16Bytes                                  // input
                + gemm1_m * gemm1_n * kBF16Bytes;                           // output

  // GEMM2: [M*topk, N] × [experts_accessed, K, N] → [M*topk, K],  N = inter
  double gemm2_m = M * topk;
  double gemm2_n = static_cast<double>(inter_per_rank);
  gemm2_bytes = experts_accessed * K * gemm2_n * (kFp4Bytes + kScaleBytes)  // weights + scales
                + gemm2_m * gemm2_n * kBF16Bytes                            // input
                + gemm2_m * K * kBF16Bytes;                                 // output

  total_bytes = gemm1_bytes + gemm2_bytes;
}

// ===========================================================================
// Measure one (num_tokens, layer) configuration.
// Encapsulates warmup + benchmark + per-GEMM timing, similar in spirit to
// MeasureCudaExecutionTime but also harvests MoePerfStats from the layer.
// ===========================================================================
template <typename Func>
BenchResult MeasureMoeCudaExecutionTime(Func&& forward, MoePerfStats* perf_stats, cudaStream_t stream,
                                        const Fp4MoeTestConfig& cfg, size_t num_tokens, int warmups, int iterations) {
  // Warmup
  for (int i = 0; i < warmups; ++i) forward();
  cudaDeviceSynchronize();

  // Reset perf stats after warmup so only benchmark iterations count
  perf_stats->Reset();

  // Benchmark: measure total GPU time with stream events
  cudaEvent_t begin, end;
  cudaEventCreate(&begin);
  cudaEventCreate(&end);

  cudaEventRecord(begin, stream);
  for (int i = 0; i < iterations; ++i) forward();
  cudaEventRecord(end, stream);
  cudaEventSynchronize(end);

  float total_ms;
  cudaEventElapsedTime(&total_ms, begin, end);
  cudaEventDestroy(begin);
  cudaEventDestroy(end);

  BenchResult br;
  br.total_ms = total_ms / iterations;
  br.gemm1_ms = perf_stats->gemm1_time_ms / iterations;
  br.gemm2_ms = perf_stats->gemm2_time_ms / iterations;
  br.nogemm_ms = br.total_ms - br.gemm1_ms - br.gemm2_ms;
  // Humming-only sub-phase times (0 for marlin — fields stay at default)
  br.gating_ms = perf_stats->gating_time_ms / iterations;
  br.align_ms = perf_stats->align_time_ms / iterations;
  br.quant_ms = perf_stats->quant_time_ms / iterations;
  br.silu_ms = perf_stats->silu_time_ms / iterations;
  br.mulsum_ms = perf_stats->mulsum_time_ms / iterations;
  // Humming-only: selected kernel meta (nullptr for marlin — same meta across all iters).
  // Copy by value: the meta pointer points into the static jit_compiler_'s
  // kernel_table_, which may be reallocated when a subsequent backend compiles.
  if (perf_stats->last_gemm1_meta != nullptr) {
    br.gemm1_meta = *perf_stats->last_gemm1_meta;
    br.has_gemm1_meta = true;
  }
  if (perf_stats->last_gemm2_meta != nullptr) {
    br.gemm2_meta = *perf_stats->last_gemm2_meta;
    br.has_gemm2_meta = true;
  }
  ComputeFlops(cfg, num_tokens, br.gemm1_flops, br.gemm2_flops, br.total_flops);
  ComputeBytes(cfg, num_tokens, br.gemm1_bytes, br.gemm2_bytes, br.total_bytes);

  return br;
}

// ===========================================================================
// Run benchmark across all token sizes for one (layer, inputs) pair.
// Templated on the concrete layer type so we can call SetPerfStats directly.
// Also captures the last forward output per token size for accuracy checking.
// ===========================================================================
template <typename LayerT>
std::pair<std::vector<BenchResult>, std::vector<torch::Tensor>> RunBenchmark(LayerT* layer,
                                                                             std::vector<Tensor>& persistent_inputs,
                                                                             const Fp4MoeTestConfig& cfg,
                                                                             cudaStream_t stream, const void* hs_full,
                                                                             const void* gate_full) {
  static constexpr size_t kBytesPerBF16 = 2;
  size_t K = cfg.hidden_size;
  size_t E = cfg.num_experts;

  MoePerfStats perf_stats;
  layer->SetPerfStats(&perf_stats);

  std::vector<BenchResult> results;
  results.reserve(kTokenSizes.size());
  std::vector<torch::Tensor> outputs;
  outputs.reserve(kTokenSizes.size());

  for (size_t num_tokens : kTokenSizes) {
    persistent_inputs[0] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, {num_tokens, K});
    persistent_inputs[1] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, {num_tokens, E});
    cudaMemcpyAsync(persistent_inputs[0].GetPtr<void>(), hs_full, num_tokens * K * kBytesPerBF16,
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(persistent_inputs[1].GetPtr<void>(), gate_full, num_tokens * E * kBytesPerBF16,
                    cudaMemcpyDeviceToDevice, stream);

    Tensor output(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, {num_tokens, K});
    Tensor eplb_map;
    Tensor workspace;

    auto forward = [&]() {
      layer->Forward(&persistent_inputs[0], &persistent_inputs[1], &persistent_inputs[2], &persistent_inputs[3],
                     &eplb_map, &output, &workspace);
    };

    results.push_back(
        MeasureMoeCudaExecutionTime(forward, &perf_stats, stream, cfg, num_tokens, kWarmupIters, kBenchIters));

    // Capture output as FP32 on GPU for accuracy verification
    cudaStreamSynchronize(stream);
    auto gpu_out = torch::from_blob(output.GetPtr<void>(), {static_cast<int64_t>(num_tokens), static_cast<int64_t>(K)},
                                    torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kBFloat16))
                       .to(torch::kFloat32);
    outputs.push_back(std::move(gpu_out));
  }

  layer->SetPerfStats(nullptr);
  return {results, outputs};
}

// ===========================================================================
// Compute TFLOPS from FLOPs and time
// ===========================================================================
inline double ComputeTFLOPS(double flops, float time_ms) {
  return time_ms > 0.f ? (flops / (time_ms / 1e3)) / 1e12 : 0.;
}

// ===========================================================================
// Compute bandwidth (GB/s) from bytes and time
// ===========================================================================
inline double ComputeBandwidthGBps(double bytes, float time_ms) {
  return time_ms > 0.f ? bytes / (time_ms / 1e3) / 1e9 : 0.;
}

// ===========================================================================
// Print latency table.  Cell format: "g1/g2/nogemm/total" (ms, 4 decimals).
// nogemm = total - gemm1 - gemm2 (driver-computed; works for all backends).
// ===========================================================================
void PrintLatencyTable(const std::string& config_name, const std::vector<std::vector<BenchResult>>& all_results) {
  static constexpr int kCellW = 32;  // "0.0123/0.0456/0.0789/0.1368" fits in 32 chars

  std::cout << "\n=== " << config_name << " Latency (ms, GEMM1 / GEMM2 / NoGEMM / Total) ===" << std::endl;
  std::cout << " " << std::setw(10) << "tokens";
  std::cout << std::setw(kCellW) << "Marlin";
  std::cout << std::setw(kCellW) << "HummingW4A16";
  std::cout << std::setw(kCellW) << "HummingW4A8";
  std::cout << std::endl;

  for (size_t i = 0; i < kTokenSizes.size(); ++i) {
    std::cout << " " << std::setw(10) << kTokenSizes[i];
    for (size_t b = 0; b < all_results.size(); ++b) {
      if (i < all_results[b].size()) {
        const auto& r = all_results[b][i];
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(4) << r.gemm1_ms << "/" << r.gemm2_ms << "/" << r.nogemm_ms << "/"
            << r.total_ms;
        std::cout << std::setw(kCellW) << oss.str();
      } else {
        std::cout << std::setw(kCellW) << "N/A";
      }
    }
    std::cout << std::endl;
  }
}

// ===========================================================================
// Print a single TFLOPS table with format "g1/g2/total" per cell.
// Marlin is always present as the accuracy reference.
// ===========================================================================
void PrintTFLOPS(const std::string& config_name, const std::vector<std::vector<BenchResult>>& all_results) {
  static constexpr int kCellW = 24;  // "xx.xx/xx.xx/xx.xx" fits in 24 chars

  // Header
  std::cout << "\n=== " << config_name << " TFLOPS (GEMM1 / GEMM2 / Total) ===" << std::endl;
  std::cout << " " << std::setw(10) << "tokens";
  std::cout << std::setw(kCellW) << "Marlin";
  std::cout << std::setw(kCellW) << "HummingW4A16";
  std::cout << std::setw(kCellW) << "HummingW4A8";
  std::cout << std::endl;

  for (size_t i = 0; i < kTokenSizes.size(); ++i) {
    std::cout << " " << std::setw(10) << kTokenSizes[i];
    for (size_t b = 0; b < all_results.size(); ++b) {
      if (i < all_results[b].size()) {
        const auto& r = all_results[b][i];
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << ComputeTFLOPS(r.gemm1_flops, r.gemm1_ms) << "/"
            << ComputeTFLOPS(r.gemm2_flops, r.gemm2_ms) << "/" << ComputeTFLOPS(r.total_flops, r.total_ms);
        std::cout << std::setw(kCellW) << oss.str();
      } else {
        std::cout << std::setw(kCellW) << "N/A";
      }
    }
    std::cout << std::endl;
  }
}

// ===========================================================================
// Print a single Bandwidth table with format "g1/g2/total" per cell (GB/s).
// ===========================================================================
void PrintBandwidth(const std::string& config_name, const std::vector<std::vector<BenchResult>>& all_results) {
  static constexpr int kCellW = 24;  // "xx.xx/xx.xx/xx.xx" fits in 24 chars

  // Header
  std::cout << "\n=== " << config_name << " Bandwidth (GB/s, GEMM1 / GEMM2 / Total) ===" << std::endl;
  std::cout << " " << std::setw(10) << "tokens";
  std::cout << std::setw(kCellW) << "Marlin";
  std::cout << std::setw(kCellW) << "HummingW4A16";
  std::cout << std::setw(kCellW) << "HummingW4A8";
  std::cout << std::endl;

  for (size_t i = 0; i < kTokenSizes.size(); ++i) {
    std::cout << " " << std::setw(10) << kTokenSizes[i];
    for (size_t b = 0; b < all_results.size(); ++b) {
      if (i < all_results[b].size()) {
        const auto& r = all_results[b][i];
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(0) << ComputeBandwidthGBps(r.gemm1_bytes, r.gemm1_ms) << "/"
            << ComputeBandwidthGBps(r.gemm2_bytes, r.gemm2_ms) << "/"
            << ComputeBandwidthGBps(r.total_bytes, r.total_ms);
        std::cout << std::setw(kCellW) << oss.str();
      } else {
        std::cout << std::setw(kCellW) << "N/A";
      }
    }
    std::cout << std::endl;
  }
}

// ===========================================================================
// Print JIT-compiled kernel table (humming only).
// Lists all kernel variants compiled by HummingJitCompiler for this process.
// Called once per config, after the performance tables.
// ===========================================================================
void PrintJitKernelTable(const std::string& config_name, const Fp4MoeTestConfig& cfg) {
  const auto* table = HummingFp4MoeLayer::GetJitKernelTable();
  if (table == nullptr || table->empty()) {
    std::cout << "\n=== " << config_name << " JIT Kernel Table: EMPTY (no JIT compiler) ===" << std::endl;
    return;
  }

  // Problem shape header
  size_t E = ComputeExpertsPerRank(cfg);
  size_t N = ComputeInterPerRank(cfg);
  size_t K = cfg.hidden_size;
  std::cout << "\n=== " << config_name << " Problem Shapes ===" << std::endl;
  std::cout << "  experts_per_rank=" << E << "  topk=" << 6 << "  hidden=" << K << "  inter_per_rank=" << N
            << std::endl;
  std::cout << "  GEMM1: N=" << N * 2 << "  K=" << K << std::endl;
  std::cout << "  GEMM2: N=" << K << "  K=" << N << std::endl;

  std::cout << "\n=== " << config_name << " JIT Kernel Table ===" << std::endl;
  // Header
  std::cout << " " << std::setw(6) << "dtype" << std::setw(8) << "N" << std::setw(8) << "K" << std::setw(16)
            << "blkMxNxK" << std::setw(6) << "stg" << std::setw(6) << "cta" << std::setw(14) << "TMA(ABCS)"
            << std::setw(6) << "SK" << std::setw(6) << "WS" << std::setw(6) << "E8M0" << std::setw(22)
            << "shapeM[min..max]" << std::endl;

  // Sort by (compute_dtype, size_n) for readable output
  std::vector<llm_kernels::nvidia::humming::KernelMeta> sorted_table(table->begin(), table->end());
  std::sort(sorted_table.begin(), sorted_table.end(),
            [](const llm_kernels::nvidia::humming::KernelMeta& a, const llm_kernels::nvidia::humming::KernelMeta& b) {
              if (a.compute_dtype != b.compute_dtype) return a.compute_dtype < b.compute_dtype;
              return a.size_n < b.size_n;
            });

  for (const auto& m : sorted_table) {
    std::string dtype_str = (m.compute_dtype == llm_kernels::nvidia::humming::ComputeDtype::FP8) ? "FP8" : "BF16";
    std::string blk_str = std::to_string(m.block_m) + "x" + std::to_string(m.block_n) + "x" + std::to_string(m.block_k);
    std::string tma_str = std::string(m.use_tma_a ? "1" : "0") + std::string(m.use_tma_b ? "1" : "0") +
                          std::string(m.use_tma_c ? "1" : "0") + std::string(m.use_tma_bs ? "1" : "0");
    std::string shape_str = "[" + std::to_string(m.min_shape_m) + ".." + std::to_string(m.max_shape_m) + "]";

    std::cout << " " << std::setw(6) << dtype_str << std::setw(8) << m.size_n << std::setw(8) << m.size_k
              << std::setw(16) << blk_str << std::setw(6) << m.num_stages << std::setw(6) << m.num_ctas_per_sm
              << std::setw(14) << tma_str << std::setw(6) << (m.use_stream_k ? "Y" : "N") << std::setw(6)
              << (m.use_warp_spec ? "Y" : "N") << std::setw(6) << (m.use_fused_e8m0_scale ? "Y" : "N") << std::setw(22)
              << shape_str << std::endl;
  }
}

// ===========================================================================
// Print selected kernel schedule (humming only).
// For each token size, shows the actual KernelMeta selected at runtime for
// GEMM1 and GEMM2 by SelectGemmMeta (profiled right-align or heuristic fallback).
// ===========================================================================
void PrintSelectedKernels(const std::string& config_name, const std::vector<std::vector<BenchResult>>& all_results) {
  // Backend indices: 0=Marlin (skip), 1=HummingW4A16, 2=HummingW4A8
  const char* humming_names[] = {"HummingW4A16", "HummingW4A8"};
  const int humming_indices[] = {1, 2};

  for (int hi = 0; hi < 2; ++hi) {
    int b = humming_indices[hi];
    if (b >= static_cast<int>(all_results.size())) continue;

    std::cout << "\n=== " << config_name << " Selected Kernels — " << humming_names[hi] << " ===" << std::endl;
    std::cout << " " << std::setw(10) << "tokens" << std::setw(36) << "GEMM1" << std::setw(36) << "GEMM2" << std::endl;

    for (size_t i = 0; i < kTokenSizes.size(); ++i) {
      if (i >= all_results[b].size()) break;
      const auto& r = all_results[b][i];

      auto format_meta = [](const llm_kernels::nvidia::humming::KernelMeta* m) -> std::string {
        if (m == nullptr) return "nullptr(heuristic)";
        std::ostringstream oss;
        oss << m->block_m << "x" << m->block_n << "x" << m->block_k << " S" << m->num_stages << " C"
            << m->num_ctas_per_sm << (m->use_stream_k ? " [SK]" : "");
        return oss.str();
      };

      std::cout << " " << std::setw(10) << kTokenSizes[i] << std::setw(36)
                << format_meta(r.has_gemm1_meta ? &r.gemm1_meta : nullptr) << std::setw(36)
                << format_meta(r.has_gemm2_meta ? &r.gemm2_meta : nullptr) << std::endl;
    }
  }
}

// ===========================================================================
// Print non-GEMM kernel breakdown (humming only).
// For each token size, shows per-phase times: gating, align, quant, silu,
// mulsum, other (residual), nogemm (total-gemm1-gemm2), nogemm%.
// other = nogemm - (gating + align + quant + silu + mulsum) — captures memsets + gaps.
// ===========================================================================
void PrintHummingNonGemmBreakdown(const std::string& config_name,
                                  const std::vector<std::vector<BenchResult>>& all_results) {
  const char* humming_names[] = {"HummingW4A16", "HummingW4A8"};
  const int humming_indices[] = {1, 2};

  for (int hi = 0; hi < 2; ++hi) {
    int b = humming_indices[hi];
    if (b >= static_cast<int>(all_results.size())) continue;

    std::cout << "\n=== " << config_name << " Non-GEMM Breakdown (ms) — " << humming_names[hi] << " ===" << std::endl;
    std::cout << " " << std::setw(10) << "tokens" << std::setw(10) << "gating" << std::setw(10) << "align"
              << std::setw(10) << "quant" << std::setw(10) << "silu" << std::setw(10) << "mulsum" << std::setw(10)
              << "other" << std::setw(10) << "nogemm" << std::setw(10) << "nogemm%" << std::endl;

    for (size_t i = 0; i < kTokenSizes.size(); ++i) {
      if (i >= all_results[b].size()) break;
      const auto& r = all_results[b][i];

      float measured = r.gating_ms + r.align_ms + r.quant_ms + r.silu_ms + r.mulsum_ms;
      float other = r.nogemm_ms - measured;
      float nogemm_pct = (r.total_ms > 0.f) ? (r.nogemm_ms / r.total_ms * 100.f) : 0.f;

      std::cout << " " << std::setw(10) << kTokenSizes[i] << std::fixed << std::setprecision(4) << std::setw(10)
                << r.gating_ms << std::setw(10) << r.align_ms << std::setw(10) << r.quant_ms << std::setw(10)
                << r.silu_ms << std::setw(10) << r.mulsum_ms << std::setw(10) << other << std::setw(10) << r.nogemm_ms
                << std::setw(9) << std::setprecision(1) << nogemm_pct << "%" << std::endl;
    }
  }
}

// ===========================================================================
// Cosine similarity helper for flattened tensors
// ===========================================================================
inline float CosineSim(const torch::Tensor& a, const torch::Tensor& b) {
  return torch::nn::functional::cosine_similarity(a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item<float>();
}

// ===========================================================================
// Full benchmark + accuracy driver for one config.
//
// Serialized design: base weights + max-size input data are generated once,
// then each backend is tested independently and its resources freed before
// the next backend starts.  After perf measurement, dequantized FP32 weights
// are computed for accuracy reference (TP configs only).
//
// Uses accuracy-friendly data (constrained E8M0 scales [119,123), swiglu_limit=10)
// so that GPU reference comparison can run alongside performance measurement.
// Accuracy verification is skipped for EP configs (expert_map remapping not
// modeled in the GPU reference).
// ===========================================================================
void RunBenchmarkForConfig(const Fp4MoeTestConfig& cfg, std::shared_ptr<Context> context,
                           RuntimeConfig& base_runtime_config) {
  if (context->device->ext->GetComputeCapacity() != 90) {
    GTEST_SKIP() << "Requires SM90+";
  }

  // QUANT_PROFILE controls profiling in cutlass/humming MoE layers.
  // If the user has set it externally, respect their choice (enables profiling tests).
  // Otherwise default to 0 (profiling disabled) for fast, deterministic baseline tests.
  if (std::getenv("QUANT_PROFILE") == nullptr) {
    setenv("QUANT_PROFILE", "0", 1);
  }

  constexpr int kDeviceRank = 0;
  cudaStream_t stream = GetStream(context);
  size_t K = cfg.hidden_size;
  size_t E = cfg.num_experts;
  size_t E_local = ComputeExpertsPerRank(cfg);
  size_t N = ComputeInterPerRank(cfg);
  size_t max_ntok = kTokenSizes.back();
  const bool is_ep = (cfg.expert_para_size > 1);

  // Accuracy constants (applied to all configs; EP skips comparison)
  constexpr float kSwigluLimit = 10.0f;
  constexpr int kScaleLo = 119;
  constexpr int kScaleHi = 123;
  constexpr float kCosSimBF16 = 0.95f;  // Marlin & W4A16 (BF16 compute)
  constexpr float kCosSimFP8 = 0.90f;   // W4A8 (FP8 quant residual)

  // Configure EP so layers properly simulate expert parallelism:
  // sets global_expert_para_size_ = ep_size * expert_world_size, triggers EP
  // routing (ExpertMap + InvokeMoeAlignBlockSize with EP flag) in Forward.
  base_runtime_config.parallel_basic_config.expert_parallel_size = cfg.expert_para_size;

  // =========================================================================
  // 1. Generate base weights once (seed 777, constrained scales for accuracy)
  // =========================================================================
  torch::manual_seed(777);
  torch::cuda::manual_seed(777);
  ProcessedWeights base = GenerateBaseWeights(cfg, kScaleLo, kScaleHi);

  // =========================================================================
  // 2. Pre-generate max-size input data (seed 888).
  //    All backends slice from these, guaranteeing identical inputs per M.
  // =========================================================================
  torch::manual_seed(888);
  torch::cuda::manual_seed(888);
  auto bf16_opt = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kBFloat16);
  torch::Tensor hs_full = torch::randn({static_cast<int64_t>(max_ntok), static_cast<int64_t>(K)}, bf16_opt);
  torch::Tensor gate_full = torch::randn({static_cast<int64_t>(max_ntok), static_cast<int64_t>(E)}, bf16_opt);

  const void* hs_ptr = hs_full.data_ptr();
  const void* gate_ptr = gate_full.data_ptr();

  // =========================================================================
  // 3. Shared params (swiglu_limit for accuracy; perf unaffected)
  // =========================================================================
  auto params = BuildMoeParams(cfg, max_ntok, kSwigluLimit);

  std::vector<std::vector<BenchResult>> all_results;
  std::vector<std::vector<torch::Tensor>> all_outputs;  // [backend][token_size]

  // =========================================================================
  // 4-6. Run each backend (serialized)
  // =========================================================================
  auto run_backend = [&](Fp4Backend backend) {
    ProcessedWeights pw;
    switch (backend) {
      case Fp4Backend::kMarlin:
        pw = ProcessWeightsMarlin(base, cfg, stream);
        break;
      case Fp4Backend::kHummingW4A16:
        pw = ProcessWeightsHummingW4A16(base, cfg, stream);
        break;
      case Fp4Backend::kHummingW4A8:
        pw = ProcessWeightsHummingW4A8(base, cfg, stream);
        break;
      default:
        KLLM_THROW(fmt::format("Unsupported backend: {}", static_cast<int>(backend)));
    }
    StreamSynchronize(context->device->GetPrimaryComputeStream());

    RuntimeConfig rc = base_runtime_config;
    rc.inter_data_type = TYPE_BF16;
    auto layer = CreateLayer(backend, params, rc, context, kDeviceRank);

    auto inputs = SetupPersistentInputs(pw, backend, stream);
    cudaStreamSynchronize(stream);

    if (backend == Fp4Backend::kMarlin) {
      auto [results, outputs] =
          RunBenchmark(static_cast<MarlinFp4MoeLayer*>(layer.get()), inputs, cfg, stream, hs_ptr, gate_ptr);
      all_results.push_back(std::move(results));
      all_outputs.push_back(std::move(outputs));
    } else {
      auto [results, outputs] =
          RunBenchmark(static_cast<HummingFp4MoeLayer*>(layer.get()), inputs, cfg, stream, hs_ptr, gate_ptr);
      all_results.push_back(std::move(results));
      all_outputs.push_back(std::move(outputs));
    }
  };

  run_backend(Fp4Backend::kMarlin);
  run_backend(Fp4Backend::kHummingW4A16);
  run_backend(Fp4Backend::kHummingW4A8);

  // =========================================================================
  // 7. Print performance results
  // =========================================================================
  PrintLatencyTable(cfg.name, all_results);
  if (!is_ep) {
    PrintTFLOPS(cfg.name, all_results);
    PrintBandwidth(cfg.name, all_results);
  } else {
    std::cout << "\n=== " << cfg.name << " TFLOPS: SKIPPED (EP — per-rank FLOPs depend on "
              << "token-expert distribution)" << std::endl;
  }

  // Humming-only detailed analysis (printed for all configs, TP and EP)
  PrintJitKernelTable(cfg.name, cfg);
  PrintSelectedKernels(cfg.name, all_results);
  PrintHummingNonGemmBreakdown(cfg.name, all_results);

  // =========================================================================
  // 8. Accuracy verification (TP configs only)
  //    All three backends independently compared against GPU torch reference.
  // =========================================================================
  if (is_ep) {
    return;
  }

  // Dequant FP4 weights to FP32 on GPU for accuracy reference (after backend
  // runs to avoid increasing peak memory during perf measurement)
  torch::Tensor w1_dequant = DequantFp4WeightGpu(base.w1, base.w1_scales, E_local, N * 2, K);  // [E, N*2, K]
  torch::Tensor w2_dequant = DequantFp4WeightGpu(base.w2, base.w2_scales, E_local, K, N);      // [E, K, N]

  // all_outputs indices: 0=Marlin, 1=W4A16, 2=W4A8
  const struct {
    const char* name;
    float threshold;
  } backend_info[] = {{"Marlin", kCosSimBF16}, {"HummingW4A16", kCosSimBF16}, {"HummingW4A8", kCosSimFP8}};

  for (size_t i = 0; i < kTokenSizes.size(); ++i) {
    size_t M = kTokenSizes[i];
    auto hs_m = hs_full.slice(0, 0, static_cast<int64_t>(M));
    auto gate_m = gate_full.slice(0, 0, static_cast<int64_t>(M));

    // BF16 reference (for Marlin and W4A16)
    auto ref_bf16 = GpuRefFp4MoeForward(w1_dequant, w2_dequant, cfg, hs_m, gate_m, Fp4Backend::kMarlin, kSwigluLimit);
    // FP8 reference (for W4A8)
    auto ref_fp8 =
        GpuRefFp4MoeForward(w1_dequant, w2_dequant, cfg, hs_m, gate_m, Fp4Backend::kHummingW4A8, kSwigluLimit);

    torch::Tensor refs[3] = {ref_bf16, ref_bf16, ref_fp8};

    for (int b = 0; b < 3; ++b) {
      float cos = CosineSim(all_outputs[b][i], refs[b]);
      bool has_nan = refs[b].isnan().any().item<bool>() || all_outputs[b][i].isnan().any().item<bool>();
      EXPECT_TRUE(cos > backend_info[b].threshold && !has_nan)
          << cfg.name << " " << backend_info[b].name << " M=" << M << " cos_sim=" << cos << " (threshold "
          << backend_info[b].threshold << ")";
    }
  }
}

// NOTE(jinxcwu) 完整性能测试时可以打开所有情况的测试，目前为了节省测试时间只打开一个
TEST_F(LayerTest, Flash_TP4) { RunBenchmarkForConfig(kFlashTP4, context_, runtime_config); }
// TEST_F(LayerTest, Flash_EP4) { RunBenchmarkForConfig(kFlashEP4, context_, runtime_config); }
// TEST_F(LayerTest, Pro_TP8) { RunBenchmarkForConfig(kProTP8, context_, runtime_config); }
// TEST_F(LayerTest, Pro_EP8) { RunBenchmarkForConfig(kProEP8, context_, runtime_config); }

}  // namespace ksana_llm

#endif  // ENABLE_CUDA
