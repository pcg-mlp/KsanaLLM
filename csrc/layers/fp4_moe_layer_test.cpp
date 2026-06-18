/* Copyright 2026 Tencent Inc.  All rights reserved.
==============================================================================*/
#include <cmath>
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
ProcessedWeights GenerateBaseWeights(const Fp4MoeTestConfig& cfg) {
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
      .random_(0, 256);

  // w2 (down): [E, K, N/2] uint8 — FP4 packed
  base.w2 = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_UINT8, {E, K, N / 2});
  torch::from_blob(base.w2.GetPtr<void>(),
                   {static_cast<int64_t>(E), static_cast<int64_t>(K), static_cast<int64_t>(N / 2)}, uint8_opt)
      .random_(0, 256);

  // w2_scales: [E, K, N/32] uint8 — E8M0 format
  base.w2_scales = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_UINT8, {E, K, N / 32});
  torch::from_blob(base.w2_scales.GetPtr<void>(),
                   {static_cast<int64_t>(E), static_cast<int64_t>(K), static_cast<int64_t>(N / 32)}, uint8_opt)
      .random_(0, 256);

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
                 static_cast<int64_t>(N), 128, ComputeDtype::BF16);
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
    process_scales(stream, s1_copy.GetPtr<void>(), out.w1_scales.GetPtr<void>(), static_cast<int64_t>(E), N1, K1, 128,
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
                   static_cast<int64_t>(N), 128, ComputeDtype::FP8);
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
// Build 24-parameter Init list
// ===========================================================================
std::vector<std::any> BuildMoeParams(const Fp4MoeTestConfig& cfg, size_t max_token_num) {
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
  params.push_back(0.0f);                          // 21: swiglu_limit
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
  ComputeFlops(cfg, num_tokens, br.gemm1_flops, br.gemm2_flops, br.total_flops);
  ComputeBytes(cfg, num_tokens, br.gemm1_bytes, br.gemm2_bytes, br.total_bytes);

  return br;
}

// ===========================================================================
// Run benchmark across all token sizes for one (layer, inputs) pair.
// Templated on the concrete layer type so we can call SetPerfStats directly.
// ===========================================================================
template <typename LayerT>
std::vector<BenchResult> RunBenchmark(LayerT* layer, std::vector<Tensor>& persistent_inputs,
                                      const Fp4MoeTestConfig& cfg, cudaStream_t stream, const void* hs_full,
                                      const void* gate_full) {
  static constexpr size_t kBytesPerBF16 = 2;
  size_t K = cfg.hidden_size;
  size_t E = cfg.num_experts;

  MoePerfStats perf_stats;
  layer->SetPerfStats(&perf_stats);

  std::vector<BenchResult> results;
  results.reserve(kTokenSizes.size());

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
  }

  layer->SetPerfStats(nullptr);
  return results;
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
// Print latency table.  Marlin is always present as the accuracy reference.
// ===========================================================================
void PrintLatencyTable(const std::string& config_name, const std::vector<std::vector<BenchResult>>& all_results) {
  std::cout << "\n=== " << config_name << " Latency (ms) ===" << std::endl;
  std::cout << " " << std::setw(10) << "tokens";
  std::cout << std::setw(16) << "Marlin";
  std::cout << std::setw(18) << "HummingW4A16";
  std::cout << std::setw(18) << "HummingW4A8";
  std::cout << std::endl;

  for (size_t i = 0; i < kTokenSizes.size(); ++i) {
    std::cout << " " << std::setw(10) << kTokenSizes[i];
    for (size_t b = 0; b < all_results.size(); ++b) {
      if (i < all_results[b].size()) {
        std::cout << std::setw(16) << std::fixed << std::setprecision(4) << all_results[b][i].total_ms;
      } else {
        std::cout << std::setw(16) << "N/A";
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
// Full benchmark driver for one config.
//
// Serialized design: base weights + max-size input data are generated once,
// then each backend is tested independently and its resources freed before
// the next backend starts.  Peak GPU memory = base weights + one backend's
// weights (~6 GB).
// ===========================================================================
void RunBenchmarkForConfig(const Fp4MoeTestConfig& cfg, std::shared_ptr<Context> context,
                           RuntimeConfig& base_runtime_config) {
  if (context->device->ext->GetComputeCapacity() != 90) {
    GTEST_SKIP() << "Requires SM90+";
  }

  constexpr int kDeviceRank = 0;
  cudaStream_t stream = GetStream(context);
  size_t K = cfg.hidden_size;
  size_t E = cfg.num_experts;
  size_t max_ntok = kTokenSizes.back();

  // Configure EP so layers properly simulate expert parallelism:
  // sets global_expert_para_size_ = ep_size * expert_world_size, triggers EP
  // routing (ExpertMap + InvokeMoeAlignBlockSize with EP flag) in Forward.
  base_runtime_config.parallel_basic_config.expert_parallel_size = cfg.expert_para_size;

  // =========================================================================
  // 1. Generate base weights once (seed 123, shared across all backends)
  // =========================================================================
  torch::manual_seed(123);
  torch::cuda::manual_seed(123);
  ProcessedWeights base = GenerateBaseWeights(cfg);

  // =========================================================================
  // 2. Pre-generate max-size input data (seed 42).
  //    All backends slice from these, guaranteeing identical inputs per M.
  // =========================================================================
  torch::manual_seed(42);
  torch::cuda::manual_seed(42);
  auto bf16_opt = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kBFloat16);
  torch::Tensor hs_full = torch::randn({static_cast<int64_t>(max_ntok), static_cast<int64_t>(K)}, bf16_opt);
  torch::Tensor gate_full = torch::rand({static_cast<int64_t>(max_ntok), static_cast<int64_t>(E)}, bf16_opt);

  const void* hs_ptr = hs_full.data_ptr();
  const void* gate_ptr = gate_full.data_ptr();

  // =========================================================================
  // 3. Shared params
  // =========================================================================
  auto params = BuildMoeParams(cfg, max_ntok);

  std::vector<std::vector<BenchResult>> all_results;

  // =========================================================================
  // 4. Marlin
  // =========================================================================
  {
    ProcessedWeights pw = ProcessWeightsMarlin(base, cfg, stream);
    StreamSynchronize(context->device->GetPrimaryComputeStream());

    RuntimeConfig rc = base_runtime_config;
    rc.inter_data_type = TYPE_BF16;
    auto layer = CreateLayer(Fp4Backend::kMarlin, params, rc, context, kDeviceRank);

    auto inputs = SetupPersistentInputs(pw, Fp4Backend::kMarlin, stream);
    cudaDeviceSynchronize();
    all_results.push_back(
        RunBenchmark(static_cast<MarlinFp4MoeLayer*>(layer.get()), inputs, cfg, stream, hs_ptr, gate_ptr));
  }  // ~pw, ~layer, ~inputs

  // =========================================================================
  // 5. HummingW4A16
  // =========================================================================
  {
    ProcessedWeights pw = ProcessWeightsHummingW4A16(base, cfg, stream);
    StreamSynchronize(context->device->GetPrimaryComputeStream());

    RuntimeConfig rc = base_runtime_config;
    rc.inter_data_type = TYPE_BF16;
    auto layer = CreateLayer(Fp4Backend::kHummingW4A16, params, rc, context, kDeviceRank);

    auto inputs = SetupPersistentInputs(pw, Fp4Backend::kHummingW4A16, stream);
    cudaDeviceSynchronize();
    all_results.push_back(
        RunBenchmark(static_cast<HummingFp4MoeLayer*>(layer.get()), inputs, cfg, stream, hs_ptr, gate_ptr));
  }

  // =========================================================================
  // 6. HummingW4A8
  // =========================================================================
  {
    ProcessedWeights pw = ProcessWeightsHummingW4A8(base, cfg, stream);
    StreamSynchronize(context->device->GetPrimaryComputeStream());

    RuntimeConfig rc = base_runtime_config;
    rc.inter_data_type = TYPE_BF16;
    auto layer = CreateLayer(Fp4Backend::kHummingW4A8, params, rc, context, kDeviceRank);

    auto inputs = SetupPersistentInputs(pw, Fp4Backend::kHummingW4A8, stream);
    cudaDeviceSynchronize();
    all_results.push_back(
        RunBenchmark(static_cast<HummingFp4MoeLayer*>(layer.get()), inputs, cfg, stream, hs_ptr, gate_ptr));
  }

  // =========================================================================
  // 7. Print results
  // =========================================================================
  PrintLatencyTable(cfg.name, all_results);

  // TFLOPS are only meaningful for TP-only configs. Under EP the current rank
  // processes only tokens routed to its local expert subset, so the static
  // 2*M*topk*K*N formula overestimates actual work.
  const bool is_ep = (cfg.expert_para_size > 1);
  if (!is_ep) {
    PrintTFLOPS(cfg.name, all_results);
    PrintBandwidth(cfg.name, all_results);
  } else {
    std::cout << "\n=== " << cfg.name << " TFLOPS: SKIPPED (EP — per-rank FLOPs depend on "
              << "token-expert distribution)" << std::endl;
  }
}  // TEST_F cases
// ===========================================================================

// NOTE(jinxcwu) 完整性能测试时可以打开所有情况的测试，目前为了节省测试时间只打开一个
TEST_F(LayerTest, Flash_TP4) { RunBenchmarkForConfig(kFlashTP4, context_, runtime_config); }
// TEST_F(LayerTest, Flash_EP4) { RunBenchmarkForConfig(kFlashEP4, context_, runtime_config); }
// TEST_F(LayerTest, Pro_TP8) { RunBenchmarkForConfig(kProTP8, context_, runtime_config); }
// TEST_F(LayerTest, Pro_EP8) { RunBenchmarkForConfig(kProEP8, context_, runtime_config); }

}  // namespace ksana_llm

#endif  // ENABLE_CUDA
