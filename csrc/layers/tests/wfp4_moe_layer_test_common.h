/* Copyright 2026 Tencent Inc.  All rights reserved.
==============================================================================*/
#pragma once

// FP4 MoE test — shared test infrastructure: backend metadata table +
// FP4_BACKENDS env parsing, stream helper, ProcessedWeights container,
// base-weight generation, per-backend weight processing (marlin, humming
// w4a16, humming w4a8), tensor deep-copy, Init-param builder, layer factory,
// and persistent-input setup.
//
// Expected to be included after the layer/kernel/torch headers from
// wfp4_moe_layer_test.cpp's ENABLE_CUDA block.

#include "layers/base_moe_layer.h"
#include "wfp4_moe_layer_test_config.h"

namespace ksana_llm {

// ===========================================================================
// Backend metadata table — single source of truth for backend specs.
// Consumed by ParseRequestedBackends (below) and referenced wherever the full
// canonical backend set is needed.
// ===========================================================================
static const std::vector<BackendSpec>& MasterBackendSpecs() {
  static const std::vector<BackendSpec> kMaster = {
      {Fp4Backend::kMarlin, "Marlin", "marlin", nullptr, 0.95f, false, false},
      {Fp4Backend::kHummingW4A16, "HummingW4A16", "humming_w4a16", "w4a16", 0.95f, true, false},
      {Fp4Backend::kHummingW4A8, "HummingW4A8", "humming_w4a8", "w4a8", 0.90f, true, true},
  };
  return kMaster;
}

// ===========================================================================
// Parse FP4_BACKENDS env var into an ordered, deduped list of BackendSpec.
// Unset/empty → all backends (canonical order). Unknown names → warn + skip.
// Matching is exact, case-insensitive against env_name or alias.
// ===========================================================================
std::vector<BackendSpec> ParseRequestedBackends() {
  const char* env = std::getenv("FP4_BACKENDS");
  if (env == nullptr || env[0] == '\0') {
    return MasterBackendSpecs();  // default: all backends
  }

  // Helper: lowercase-trim a token and match against a spec's names.
  auto match = [](const std::string& token, const BackendSpec& spec) {
    std::string t = token;
    // trim whitespace
    t.erase(std::remove_if(t.begin(), t.end(), ::isspace), t.end());
    std::transform(t.begin(), t.end(), t.begin(), ::tolower);
    if (t == spec.env_name) return true;
    if (spec.alias != nullptr && t == spec.alias) return true;
    return false;
  };

  std::vector<BackendSpec> selected;
  // Split on commas
  std::string s(env);
  size_t start = 0;
  while (start <= s.size()) {
    size_t comma = s.find(',', start);
    std::string token = (comma == std::string::npos) ? s.substr(start) : s.substr(start, comma - start);
    bool found = false;
    for (const auto& spec : MasterBackendSpecs()) {
      if (match(token, spec)) {
        // dedup: skip if already selected
        bool already = false;
        for (const auto& sel : selected) {
          if (sel.backend == spec.backend) {
            already = true;
            break;
          }
        }
        if (!already) {
          selected.push_back(spec);
        }
        found = true;
        break;
      }
    }
    if (!found) {
      KLLM_LOG_WARNING << fmt::format("FP4_BACKENDS: unknown backend '{}', skipping", token);
    }
    if (comma == std::string::npos) break;
    start = comma + 1;
  }

  if (selected.empty()) {
    KLLM_LOG_WARNING << "FP4_BACKENDS: no valid backends parsed, falling back to all";
    return MasterBackendSpecs();
  }
  return selected;
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
  size_t K = cfg.hidden_size();

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
  using llm_kernels::nvidia::marlin_moe_wna16::marlin_moe_permute_scales;
  using llm_kernels::nvidia::marlin_moe_wna16::marlin_moe_repack;
  using llm_kernels::nvidia::marlin_moe_wna16::marlin_moe_repack_meta;
  using llm_kernels::nvidia::marlin_moe_wna16::mxfp4_process_scales;
  using llm_kernels::nvidia::marlin_moe_wna16::transpose_2d_batched;
  size_t E = ComputeExpertsPerRank(cfg);
  size_t N = ComputeInterPerRank(cfg);
  size_t K = cfg.hidden_size();
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
  using llm_kernels::nvidia::humming::ComputeDtype;
  using llm_kernels::nvidia::humming::process_scales;
  using llm_kernels::nvidia::humming::repack_weight_wgmma;
  size_t E = ComputeExpertsPerRank(cfg);
  size_t N = ComputeInterPerRank(cfg);
  size_t K = cfg.hidden_size();
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
  using llm_kernels::nvidia::humming::ComputeDtype;
  using llm_kernels::nvidia::humming::process_mxfp4_w4a8_weight;
  using llm_kernels::nvidia::humming::process_scales;
  using llm_kernels::nvidia::humming::repack_weight_wgmma;
  size_t E = ComputeExpertsPerRank(cfg);
  size_t N = ComputeInterPerRank(cfg);
  size_t K = cfg.hidden_size();
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
// Build MoeLayerParameters Init struct
// ===========================================================================
std::vector<std::any> BuildMoeParams(const Fp4MoeTestConfig& cfg, size_t max_token_num) {
  size_t inter_per_rank = ComputeInterPerRank(cfg);
  size_t experts_per_rank = ComputeExpertsPerRank(cfg);
  size_t moe_tp = ComputeMoeTp(cfg);

  MoeLayerParameters p;
  p.moe_scale_norm_mode = MoeScaleNormMode::NO_NORM;
  p.max_token_num = max_token_num;
  p.layer_idx = 0;
  p.expert_num_per_node = experts_per_rank;
  p.expert_hidden_size = cfg.hidden_size();
  p.expert_inter_size = inter_per_rank;
  p.expert_topk = cfg.topk();
  p.tp_size = moe_tp;
  p.use_vllm_moe = true;
  p.num_expert_group = 1;
  p.expert_groups_topk = 1;
  p.scoring_func = std::string(cfg.scoring_func());
  p.topk_method = std::string("noaux_tc");
  p.norm_topk_prob = true;
  p.routed_scaling_factor = cfg.routed_scaling_factor();
  p.use_e_score_correction_bias = true;
  p.enable_full_shared_expert = false;
  p.fp8_weight_dtype = DataType::TYPE_INVALID;
  p.int_weight_dtype = DataType::TYPE_FP4_E2M1FN_x2;
  p.group_size = 128;
  p.apply_weight = false;
  p.swiglu_limit = cfg.swiglu_limit();
  p.num_hash_layers = 0;
  p.vocab_size = 0;
  p.situ_beta = cfg.situ_beta();
  p.situ_linear_beta = cfg.situ_linear_beta();
  return {p};
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
      layer = std::make_unique<MarlinFusedWfp4A16MoeLayer>();
      break;
    case Fp4Backend::kHummingW4A16:
      layer = std::make_unique<HummingFusedWfp4A16MoeLayer>();
      break;
    case Fp4Backend::kHummingW4A8:
      layer = std::make_unique<HummingFusedWfp4Afp8MoeLayer>();
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

}  // namespace ksana_llm
