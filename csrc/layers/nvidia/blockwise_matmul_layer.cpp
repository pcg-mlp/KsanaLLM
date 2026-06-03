/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#ifdef ENABLE_FP8
#  include "layers/blockwise_matmul_layer.h"

#  include <fmt/ranges.h>

#  include "kernels/nvidia/kernel_wrapper.h"
#  include "profiler/timer.h"
#  include "utils/singleton.h"
#  include "utils/utils.h"

namespace ksana_llm {

Status BlockwiseMatMulLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                  std::shared_ptr<Context> context, int rank) {
  STATUS_CHECK_FAILURE(BaseLayer::Init(parameters, runtime_config, context, rank));

  size_t parameter_index = 0;
  max_m_ = std::any_cast<size_t>(parameters[parameter_index++]);
  n_ = std::any_cast<size_t>(parameters[parameter_index++]);
  k_ = std::any_cast<size_t>(parameters[parameter_index++]);
  block_size_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  weight_ = std::any_cast<Tensor>(parameters[parameter_index++]);
  if (parameters.size() >= parameter_index + 2) {
    skip_quant_ = std::any_cast<const bool>(parameters[parameter_index++]);
    swiglu_limit_ = std::any_cast<float>(parameters[parameter_index++]);
  }

  KLLM_CHECK_WITH_INFO(max_m_ % kAlignSize_ == 0,
                       fmt::format("max_m {} is not aligned to {}, please set it to a multiple of {}", max_m_,
                                   kAlignSize_, kAlignSize_));

  // currently, DeepGEMM only support bfloat16
  deepgemm_enabled_ = ((inter_data_type_ == DataType::TYPE_BF16) && std::getenv("DISABLE_DEEPGEMM") == nullptr);
  if (deepgemm_enabled_) {
    trtllm_deepgemm_wrapper_ = std::make_shared<llm_kernels::nvidia::DeepGEMMWrapper>(rank);
    bridge_deepgemm_wrapper_ = ksana_llm::nvidia::DeepSeekDeepGEMMWrapper::GetInstance();
  }

  KLLM_LOG_INFO << fmt::format("Rank[{}] BlockwiseMatMulLayer Init done, n={}, k={}", rank_, n_, k_);
  return Status();
}

size_t BlockwiseMatMulLayer::GetWorkspaceSize() { DISPATCH_BY_3_DTYPE(inter_data_type_, GetWorkspaceSizeT); }

template <typename T>
size_t BlockwiseMatMulLayer::GetCachedCutlassBufferSize() {
  auto cache = Singleton<std::unordered_map<std::string, size_t>>::GetInstance();
  std::string key = fmt::format("{}_{}", k_, n_);
  auto it = cache->find(key);
  if (it != cache->end()) {
    KLLM_LOG_DEBUG << fmt::format("Rank[{}] Using cached cutlass buffer size: {}", rank_, it->second);
    return it->second;
  }

  KLLM_LOG_DEBUG << fmt::format("Rank[{}] Computing cutlass buffer size for max_m={}, k={}, n={}", rank_, max_m_, k_,
                                n_);
  size_t cutlass_buffer_size = 0;
  for (size_t m = kAlignSize_; m <= max_m_; m += kAlignSize_) {
    cutlass_buffer_size = std::max(cutlass_buffer_size, InvokeGetBlockGemmWorkspaceSize<T>(m, k_, n_));
  }
  (*cache)[key] = cutlass_buffer_size;
  KLLM_LOG_DEBUG << fmt::format("Rank[{}] Cached cutlass buffer size: {}", rank_, cutlass_buffer_size);
  return cutlass_buffer_size;
}

Status BlockwiseMatMulLayer::Preprocess(const ModelConfig& model_config_, const RuntimeConfig& runtime_config) {
  if (!deepgemm_enabled_) {
    SetGemmType(BlockwiseMatMulLayer::Fp8GemmType::Cutlass);
    return Status();
  }

  const auto start_time = ProfileTimer::GetCurrentTimeInMs();
  const size_t kPerfIters = GetEnvAsPositiveInt("QUANT_PROFILE", 20);
  if (kPerfIters == 0) {
    SetGemmType(BlockwiseMatMulLayer::Fp8GemmType::Cutlass);
    return Status();
  }
  const size_t kWarmupIters = std::max(1UL, kPerfIters / 2);

  static std::mutex g_mtx;
  std::lock_guard<std::mutex> guard(g_mtx);

  auto sel_cache = Singleton<std::unordered_map<std::string, GemmTuningResult>>::GetInstance();
  std::string sel_key = fmt::format("{}_{}_{}_{}", static_cast<int>(inter_data_type_), max_m_, k_, n_);
  auto sel_it = sel_cache->find(sel_key);
  if (sel_it != sel_cache->end()) {
    search_cache_ = sel_it->second;
    KLLM_LOG_DEBUG << fmt::format("Reusing Profile BlockwiseMatMulLayer in rank:{}, ({},{},{},{})", rank_,
                                  inter_data_type_, max_m_, k_, n_);
    SetGemmType(BlockwiseMatMulLayer::Fp8GemmType::Dynamic);
    return Status();
  }

  // Generate random input for profiling
  auto input = torch::randn({static_cast<int64_t>(max_m_), static_cast<int64_t>(k_)},
                            torch::TensorOptions().device(torch::kCUDA).dtype(torch::kBFloat16));
  Tensor output_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, {max_m_, n_});
  Tensor input_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, {max_m_, k_},
                      static_cast<void*>(input.data_ptr<torch::BFloat16>()));

  // ===== Phase 1: threshold-based search for DeepGemm / SwapAB vs Cutlass =====
  // Strategy (validated in upstream):
  //   1. Start from m=kAlignSize_, increase step size adaptively
  //   2. For each m, measure Cutlass, DeepGemm, DeepGemmSwapAB
  //   3. Early termination: stop when Cutlass beats both DeepGemm variants
  //   4. deepgemm_max_m_threshold = largest m where DeepGEMM is better
  //   5. swap_ab_max_m_thresholds: per-64-block threshold for SwapAB vs DeepGemm
  {
    auto profile_phase1_m = [&](size_t m) -> std::tuple<float, float, float> {
      input_tensor.shape = {m, k_};
      std::vector<Tensor> input_tensors = {input_tensor, weight_};
      std::vector<Tensor> output_tensors = {output_tensor};

      auto cutlass_kernel = [&]() {
        SetGemmType(Fp8GemmType::Cutlass);
        this->Forward(input_tensors, output_tensors);
      };
      auto deepgemm_kernel = [&]() {
        SetGemmType(Fp8GemmType::DeepGemm);
        this->Forward(input_tensors, output_tensors);
      };
      auto swap_ab_kernel = [&]() {
        SetGemmType(Fp8GemmType::DeepGemmSwapAB);
        this->Forward(input_tensors, output_tensors);
      };

      float ct = MeasureCudaExecutionTime(cutlass_kernel, context_->device->GetPrimaryComputeStream().Get(),
                                          kWarmupIters, kPerfIters);
      float dt = MeasureCudaExecutionTime(deepgemm_kernel, context_->device->GetPrimaryComputeStream().Get(),
                                          kWarmupIters, kPerfIters);
      float st = MeasureCudaExecutionTime(swap_ab_kernel, context_->device->GetPrimaryComputeStream().Get(),
                                          kWarmupIters, kPerfIters);
      return {ct, dt, st};
    };

    size_t m = kAlignSize_;
    while (m <= max_m_) {
      auto [cutlass_time, deepgemm_time, swap_ab_time] = profile_phase1_m(m);

      if (std::min(deepgemm_time, swap_ab_time) < cutlass_time) {
        search_cache_.deepgemm_max_m_threshold = m;
      } else {
        break;
      }
      if (m <= kPhase1StepThreshold_) {
        if ((m - kAlignSize_) / kWgmmaBlockM_ == search_cache_.swap_ab_max_m_thresholds.size()) {
          search_cache_.swap_ab_max_m_thresholds.push_back((m - kAlignSize_) / kWgmmaBlockM_ * kWgmmaBlockM_);
        }
        if (swap_ab_time < deepgemm_time) {
          search_cache_.swap_ab_max_m_thresholds.back() = m;
        }
      }

      m += m < kPhase1StepThreshold_ ? std::min(m, kPhase1Step_) : m;
    }
  }

  // ===== Phase 2: point-based search for Cutlass vs BridgeDeepGemm =====
  // From deepgemm_max_m_threshold to max_m_, profile only Cutlass vs BridgeDeepGemm.
  // Step: 512 for m < 8192, 1024 for m >= 8192.
  {
    size_t phase2_start;
    if (search_cache_.deepgemm_max_m_threshold == 0) {
      phase2_start = kAlignSize_;
    } else {
      size_t step =
          (search_cache_.deepgemm_max_m_threshold < kPhase2StepThreshold_) ? kPhase2StepSmall_ : kPhase2StepLarge_;
      phase2_start = ((search_cache_.deepgemm_max_m_threshold / step) + 1) * step;
    }

    auto profile_phase2_m = [&](size_t m) -> Fp8GemmType {
      input_tensor.shape = {m, k_};
      std::vector<Tensor> input_tensors = {input_tensor, weight_};
      std::vector<Tensor> output_tensors = {output_tensor};

      SetGemmType(Fp8GemmType::Cutlass);
      float ct = MeasureCudaExecutionTime([&] { this->Forward(input_tensors, output_tensors); },
                                          context_->device->GetPrimaryComputeStream().Get(), kWarmupIters, kPerfIters);
      SetGemmType(Fp8GemmType::BridgeDeepGemm);
      float bt = MeasureCudaExecutionTime([&] { this->Forward(input_tensors, output_tensors); },
                                          context_->device->GetPrimaryComputeStream().Get(), kWarmupIters, kPerfIters);
      return ct < bt ? Fp8GemmType::Cutlass : Fp8GemmType::BridgeDeepGemm;
    };

    for (size_t m = phase2_start; m <= max_m_;) {
      search_cache_.phase2_keys.push_back(m);
      search_cache_.phase2_types.push_back(profile_phase2_m(m));
      m += (m < kPhase2StepThreshold_) ? kPhase2StepSmall_ : kPhase2StepLarge_;
    }
  }

  // Cache the search results
  (*sel_cache)[sel_key] = search_cache_;

  std::vector<int32_t> types_int(search_cache_.phase2_types.size());
  for (size_t i = 0; i < search_cache_.phase2_types.size(); ++i)
    types_int[i] = static_cast<int32_t>(search_cache_.phase2_types[i]);
  KLLM_LOG_INFO << fmt::format(
      "Rank[{}] BlockwiseMatMulLayer search: threshold={}, swap_ab_thresholds={}, phase2_points={}, phase2_types={}, "
      "cost: {} ms",
      rank_, search_cache_.deepgemm_max_m_threshold, search_cache_.swap_ab_max_m_thresholds, search_cache_.phase2_keys,
      types_int, ProfileTimer::GetCurrentTimeInMs() - start_time);

  BuildDeepGemmKernels();
  SetGemmType(BlockwiseMatMulLayer::Fp8GemmType::Dynamic);
  return Status();
}

void BlockwiseMatMulLayer::BuildDeepGemmKernels() {
  if (!deepgemm_enabled_) {
    return;
  }

  const auto start_time = ProfileTimer::GetCurrentTimeInMs();

  // Build kernel
  static std::mutex g_mtx;
  std::lock_guard<std::mutex> guard(g_mtx);

  static_assert(kWgmmaBlockM_ % kAlignSize_ == 0);
  for (size_t begin_m = kAlignSize_; begin_m <= search_cache_.deepgemm_max_m_threshold; begin_m += kWgmmaBlockM_) {
    const size_t swap_ab_max_m_threshold =
        begin_m > search_cache_.swap_ab_max_m_thresholds.back()
            ? begin_m - kAlignSize_
            : search_cache_.swap_ab_max_m_thresholds[(begin_m - kAlignSize_) / kWgmmaBlockM_];
    // Build SwapAB kernel
    for (size_t m = begin_m; m <= swap_ab_max_m_threshold; m += kAlignSize_) {
      trtllm_deepgemm_wrapper_->BuildGemmSwapABKernel(m, n_, k_);
    }
    // Build regular kernel
    for (size_t m = swap_ab_max_m_threshold + kAlignSize_;
         m < begin_m + kWgmmaBlockM_ && m <= search_cache_.deepgemm_max_m_threshold; m += kAlignSize_) {
      trtllm_deepgemm_wrapper_->BuildGemmKernel(m, n_, k_);
    }
  }

  KLLM_LOG_INFO << fmt::format("Rank[{}] BlockwiseMatMulLayer BuildDeepGemmKernels cost time: {} ms", rank_,
                               ProfileTimer::GetCurrentTimeInMs() - start_time);
}

void BlockwiseMatMulLayer::SetGemmType(BlockwiseMatMulLayer::Fp8GemmType gemm_type) { gemm_type_ = gemm_type; }

BlockwiseMatMulLayer::Fp8GemmType BlockwiseMatMulLayer::PickGemmType(size_t m) {
  if (gemm_type_ != BlockwiseMatMulLayer::Fp8GemmType::Dynamic) {
    return gemm_type_;
  }
  const size_t aligned_m = RoundUp(m, kAlignSize_);

  // Phase 1: small m — DeepGemm variants (threshold-based from search)
  if (aligned_m <= search_cache_.deepgemm_max_m_threshold) {
    if (aligned_m > search_cache_.swap_ab_max_m_thresholds.back() ||
        aligned_m > search_cache_.swap_ab_max_m_thresholds[(aligned_m - kAlignSize_) / kWgmmaBlockM_]) {
      return Fp8GemmType::DeepGemm;
    }
    return Fp8GemmType::DeepGemmSwapAB;
  }

  // Phase 2: large m — Cutlass vs BridgeDeepGemm (point-based from search)
  auto it = std::lower_bound(search_cache_.phase2_keys.begin(), search_cache_.phase2_keys.end(), aligned_m);
  if (it != search_cache_.phase2_keys.end()) {
    return search_cache_.phase2_types[it - search_cache_.phase2_keys.begin()];
  }
  return BlockwiseMatMulLayer::Fp8GemmType::Cutlass;
}

template <typename T>
size_t BlockwiseMatMulLayer::GetWorkspaceSizeT() {
  size_t input_size = max_m_ * k_ * GetTypeSize(TYPE_FP8_E4M3);
  size_t scale_size = max_m_ * DivRoundUp(k_, block_size_) * GetTypeSize(TYPE_FP32);
  input_buffer_size_ = input_size + scale_size;
  cutlass_gemm_workspace_size_ = GetCachedCutlassBufferSize<T>();
  size_t deepgemm_workspace_size = 0;
  workspace_size_ = input_size + scale_size + std::max(cutlass_gemm_workspace_size_, deepgemm_workspace_size);
  KLLM_LOG_DEBUG << fmt::format("Rank[{}] Request {} for BlockwiseMatMulLayer", rank_, workspace_size_);
  return workspace_size_;
}

Status BlockwiseMatMulLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors,
                         const LayerExecContext& exec_ctx) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, input_tensors, output_tensors, exec_ctx);
}

template <typename T>
Status BlockwiseMatMulLayer::ForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors,
                                      const LayerExecContext& exec_ctx) {
  if (workspace_size_ > workspace_buffer_->GetTotalBytes()) {
    KLLM_THROW(fmt::format("workspace size {} > buffer size {}", workspace_size_, workspace_buffer_->GetTotalBytes()));
  }
  const size_t m = input_tensors[0].shape[0];
  const size_t k = input_tensors[0].shape[1];
  // input_tensors[0].shape[1] is k_ (normal case) or 2*k_ (need to do silu mul first)
  // input_tensors[1].shape[0] is n_
  BlockwiseMatMulLayer::Fp8GemmType gemm_type = PickGemmType(m);
  const size_t cur_m = RoundUp(m, kAlignSize_);
  T* a = static_cast<T*>(input_tensors[0].GetPtr<void>());
  void* a_q = workspace_buffer_->GetPtr<void>();
  void* a_s = a_q + GetTypeSize(TYPE_FP8_E4M3) * cur_m * k_;

  cudaStream_t compute_stream = GetComputeStream(exec_ctx);
  if (!skip_quant_) {
    InvokePerTokenGroupQuantFp8E4m3<T>(a, a_q, a_s, cur_m, k_, /*is_column_major*/ true, compute_stream, block_size_,
                                       PerTokenGroupQuantFusionParams{
                                           .fuse_silu_mul = (k == 2 * k_),
                                           .swiglu_limit = swiglu_limit_,
                                       });
  }

  void* b = input_tensors[1].GetPtr<void>();
  void* b_s = input_tensors[1].weight_scales->GetPtr<void>();

  void* out = output_tensors[0].GetPtr<void>();

  // Execute GEMM operation based on selected type
  switch (gemm_type) {
    case BlockwiseMatMulLayer::Fp8GemmType::DeepGemmSwapAB:
      trtllm_deepgemm_wrapper_->GemmSwapAB(a_q, a_s, b, b_s, out, cur_m, n_, k_, compute_stream);
      break;
    case BlockwiseMatMulLayer::Fp8GemmType::DeepGemm:
      trtllm_deepgemm_wrapper_->Gemm(a_q, a_s, b, b_s, out, cur_m, n_, k_, compute_stream);
      break;
    case BlockwiseMatMulLayer::Fp8GemmType::Cutlass: {
      void* cutlass_buffer = workspace_buffer_->GetPtr<void>() + input_buffer_size_;
      InvokeBlockGemm<T>(a_q, static_cast<float*>(a_s), b, static_cast<float*>(b_s), out, cur_m, k_, n_, compute_stream,
                         cutlass_buffer, cutlass_gemm_workspace_size_);
      break;
    }
    case BlockwiseMatMulLayer::Fp8GemmType::BridgeDeepGemm:
      bridge_deepgemm_wrapper_->Fp8BlockGemm(a_q, a_s, b, b_s, out, cur_m, n_, k_, compute_stream);
      break;
    default:
      KLLM_THROW(fmt::format("Invalid or unsupported gemm type: {}", static_cast<int>(gemm_type)));
      break;
  }

  output_tensors[0].shape = {m, n_};
  output_tensors[0].dtype = input_tensors[0].dtype;
  return Status();
}

}  // namespace ksana_llm
#endif