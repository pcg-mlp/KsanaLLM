/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/layers/group_matmul_layer.h"
#include "csrc/kernels/nvidia/asymmetric_gemm/cutlass_preprocessors.h"
#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"
#include "ksana_llm/utils/utils.h"

namespace ksana_llm {

template <typename T, DataType WT>
Status GroupMatMulLayer<T, WT>::Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context,
                                     int rank) {
  context_ = context;
  rank_ = rank;

  int parameter_index = 0;
  max_m_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  max_n_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  max_k_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  groupsize_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  use_gemv_cuda_core_ = std::any_cast<const bool>(parameters[parameter_index++]);
  // double check
  if (use_gemv_cuda_core_) {
    use_gemv_cuda_core_ = GetFpAIntBGroupCudaGemmSupported<T, llm_kernels::nvidia::WeightType::INT4>();
  }
  return Status();
}

template <typename T, DataType WT>
size_t GroupMatMulLayer<T, WT>::GetWorkSpaceSize() {
  if constexpr (WT == TYPE_I4_GROUP) {
    size_t max_ws_bytes;
    GetFpAIntBGroupCutlassGemmWorkspaceSize<T, llm_kernels::nvidia::WeightType::INT4>(max_m_, max_n_, max_k_,
                                                                                      max_ws_bytes);
    return max_ws_bytes;
  } else {
    KLLM_THROW(fmt::format("Not supported weight data type: {}. GroupMatMul only supports TYPE_I4_GROUP.", WT));
  }
}

template <typename T, DataType WT>
Status GroupMatMulLayer<T, WT>::Preprocess(const ModelConfig& model_config_) {
  if constexpr (WT == TYPE_I4_GROUP) {
    config_map_.resize(model_config_.max_batch_size + 1);  // start from 1 but not 0

    const size_t n = max_n_;
    const size_t k = max_k_;

    auto start_time = std::chrono::high_resolution_clock::now();

    Tensor buffer_input_activation;
    Tensor buffer_input_weight;
    Tensor buffer_input_scales;
    Tensor buffer_input_zeros;
    Tensor buffer_output;
    CreateTensor(buffer_input_activation, {max_m_, k}, DataType::TYPE_FP16, rank_, MemoryDevice::MEMORY_DEVICE);
    CreateTensor(buffer_input_weight, {k, n / 2}, DataType::TYPE_UINT8, rank_, MemoryDevice::MEMORY_DEVICE);
    CreateTensor(buffer_input_scales, {k / groupsize_, n}, DataType::TYPE_FP16, rank_, MemoryDevice::MEMORY_DEVICE);
    CreateTensor(buffer_input_zeros, {k / groupsize_, n}, DataType::TYPE_FP16, rank_, MemoryDevice::MEMORY_DEVICE);
    CreateTensor(buffer_output, {max_m_, n}, DataType::TYPE_FP16, rank_, MemoryDevice::MEMORY_DEVICE);

    void* zeros_ptr = buffer_input_zeros.GetPtr<void>();
    if (model_config_.quant_config.method == QUANT_GPTQ) {
      zeros_ptr = nullptr;
    }

    const size_t warmup_iters = GetEnvAsPositiveInt("QUANT_WARMUP", 2);
    const size_t record_iters = GetEnvAsPositiveInt("QUANT_PROFILE", 5);
    for (size_t m = 1; m <= static_cast<size_t>(model_config_.max_batch_size); m++) {
      size_t best_config_index = InvokeFpAIntBGroupCutlassGemmConfigProfile<T, llm_kernels::nvidia::WeightType::INT4>(
          warmup_iters, record_iters, buffer_output.GetPtr<void>(), buffer_input_activation.GetPtr<void>(),
          buffer_input_weight.GetPtr<void>(), buffer_input_scales.GetPtr<void>(), zeros_ptr,
          workspace_buffer_->GetPtr<void>(), m, n, k, groupsize_, context_->GetComputeStreams()[rank_].Get());
      config_map_[m] = best_config_index;
      KLLM_LOG_DEBUG << fmt::format("The best config index for mnk=({},{},{}) is {}", m, n, k, best_config_index);
    }

    DestroyTensor(buffer_input_activation, rank_);
    DestroyTensor(buffer_input_weight, rank_);
    DestroyTensor(buffer_input_scales, rank_);
    DestroyTensor(buffer_input_zeros, rank_);
    DestroyTensor(buffer_output, rank_);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    KLLM_LOG_INFO << fmt::format(
        "Profile Group MatMul Layer in rank:{}, mnk=({}~{},{},{}), warmup:{}, record:{}, cost:{}ms", rank_, 1,
        model_config_.max_batch_size, n, k, warmup_iters, record_iters, duration_ms.count());

    return Status();
  } else {
    KLLM_THROW(fmt::format("Not supported weight data type: {}. GroupMatMul only supports TYPE_I4_GROUP.", WT));
  }
}

template <typename T, DataType WT>
Status GroupMatMulLayer<T, WT>::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  if constexpr (WT == TYPE_I4_GROUP) {
    const Tensor& weight_tensor = input_tensors[1];
    const void* p_qweight_tensor = weight_tensor.GetPtr<void>();
    const void* p_scales_tensor = weight_tensor.scales->GetPtr<void>();

    const size_t m = input_tensors[0].shape[0];
    const size_t n = weight_tensor.scales->shape[1];
    const size_t k = input_tensors[0].shape[1];

    void* p_zeros_tensor = nullptr;
    if (weight_tensor.zeros != nullptr) {
      p_zeros_tensor = weight_tensor.zeros->GetPtr<void>();
    }

    if (use_gemv_cuda_core_ && m < 5) {
      InvokeFpAIntBGroupCudaGemm<T, llm_kernels::nvidia::WeightType::INT4>(
          output_tensors[0].GetPtr<void>(), input_tensors[0].GetPtr<void>(), p_qweight_tensor, p_scales_tensor,
          p_zeros_tensor, m, n, k, groupsize_, context_->GetComputeStreams()[rank_].Get());
      output_tensors[0].shape = {m, n};
      output_tensors[0].dtype = input_tensors[0].dtype;
      return Status();
    }

    size_t best_config_index = 0;
    if (m < config_map_.size()) {
      best_config_index = config_map_[m];
    }
    InvokeFpAIntBGroupCutlassGemm<T, llm_kernels::nvidia::WeightType::INT4>(
        output_tensors[0].GetPtr<void>(), input_tensors[0].GetPtr<void>(), p_qweight_tensor, p_scales_tensor,
        p_zeros_tensor, workspace_buffer_->GetPtr<void>(), m, n, k, groupsize_, best_config_index,
        context_->GetComputeStreams()[rank_].Get());

    output_tensors[0].shape = {m, n};
    output_tensors[0].dtype = input_tensors[0].dtype;
    return Status();
  } else {
    KLLM_THROW(fmt::format("Not supported weight data type: {}. GroupMatMul only supports TYPE_I4_GROUP.", WT));
  }
}

template class GroupMatMulLayer<float, TYPE_I4_GROUP>;
template class GroupMatMulLayer<half, TYPE_I4_GROUP>;
#ifdef ENABLE_BFLOAT16
template class GroupMatMulLayer<__nv_bfloat16, TYPE_I4_GROUP>;
#endif

}  // namespace ksana_llm
