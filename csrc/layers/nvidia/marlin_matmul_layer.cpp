/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "layers/marlin_matmul_layer.h"
#include "csrc/kernels/nvidia/gptq_marlin/marlin_wrapper.h"
#include "kernels/nvidia/kernel_wrapper.h"
#include "utils/utils.h"

namespace ksana_llm {

Status MarlinMatMulLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                               std::shared_ptr<Context> context, int rank) {
  context_ = context;
  rank_ = rank;
  inter_data_type_ = runtime_config.inter_data_type;

  int parameter_index = 0;
  max_m_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  max_n_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  max_k_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  groupsize_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  is_awq_ = std::any_cast<const bool>(parameters[parameter_index++]);
  is_gptq_desc_ = std::any_cast<const bool>(parameters[parameter_index++]);
  is_k_full_ = std::any_cast<const bool>(parameters[parameter_index++]);
  bool cutlass_use_gemv_cuda_core_ = std::any_cast<const bool>(parameters[parameter_index++]);  // unused
  weight_data_type_ = std::any_cast<const DataType>(parameters[parameter_index++]);

  // double check some parameter
  is_k_full_ = is_gptq_desc_ ? is_k_full_ : true;

  return Status();
}

size_t MarlinMatMulLayer::GetWorkspaceSize() { DISPATCH_BY_3_DTYPE(inter_data_type_, GetWorkspaceSizeT); }

inline size_t AlignAddress(size_t size) { return (size + 255) & (~255); }

template <typename T>
size_t MarlinMatMulLayer::GetWorkspaceSizeT() {
  if (weight_data_type_ == TYPE_I4_GROUP) {
    llm_kernels::nvidia::marlin::WorkspaceInfo info = GetMarlinWorkspace<T>(true, is_gptq_desc_, rank_, max_m_, max_k_);

    marlin_workspace_size_ = AlignAddress(info.workspace_size);
    marlin_input_tmp_size_ = AlignAddress(info.a_tmp_size);
    marlin_output_tmp_size_ = AlignAddress(info.c_tmp_size);

    marlin_workspace_offset_ = 0;
    marlin_input_tmp_offset_ = marlin_workspace_offset_ + marlin_workspace_size_;
    marlin_output_tmp_offset_ = marlin_input_tmp_offset_ + marlin_input_tmp_size_;

    size_t max_ws_bytes = marlin_workspace_size_ + marlin_input_tmp_size_ + marlin_output_tmp_size_;

    KLLM_LOG_DEBUG << fmt::format("Rank[{}] Request {} for MarlinMatMulLayer", rank_, max_ws_bytes);
    return max_ws_bytes;
  } else {
    KLLM_THROW(fmt::format("Not supported weight data type: {}. MarlinMatMul only supports TYPE_I4_GROUP.",
                           weight_data_type_));
  }
}

template <typename T>
Status MarlinMatMulLayer::ForwardT(const Tensor& input, const Tensor& weight, Tensor& output,
                                   const LayerExecContext& exec_ctx) {
  if (weight_data_type_ == TYPE_I4_GROUP) {
    void* p_qweight_tensor = weight.GetPtr<void>();
    void* p_scales_tensor = weight.scales->GetPtr<void>();
    void* p_zeros_tensor = is_awq_ ? weight.zeros->GetPtr<void>() : nullptr;
    void* p_g_idx_tensor = is_gptq_desc_ ? weight.g_idx->GetPtr<void>() : nullptr;
    void* p_perm_tensor = is_gptq_desc_ ? weight.perm->GetPtr<void>() : nullptr;

    const size_t m = input.shape[0];
    const size_t n = max_n_;
    const size_t k = max_k_;

    const size_t num_group = weight.scales->shape[0];
    uint8_t* ws_ptr = reinterpret_cast<uint8_t*>(workspace_buffer_->GetPtr<void>());
    MemsetAsync(ws_ptr + marlin_workspace_offset_, 0, marlin_workspace_size_,
                exec_ctx.HasStreamOverride() ? *exec_ctx.stream : context_->device->GetPrimaryComputeStream());
    InvokeMarlinGemm<T>(input.GetPtr<void>(), ws_ptr + marlin_input_tmp_offset_, p_qweight_tensor, p_scales_tensor,
                        p_zeros_tensor, p_g_idx_tensor, p_perm_tensor, ws_ptr + marlin_workspace_offset_,
                        output.GetPtr<void>(), ws_ptr + marlin_output_tmp_offset_, m, n, k, num_group, is_k_full_, true,
                        true, false, is_awq_, is_gptq_desc_, is_awq_, rank_, GetComputeStream(exec_ctx));

    output.shape = {m, n};
    output.dtype = input.dtype;
    return Status();
  } else {
    KLLM_THROW(fmt::format("Not supported weight data type: {}. MarlinMatMul only supports TYPE_I4_GROUP.",
                           weight_data_type_));
  }
}

Status MarlinMatMulLayer::Forward(const Tensor* input, const Tensor* weight, Tensor* output,
                                  const LayerExecContext& exec_ctx) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, *input, *weight, *output, exec_ctx);
}

}  // namespace ksana_llm
