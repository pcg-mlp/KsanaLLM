/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/layernorm_layer.h"

#include "3rdparty/LLM_kernels/csrc/kernels/nvidia/layernorm/rmsnorm_fused_parallel.h"
#include "3rdparty/LLM_kernels/csrc/kernels/nvidia/others/sglang/main/deepseek_v4/rmsnorm.h"
#include "kernels/nvidia/kernel_wrapper.h"

namespace ksana_llm {

Status LayernormLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                            std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);
  int parameter_index = 0;
  rms_norm_eps_ = std::any_cast<const float>(parameters[parameter_index++]);
  KLLM_LOG_DEBUG << fmt::format("rms_norm_eps {}", rms_norm_eps_);

  // Optional: num_heads and head_dim for per-head no-weight RMSNorm
  if (parameter_index < static_cast<int>(parameters.size())) {
    num_heads_ = std::any_cast<const uint32_t>(parameters[parameter_index++]);
    head_dim_ = std::any_cast<const uint32_t>(parameters[parameter_index++]);
  }

  return Status();
}

Status LayernormLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, input_tensors, output_tensors);
}

template <typename T>
Status LayernormLayer::ForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  // input_tensors:
  //   0: input [token_num, hidden_size]
  //   1: weight [hidden_size] (optional)
  //   2: bias [hidden_size] (optional)
  // output_tensors:
  //   0: output [token_num, hidden_size]
  // Note: when bias is provided, compute layernorm, otherwise compute rmsnorm.
  //       when no weight is provided (size==1), compute per-head no-weight rmsnorm via sglang kernel.
  //
  // Parallel path (size==4):
  //   input_tensors[0]: input1 [token_num, hidden_size1]
  //   input_tensors[1]: weight1 [hidden_size1]
  //   input_tensors[2]: input2 [token_num, hidden_size2]
  //   input_tensors[3]: weight2 [hidden_size2]
  //   output_tensors[0]: output1 [token_num, hidden_size1]
  //   output_tensors[1]: output2 [token_num, hidden_size2]

  if (input_tensors.size() == 4) {
    // Parallel RmsNorm for two inputs
    const int token_num = input_tensors[0].shape[0];
    const int hidden_size1 = input_tensors[0].shape[1];
    const int hidden_size2 = input_tensors[2].shape[1];
    // When shape has 3 dims, the 3rd dim encodes the row stride (in elements);
    // otherwise stride == hidden_size (contiguous).
    const int stride1 = input_tensors[0].shape.size() > 2 ? static_cast<int>(input_tensors[0].shape[2]) : hidden_size1;
    const int stride2 = input_tensors[2].shape.size() > 2 ? static_cast<int>(input_tensors[2].shape[2]) : hidden_size2;
    llm_kernels::nvidia::InvokeParallelRmsNorm<T>(
        input_tensors[0].GetPtr<const T>(), input_tensors[1].GetPtr<const T>(), output_tensors[0].GetPtr<T>(),
        hidden_size1,
        /*stride_input1*/ stride1, input_tensors[2].GetPtr<const T>(), input_tensors[3].GetPtr<const T>(),
        output_tensors[1].GetPtr<T>(), hidden_size2,
        /*stride_input2*/ stride2, token_num, rms_norm_eps_, context_->device->GetPrimaryComputeStream().Get());
    output_tensors[0].shape = {static_cast<size_t>(token_num), static_cast<size_t>(hidden_size1)};
    output_tensors[0].dtype = input_tensors[0].dtype;
    output_tensors[1].shape = {static_cast<size_t>(token_num), static_cast<size_t>(hidden_size2)};
    output_tensors[1].dtype = input_tensors[2].dtype;
    return Status();
  } else if (input_tensors.size() > 1) {
    const void* bias = input_tensors.size() > 2 ? input_tensors[2].GetPtr<void>() : nullptr;
    if (bias) {
      InvokeLayerNorm<T>(input_tensors[0].GetPtr<void>(), input_tensors[1].GetPtr<void>(), bias, rms_norm_eps_,
                         input_tensors[0].shape[0], input_tensors[0].shape[1], output_tensors[0].GetPtr<void>(),
                         context_->device->GetPrimaryComputeStream().Get());
    } else {
      InvokeRMSNorm<T>(input_tensors[0].GetPtr<void>(), input_tensors[1].GetPtr<void>(), rms_norm_eps_,
                       input_tensors[0].shape[0], input_tensors[0].shape[1], output_tensors[0].GetPtr<void>(),
                       context_->device->GetPrimaryComputeStream().Get());
    }
  } else {
    KLLM_CHECK_WITH_INFO(num_heads_ > 0 && head_dim_ > 0,
                         "num_heads and head_dim must be set for no-weight per-head RMSNorm");

    const uint32_t batch_size = static_cast<uint32_t>(input_tensors[0].shape[0]);
    const int64_t stride_batch_bytes = static_cast<int64_t>(num_heads_ * head_dim_ * sizeof(T));
    const int64_t stride_head_bytes = static_cast<int64_t>(head_dim_ * sizeof(T));

    cudaStream_t stream = context_->device->GetPrimaryComputeStream().Get();

    // Dispatch by head_dim
    if (head_dim_ == 512) {
      llm_kernels::nvidia::rmsnorm_self<T, 512>(input_tensors[0].GetPtr<void>(), output_tensors[0].GetPtr<void>(),
                                                stride_batch_bytes, stride_head_bytes, batch_size, num_heads_,
                                                rms_norm_eps_, stream);
    } else {
      KLLM_THROW(fmt::format("Unsupported head_dim {} for no-weight per-head RMSNorm", head_dim_));
    }
  }

  output_tensors[0].shape = input_tensors[0].shape;
  output_tensors[0].dtype = input_tensors[0].dtype;
  return Status();
}

}  // namespace ksana_llm
