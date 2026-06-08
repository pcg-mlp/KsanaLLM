/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/mul_layer.h"
#include "kernels/nvidia/kernel_wrapper.h"

namespace ksana_llm {

template <typename T>
Status MulLayer::ForwardT(const Tensor& a, const Tensor& b, Tensor& output) {
  int input_0_m = a.shape[0];
  int input_0_n = a.shape[1];
  int input_1_m = b.shape[0];
  int input_1_n = b.shape[1];
  if (input_0_m != input_1_m && input_0_m != 1 && input_1_m != 1) {
    KLLM_THROW(
        fmt::format("The size of first tensor {} must match the size of second tensor {} at non-singleton dimension 0",
                    input_0_m, input_1_m));
  }
  if (input_0_n != input_1_n && input_0_n != 1 && input_1_n != 1) {
    KLLM_THROW(
        fmt::format("The size of first tensor {} must match the size of second tensor {} at non-singleton dimension 1",
                    input_0_n, input_1_n));
  }
  InvokeMul<T>(a.GetPtr<void>(), b.GetPtr<void>(), output.GetPtr<void>(), input_0_m, input_0_n, input_1_m, input_1_n,
               rank_);
  size_t output_m = static_cast<size_t>((input_0_m >= input_1_m) ? input_0_m : input_1_m);
  size_t output_n = static_cast<size_t>((input_0_n >= input_1_n) ? input_0_n : input_1_n);
  output.shape = {output_m, output_n};
  output.dtype = b.dtype;
  return Status();
}

Status MulLayer::Forward(const Tensor* a, const Tensor* b, Tensor* output, const LayerExecContext& exec_ctx) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, *a, *b, *output);
}

}  // namespace ksana_llm
