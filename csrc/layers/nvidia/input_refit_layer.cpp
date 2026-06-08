/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/input_refit_layer.h"

namespace ksana_llm {

template <typename T>
Status InputRefitLayer::ForwardT(const std::vector<const Tensor*>& emb_tensors, const Tensor* pos_tensor,
                                 Tensor* output) {
  if (emb_tensors.empty()) {
    return Status();
  }
  cast_cache_.clear();

  const size_t pos_num = pos_tensor->shape[0];
  const int64_t* pos_ptr = pos_tensor->GetPtr<int64_t>();
  KLLM_CHECK(emb_tensors.size() == pos_num);

  const size_t hidden_units = output->shape[1];
  void* output_ptr = output->GetPtr<void>();

  for (size_t i = 0; i < pos_num; i++) {
    const int64_t token_pos = pos_ptr[i];
    const Tensor* embedding = emb_tensors[i];
    const int64_t len = embedding->GetElementNumber();
    void* embedding_ptr = embedding->GetPtr<void>();
    if (embedding->dtype != output->dtype) {
      KLLM_LOG_DEBUG << "Casting input refit embedding data type";
      auto input_option = torch::TensorOptions().dtype(GetTorchTypeFromDataType(embedding->dtype));
      torch::Tensor src_tensor = torch::from_blob(embedding_ptr, {len}, input_option);
      torch::Tensor dst_tensor = src_tensor.to(GetTorchTypeFromDataType(output->dtype));
      cast_cache_.push_back(dst_tensor);
      embedding_ptr = dst_tensor.data_ptr();
    }
    const size_t output_offset = token_pos * hidden_units;
    if (output_offset + len <= output->GetElementNumber()) {
      MemcpyAsync(reinterpret_cast<uint8_t*>(output_ptr) + output_offset * sizeof(T), embedding_ptr, len * sizeof(T),
                  MEMCPY_HOST_TO_DEVICE, context_->device->GetPrimaryComputeStream());
    }
  }
  return Status();
}

Status InputRefitLayer::Forward(const std::vector<const Tensor*>& emb_tensors, const Tensor* pos_tensor, Tensor* output,
                                const LayerExecContext& exec_ctx) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, emb_tensors, pos_tensor, output);
}

}  // namespace ksana_llm
