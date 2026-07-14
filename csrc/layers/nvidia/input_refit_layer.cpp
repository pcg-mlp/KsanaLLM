/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/input_refit_layer.h"

namespace ksana_llm {

Status InputRefitLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, input_tensors, output_tensors);
}

template <typename T>
Status InputRefitLayer::ForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  // input_tensors: emb0, emb1, ..., embN, pos
  if (input_tensors.size() <= 1) {
    return Status();
  }
  cast_cache_.clear();

  const Tensor& pos_tensor = input_tensors.back();
  const size_t pos_num = pos_tensor.shape[0];
  const int64_t* pos_ptr = pos_tensor.GetPtr<int64_t>();
  KLLM_CHECK(input_tensors.size() == pos_num + 1);

  const size_t hidden_units = output_tensors[0].shape[1];
  void* output_ptr = output_tensors[0].GetPtr<void>();

  for (size_t i = 0; i < pos_num; i++) {
    const int64_t token_pos = pos_ptr[i];
    const Tensor& embedding = input_tensors[i];
    const int64_t len = embedding.GetElementNumber();
    void* embedding_ptr = embedding.GetPtr<void>();
    // 智能cast
    if (embedding.dtype != output_tensors[0].dtype) {
      KLLM_LOG_DEBUG << "Casting input refit embedding data type";
      auto input_option = torch::TensorOptions().dtype(GetTorchTypeFromDataType(embedding.dtype));
      torch::Tensor src_tensor = torch::from_blob(embedding_ptr, {len}, input_option);
      torch::Tensor dst_tensor = src_tensor.to(GetTorchTypeFromDataType(output_tensors[0].dtype));
      cast_cache_.push_back(dst_tensor);  // 需要cache住避免提前释放
      embedding_ptr = dst_tensor.data_ptr();
    }
    // 拷贝
    const size_t output_offset = token_pos * hidden_units;
    if (output_offset + len <= output_tensors[0].GetElementNumber()) {
      MemcpyAsync(output_ptr + output_offset * sizeof(T), embedding_ptr, len * sizeof(T), MEMCPY_HOST_TO_DEVICE,
                  context_->device->GetPrimaryComputeStream());
    }
  }
  return Status();
}

}  // namespace ksana_llm
