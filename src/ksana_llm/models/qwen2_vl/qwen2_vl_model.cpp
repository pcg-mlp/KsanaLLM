/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/qwen2_vl/qwen2_vl_model.h"

#include "torch/csrc/autograd/python_variable.h"

namespace ksana_llm {

template <typename T>
Qwen2VLModel<T>::Qwen2VLModel(const ModelConfig& model_config, const int rank, std::shared_ptr<Context> context,
                              std::shared_ptr<BaseWeight> base_weight)
    : CommonModel<T>(model_config, rank, context) {
  ModelRunConfig model_run_config;
  model_run_config.position_encoding = PositionEncoding::ROPE;
  model_run_config.qkv_add_bias = true;
  CommonModel<T>::InitRunConfig(model_run_config, base_weight);
}

template <typename T>
Status Qwen2VLModel<T>::FlashAttentionForward(const int layer_idx) {
  bool reuse_prefix_caching = prefix_caching_enabled_;

#ifndef ENABLE_FLASH_ATTN_WITH_CACHE
  if (reuse_prefix_caching) {
    CommonModel<T>::AddAttentionPrefixCache();
  }
#endif

#ifdef ENABLE_CUDA
  STATUS_CHECK_RETURN(flash_attention_layers_[layer_idx]->Forward(
      {hidden_buffer_0_[0], model_input_->input_offset_uint64_tensor, model_input_->kv_list,
       model_input_->input_prefix_uint64_tensor, model_input_->kv_cache_offset_tensor,
       model_input_->mrotary_embedding_pos, model_input_->rotary_embedding_mask,
       model_input_->flexible_rotary_embedding_pos, model_input_->flexible_rotary_embedding_mask,
       model_input_->dst_flexible_kv_cache_tensor, model_input_->src_flexible_kv_cache_tensor,
       model_input_->dst_flexible_token_idx_tensor, model_input_->src_flexible_token_idx_tensor,
       model_input_->flexible_offset_uint64_tensor, forward_shape_, flag_tensor_
#  ifdef ENABLE_FLASH_ATTN_WITH_CACHE
       ,
       model_input_->layer_kv_cache_ptr_tensor, model_input_->multi_token_request_block_table,
       model_input_->input_without_prefix_uint64_tensor
#  endif
      },
      hidden_buffer_1_));
#elif defined(ENABLE_ACL)
  // inference on NPU with ATB
  KLLM_THROW("Qwen2_vl not supported on NPU");
#endif
  std::swap(hidden_buffer_1_, hidden_buffer_0_);

#ifndef ENABLE_FLASH_ATTN_WITH_CACHE
  if (reuse_prefix_caching) {
    CommonModel<T>::RemoveAttentionPrefixCache();
  }
#endif

  return Status();
}

template <typename T>
Status Qwen2VLModel<T>::Forward(std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                                std::vector<ForwardRequest>& forward_reqs, bool epilogue) {
  STATUS_CHECK_RETURN(PrepareMRopePos(forward_reqs));
  return CommonModel<T>::Forward(base_weight, forward_reqs, epilogue);
}

/**
 * The MRope position information (position and offset) of qwen2_vl is computed by the `_get_input_positions` function
 * in the Python plugin and is passed as additional tensors.
 * Before model inference, copy the position tensor (`additional_tensors[0]`) to the corresponding GPU tensor
 * (`mrotary_embedding_pos`), and record the offset value (`additioanl_tensors[1]`).
 */
template <typename T>
Status Qwen2VLModel<T>::PrepareMRopePos(std::vector<ForwardRequest>& forward_reqs) {
  const size_t batch_size = forward_reqs.size();
  int64_t mrotary_embedding_pos_size = 0;

  for (size_t idx = 0; idx < batch_size && forward_reqs[idx].infer_stage == STAGE_CONTEXT; idx++) {
    auto& tensors = (*forward_reqs[idx].input_refit_embedding).additional_tensors;
    // This is a plain text input.
    if (tensors.empty()) {
      int64_t list_size = forward_reqs[idx].output_tokens->size() * 3;
      std::vector<int64_t> mrotary_embedding_pos_list(list_size);
      for (int64_t i = 0; i < list_size; i += 3) {
        mrotary_embedding_pos_list[i] = mrotary_embedding_pos_list[i + 1] = mrotary_embedding_pos_list[i + 2] = i;
      }
      MemcpyAsync(
          model_input_->mrotary_embedding_pos.template GetPtr<void>() + sizeof(int64_t) * mrotary_embedding_pos_size,
          mrotary_embedding_pos_list.data(), sizeof(int64_t) * list_size, MEMCPY_HOST_TO_DEVICE,
          context_->GetD2HStreams()[rank_]);
      mrotary_embedding_pos_size += list_size;

      *forward_reqs[idx].mrotary_embedding_pos_offset = 0;
      continue;
    }
    // This is a input with visual information.
    torch::Tensor mrotary_embedding_pos_tensor;
    {
      py::gil_scoped_acquire acquire;
      mrotary_embedding_pos_tensor = THPVariable_Unpack(tensors[0].ptr());
    }
    int64_t tensor_size = mrotary_embedding_pos_tensor.numel();
    MemcpyAsync(
        model_input_->mrotary_embedding_pos.template GetPtr<void>() + sizeof(int64_t) * mrotary_embedding_pos_size,
        mrotary_embedding_pos_tensor.data_ptr(), sizeof(int64_t) * tensor_size, MEMCPY_HOST_TO_DEVICE,
        context_->GetD2HStreams()[rank_]);
    mrotary_embedding_pos_size += tensor_size;

    torch::Tensor mrotary_embedding_pos_offset_tensor = THPVariable_Unpack(tensors[1].ptr());
    *forward_reqs[idx].mrotary_embedding_pos_offset = mrotary_embedding_pos_offset_tensor.item().toLong();
  }
  return Status();
}

template class Qwen2VLModel<float>;
template class Qwen2VLModel<float16>;
#ifdef ENABLE_BFLOAT16
template class Qwen2VLModel<bfloat16>;
#endif

}  // namespace ksana_llm
