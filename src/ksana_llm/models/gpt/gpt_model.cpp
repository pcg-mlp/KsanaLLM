/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/gpt/gpt_model.h"

namespace ksana_llm {

template <typename T>
GPTModel<T>::GPTModel(const ModelConfig& model_config, const int rank, std::shared_ptr<Context> context,
                      std::shared_ptr<BaseWeight> base_weight)
    : CommonModel<T>(model_config, rank, context) {
  ModelRunConfig model_run_config;
  model_run_config.position_encoding = PositionEncoding::LEARNED_ABSOLUTE;
  model_run_config.qkv_add_bias = true;
  // Use the vocab size to distinguish each model
  if (model_config.vocab_size == 40478) {  // GPT-1
    model_run_config.layernorm_position = LayerNormPosition::POST_NORM;
  } else if (model_config.vocab_size == 7000) {  // Fairseq transformer
    model_run_config.layernorm_position = LayerNormPosition::POST_NORM;
    // https://datascience.stackexchange.com/questions/87906/transformer-model-why-are-word-embeddings-scaled-before-adding-positional-encod
    model_run_config.emb_scale = std::sqrt(model_config.hidden_units);
  }
  CommonModel<T>::InitRunConfig(model_run_config, base_weight);

  if (model_config_.activation_function == "gelu" || model_config_.activation_function == "gelu_new") {
    activation_layer_ = std::make_shared<ActivationLayer<ActivationType::Gelu, T>>();
  } else if (model_config_.activation_function == "relu") {
    activation_layer_ = std::make_shared<ActivationLayer<ActivationType::Relu, T>>();
  } else {
    KLLM_THROW(fmt::format("Unsupport activation function: {}", model_config_.activation_function));
  }
  activation_layer_->Init({}, context_, rank_);
}

template <typename T>
Status GPTModel<T>::LayerNormForward(const std::string& layer_name, std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                                     const std::vector<Tensor>& layernorm_input,
                                     std::vector<Tensor>& layernorm_output) {
  Tensor layernorm_weight = base_weight->GetModelWeights(layer_name);
  Tensor layernorm_bias =
      base_weight->GetModelWeights(layer_name.substr(0, layer_name.size() - strlen("weight")) + "bias");
  STATUS_CHECK_RETURN(
      layernorm_layer_->Forward({layernorm_input[0], layernorm_weight, layernorm_bias}, layernorm_output));
  return Status();
}

template <typename T>
Status GPTModel<T>::CommonAttention(const int layer_idx, std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                                    const std::vector<Tensor>& attention_input, const bool is_context_stage) {
  // Attn proj MatMul
  Tensor attn_proj_weight =
      base_weight->GetModelWeights(fmt::format("model.layers.{}.self_attn.query_key_value.weight", layer_idx));
  STATUS_CHECK_RETURN(attn_qkv_proj_layer_->Forward({attention_input[0], attn_proj_weight}, hidden_buffer_1_));
  Tensor attn_proj_bias =
      base_weight->GetModelWeights(fmt::format("model.layers.{}.self_attn.query_key_value.bias", layer_idx));
  STATUS_CHECK_RETURN(add_layer_->Forward({hidden_buffer_1_[0], attn_proj_bias}, hidden_buffer_1_));
  std::swap(hidden_buffer_1_, hidden_buffer_0_);

  // MMHA Flash/Paged Attention
  if (layer_idx == 0) {
    // only need sync in the first layer
    StreamWaitEvent(context_->GetComputeStreams()[rank_], model_input_->kvcache_offset_event);
    StreamWaitEvent(context_->GetComputeStreams()[rank_], model_input_->rotary_embedding_event);
  }

  if (is_context_stage) {
    CommonModel<T>::FlashAttentionForward(layer_idx);
  } else {
    CommonModel<T>::PagedAttentionForward(layer_idx);
  }

  // Attn o_proj MatMul
  Tensor attn_o_proj_weight =
      base_weight->GetModelWeights(fmt::format("model.layers.{}.self_attn.o_proj.weight", layer_idx));
  if (model_communicator_) {
    // Put output to `reduce_buffer_` to ensure that the input for custom reduce sum is always in `reduce_buffer_`.
    STATUS_CHECK_RETURN(attn_o_proj_layer_->Forward({hidden_buffer_0_[0], attn_o_proj_weight}, reduce_buffer_));
  } else {
    STATUS_CHECK_RETURN(attn_o_proj_layer_->Forward({hidden_buffer_0_[0], attn_o_proj_weight}, hidden_buffer_1_));
    std::swap(hidden_buffer_1_, hidden_buffer_0_);
  }
  // Only Add o_proj bias on rank 0 to avoid duplication.
  if (rank_ == 0) {
    Tensor attn_o_proj_bias =
        base_weight->GetModelWeights(fmt::format("model.layers.{}.self_attn.o_proj.bias", layer_idx));
    STATUS_CHECK_RETURN(add_layer_->Forward({hidden_buffer_0_[0], attn_o_proj_bias}, hidden_buffer_0_));
  }

  // NOTE(karlluo): multiple event in nccl will cause preformance regression
  // nccl multiple event just enable when context.IsRunContextDecodeAndDecodeSerially() == false
  if (!context_->IsRunContextDecodeAndDecodeSerially()) {
    EventRecord(model_output_->compute_ready_event, context_->GetComputeStreams()[rank_]);
    StreamWaitEvent(context_->GetNCCLStreams()[rank_], model_output_->compute_ready_event);
  }

  // Attn AllReduceSum
  if (model_communicator_) {
    model_communicator_->ReduceSum(reduce_buffer_, hidden_buffer_0_, is_context_stage, true);
  }
  return Status();
}

template <typename T>
Status GPTModel<T>::CommonMlp(const int layer_idx, std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                              const std::vector<Tensor>& mlp_input, const bool is_context_stage) {
  // Mlp gate_proj MatMul
  Tensor gate_proj_weight =
      base_weight->GetModelWeights(fmt::format("model.layers.{}.mlp.gate_proj.weight", layer_idx));
  STATUS_CHECK_RETURN(mlp_gate_proj_layer_->Forward({mlp_input[0], gate_proj_weight}, hidden_buffer_1_));
  Tensor gate_proj_bias = base_weight->GetModelWeights(fmt::format("model.layers.{}.mlp.gate_proj.bias", layer_idx));
  STATUS_CHECK_RETURN(add_layer_->Forward({hidden_buffer_1_[0], gate_proj_bias}, hidden_buffer_1_));
  std::swap(hidden_buffer_1_, hidden_buffer_0_);

  // Activation is an in-place operation, just put the output in `hidden_buffer_0_`, the same as the input.
  STATUS_CHECK_RETURN(activation_layer_->Forward({hidden_buffer_0_[0]}, hidden_buffer_0_));

  // Mlp down_proj MatMul
  Tensor down_proj_weight =
      base_weight->GetModelWeights(fmt::format("model.layers.{}.mlp.down_proj.weight", layer_idx));
  if (model_communicator_) {
    // Put output to `reduce_buffer_` to ensure that the input for custom reduce sum is always in `reduce_buffer_`.
    STATUS_CHECK_RETURN(mlp_down_proj_layer_->Forward({hidden_buffer_0_[0], down_proj_weight}, reduce_buffer_));
  } else {
    STATUS_CHECK_RETURN(mlp_down_proj_layer_->Forward({hidden_buffer_0_[0], down_proj_weight}, hidden_buffer_1_));
    std::swap(hidden_buffer_1_, hidden_buffer_0_);
  }
  // Only add down_proj bias for rank 0 to avoid duplication.
  if (rank_ == 0) {
    Tensor down_proj_bias = base_weight->GetModelWeights(fmt::format("model.layers.{}.mlp.down_proj.bias", layer_idx));
    STATUS_CHECK_RETURN(add_layer_->Forward({hidden_buffer_0_[0], down_proj_bias}, hidden_buffer_0_));
  }

  // NOTE(karlluo): multiple event in nccl will cause preformance regression
  // nccl multiple event just enable when context.IsRunContextDecodeAndDecodeSerially() == false
  if (!context_->IsRunContextDecodeAndDecodeSerially()) {
    EventRecord(model_output_->compute_ready_event, context_->GetComputeStreams()[rank_]);
    StreamWaitEvent(context_->GetNCCLStreams()[rank_], model_output_->compute_ready_event);
  }

  // Mlp AllReduceSum
  if (model_communicator_) {
    model_communicator_->ReduceSum(reduce_buffer_, hidden_buffer_0_, is_context_stage, true);
  }
  return Status();
}

template <typename T>
Status GPTModel<T>::EmbedTokensUseGpu(Tensor& embedding_weight) {
  STATUS_CHECK_RETURN(emb_lookup_layer_->Forward(
      {model_input_->input_ids, model_input_->input_offset_uint64_tensor, model_input_->input_prefix_uint64_tensor,
       embedding_weight, model_input_->rotary_embedding_pos},
      residual_buffer_));

  // NOTE(karlluo): multiple event in nccl will cause preformance regression
  // nccl multiple event just enable when context.IsRunContextDecodeAndDecodeSerially() == false
  if (!context_->IsRunContextDecodeAndDecodeSerially()) {
    EventRecord(model_output_->compute_ready_event, context_->GetComputeStreams()[rank_]);
    StreamWaitEvent(context_->GetNCCLStreams()[rank_], model_output_->compute_ready_event);
  }

  if (model_communicator_) {
    model_communicator_->AllGather({residual_buffer_[0], hidden_buffer_1_[0]}, residual_buffer_);
  }
  return Status();
}

template class GPTModel<float>;
template class GPTModel<float16>;
#ifdef ENABLE_BFLOAT16
template class GPTModel<bfloat16>;
#endif

}  // namespace ksana_llm
