/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "modules/basic/mhc.h"

#include "layers/mhc_layer.h"

namespace ksana_llm {

Mhc::Mhc(const std::string& weight_name, const LayerCreationContext& creation_context, MhcBuffers& mhc_buffers,
         const std::string& norm_weight_name)
    : mhc_config_(creation_context.model_config.mhc_config), mhc_buffers_(mhc_buffers) {
  mhc_layer_ = std::make_unique<MhcLayer>();
  mhc_layer_->Init({creation_context.model_config.layernorm_eps, mhc_config_.hc_mult, mhc_config_.hc_sinkhorn_iters,
                    mhc_config_.hc_eps, creation_context.model_config.hidden_units},
                   creation_context.runtime_config, creation_context.context, creation_context.rank);
  if (!weight_name.empty()) {
    fn_weight_ = creation_context.base_weight->GetModelWeights(weight_name + "_fn");
    scale_weight_ = creation_context.base_weight->GetModelWeights(weight_name + "_scale");
    base_weight_ = creation_context.base_weight->GetModelWeights(weight_name + "_base");
  }
  if (!norm_weight_name.empty()) {
    norm_weight_ = creation_context.base_weight->GetModelWeights(norm_weight_name);
  }
}

Status Mhc::CreateBuffers(BufferManager* buffer_mgr, const MhcConfig& mhc_config, const RuntimeConfig& runtime_config,
                          MhcBuffers& mhc_buffers) {
  const size_t max_token_num = runtime_config.max_step_token_num;
  const size_t mix_hc = (2 + mhc_config.hc_mult) * mhc_config.hc_mult;
  mhc_buffers.mix_h_buffer = buffer_mgr->CreateBufferTensor(
      "mhc_buffers.mix_h_buffer", {MhcLayer::kMaxNumSplits, max_token_num, mix_hc}, TYPE_FP32);
  mhc_buffers.norm_buffer =
      buffer_mgr->CreateBufferTensor("mhc_buffers.norm_buffer", {MhcLayer::kMaxNumSplits, max_token_num}, TYPE_FP32);
  mhc_buffers.h_post_buffer =
      buffer_mgr->CreateBufferTensor("mhc_buffers.h_post_buffer", {max_token_num, mhc_config.hc_mult}, TYPE_FP32);
  mhc_buffers.h_res_buffer = buffer_mgr->CreateBufferTensor(
      "mhc_buffers.h_res_buffer", {max_token_num, mhc_config.hc_mult * mhc_config.hc_mult}, TYPE_FP32);
  return Status();
}

Status Mhc::PreForward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  KLLM_CHECK(input_tensors.size() == 1);
  KLLM_CHECK(output_tensors.size() == 1);

  PROFILE_EVENT_SCOPE(PreForward, fmt::format("Mhc_PreForward_seq{}", input_tensors[0].shape[0]));
  CREATE_BUFFER_SCOPE(mix_h_buffer_tensors, mhc_buffers_.mix_h_buffer);
  CREATE_BUFFER_SCOPE(norm_buffer_tensors, mhc_buffers_.norm_buffer);
  CREATE_BUFFER_SCOPE(h_post_buffer_tensors, mhc_buffers_.h_post_buffer);
  CREATE_BUFFER_SCOPE(h_res_buffer_tensors, mhc_buffers_.h_res_buffer);
  output_tensors.push_back(mix_h_buffer_tensors[0]);
  output_tensors.push_back(norm_buffer_tensors[0]);
  output_tensors.push_back(h_post_buffer_tensors[0]);
  output_tensors.push_back(h_res_buffer_tensors[0]);
  STATUS_CHECK_RETURN(mhc_layer_->Forward({input_tensors[0], fn_weight_, scale_weight_, base_weight_}, output_tensors));
  output_tensors.resize(1);
  return Status();
}

Status Mhc::PreForwardWithNorm(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  KLLM_CHECK(input_tensors.size() == 1);
  KLLM_CHECK(output_tensors.size() == 1);
  PROFILE_EVENT_SCOPE(Mhc, fmt::format("Mhc_PreForwardWithNorm_seq{}", input_tensors[0].shape[0]));

  CREATE_BUFFER_SCOPE(mix_h_buffer_tensors, mhc_buffers_.mix_h_buffer);
  CREATE_BUFFER_SCOPE(norm_buffer_tensors, mhc_buffers_.norm_buffer);
  CREATE_BUFFER_SCOPE(h_post_buffer_tensors, mhc_buffers_.h_post_buffer);
  CREATE_BUFFER_SCOPE(h_res_buffer_tensors, mhc_buffers_.h_res_buffer);
  output_tensors.push_back(mix_h_buffer_tensors[0]);
  output_tensors.push_back(norm_buffer_tensors[0]);
  output_tensors.push_back(h_post_buffer_tensors[0]);
  output_tensors.push_back(h_res_buffer_tensors[0]);
  STATUS_CHECK_RETURN(
      mhc_layer_->Forward({input_tensors[0], fn_weight_, scale_weight_, base_weight_, norm_weight_}, output_tensors));
  output_tensors.resize(1);
  return Status();
}

Status Mhc::PostForward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  KLLM_CHECK(input_tensors.size() == 1);
  KLLM_CHECK(output_tensors.size() == 1);

  PROFILE_EVENT_SCOPE(PostForward, fmt::format("Mhc_PostForward_seq{}", input_tensors[0].shape[0]));

  CREATE_BUFFER_SCOPE(h_post_buffer_tensors, mhc_buffers_.h_post_buffer);
  CREATE_BUFFER_SCOPE(h_res_buffer_tensors, mhc_buffers_.h_res_buffer);
  return mhc_layer_->Forward({input_tensors[0], h_post_buffer_tensors[0], h_res_buffer_tensors[0]}, output_tensors);
}

Status Mhc::InitialForward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  KLLM_CHECK(input_tensors.size() == 1);
  KLLM_CHECK(output_tensors.size() == 1);

  return mhc_layer_->Forward(input_tensors, output_tensors);
}

Status Mhc::FinalForward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  KLLM_CHECK(input_tensors.size() == 1);
  KLLM_CHECK(output_tensors.size() == 1);

  CREATE_BUFFER_SCOPE(mix_h_buffer_tensors, mhc_buffers_.mix_h_buffer);
  CREATE_BUFFER_SCOPE(norm_buffer_tensors, mhc_buffers_.norm_buffer);
  output_tensors.push_back(mix_h_buffer_tensors[0]);
  output_tensors.push_back(norm_buffer_tensors[0]);
  STATUS_CHECK_RETURN(mhc_layer_->Forward({input_tensors[0], fn_weight_, scale_weight_, base_weight_}, output_tensors));
  output_tensors.resize(1);
  return Status();
}

}  // namespace ksana_llm
