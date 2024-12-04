/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/layers/activation_layer.h"
#include "ksana_llm/models/common/common_model.h"
#include "ksana_llm/models/common/common_weight.h"
#include "ksana_llm/runtime/forward_request.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/utils.h"

namespace ksana_llm {

template <typename T>
class __attribute__((visibility("hidden"))) GPTModel : public CommonModel<T> {
 public:
  GPTModel(const ModelConfig& model_config, const int rank, std::shared_ptr<Context> context,
           std::shared_ptr<BaseWeight> base_weight);

 protected:
  Status LayerNormForward(const std::string& layer_name, std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                          const std::vector<Tensor>& layernorm_input, std::vector<Tensor>& layernorm_output) override;

  // refer to
  // github huggingface/transformers main/src/transformers/models/openai/modeling_openai.py#L130
  Status CommonAttention(const int layer_idx, std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                         const std::vector<Tensor>& attention_input, const bool is_multi_token_forward) override;

  // refer to
  // github huggingface/transformers main/src/transformers/models/openai/modeling_openai.py#L223
  Status CommonMlp(const int layer_idx, std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                   const std::vector<Tensor>& mlp_input, const bool is_multi_token_forward) override;

  Status EmbedTokensUseGpu(Tensor& embedding_weight) override;

 protected:
  using CommonModel<T>::context_;
  using CommonModel<T>::rank_;

  using CommonModel<T>::model_config_;

  using CommonModel<T>::model_input_;
  using CommonModel<T>::model_output_;
  using CommonModel<T>::model_communicator_;

  using CommonModel<T>::emb_lookup_layer_;
  using CommonModel<T>::layernorm_layer_;
  using CommonModel<T>::add_layer_;
  using CommonModel<T>::attn_qkv_proj_layer_;
  using CommonModel<T>::attn_o_proj_layer_;
  using CommonModel<T>::mlp_gate_proj_layer_;
  using CommonModel<T>::mlp_down_proj_layer_;

  using CommonModel<T>::hidden_buffer_0_;
  using CommonModel<T>::hidden_buffer_1_;
  using CommonModel<T>::residual_buffer_;
  using CommonModel<T>::reduce_buffer_;

  std::shared_ptr<BaseLayer> activation_layer_;
};

}  // namespace ksana_llm
