/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/models/common/common_model.h"
#include "ksana_llm/models/common/common_weight.h"
#include "ksana_llm/runtime/forward_request.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/utils.h"

namespace ksana_llm {

template <typename T>
class __attribute__((visibility("hidden"))) Qwen2VLModel : public CommonModel<T> {
 public:
  Qwen2VLModel(const ModelConfig& model_config, const int rank, std::shared_ptr<Context> context,
               std::shared_ptr<BaseWeight> base_weight);

 protected:
  Status FlashAttentionForward(const int layer_idx) override;

  Status LoadEmbeddings(std::vector<ForwardRequest>& forward_reqs) override;

 protected:
  using CommonModel<T>::context_;
  using CommonModel<T>::rank_;

  using CommonModel<T>::prefix_caching_enabled_;

  using CommonModel<T>::model_config_;

  using CommonModel<T>::model_input_;

  using CommonModel<T>::flash_attention_layers_;

  using CommonModel<T>::hidden_buffer_0_;
  using CommonModel<T>::hidden_buffer_1_;

  using CommonModel<T>::forward_shape_;
};

}  // namespace ksana_llm
