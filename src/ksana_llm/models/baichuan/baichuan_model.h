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
class __attribute__ ((visibility("hidden"))) BaichuanModel : public BaseModel {
 public:
  BaichuanModel(const ModelConfig& model_config, const int rank, std::shared_ptr<Context> context);

  float* GetLogitsPtr();

  // The prefill stage.
  Status ContextDecode(std::shared_ptr<ksana_llm::BaseWeight>& base_weight, std::vector<ForwardRequest>& forward_reqs);

  // The decode stage.
  Status Decode(std::shared_ptr<ksana_llm::BaseWeight>& base_weight, std::vector<ForwardRequest>& forward_reqs);

 private:
  // The common model instance.
  std::shared_ptr<CommonModel<T>> common_model_ = nullptr;
};

}  // namespace ksana_llm
