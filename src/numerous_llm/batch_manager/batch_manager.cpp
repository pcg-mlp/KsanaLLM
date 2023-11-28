/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/batch_manager/batch_manager.h"

namespace numerous_llm {

Status
BatchManager::Enqueue(int req_id, const std::vector<TensorMap> &tensor_maps,
                      const std::vector<SamplingConfig> &sampling_configs) {
  return Status();
}

} // namespace numerous_llm
