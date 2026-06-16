/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/base/base_model.h"

namespace ksana_llm {

BaseModel::BaseModel() {}

BaseModel::~BaseModel() { buffer_mgr_.ReleaseBufferTensors(); }

Status BaseModel::PrepareModelInputGpu(std::vector<ForwardRequest>& /*forward_reqs*/, RunMode /*run_mode*/,
                                       size_t /*model_input_slot*/) {
  return Status();
}

}  // namespace ksana_llm
