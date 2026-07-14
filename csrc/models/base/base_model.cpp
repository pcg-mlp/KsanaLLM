/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/base/base_model.h"

namespace ksana_llm {

BaseModel::BaseModel() {}

BaseModel::~BaseModel() { buffer_mgr_.ReleaseBufferTensors(); }

}  // namespace ksana_llm
