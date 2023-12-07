/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "numerous_llm/models/base/base_model.h"
#include "numerous_llm/utils/status.h"

namespace numerous_llm {

class Llama : public BaseModel {
 public:
  explicit Llama();
  ~Llama() = default;

  // The prefill stage.
  Status ContextDecode();

  // The decode stage.
  Status Decode();
};

}  // namespace numerous_llm
