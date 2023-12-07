/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "numerous_llm/utils/status.h"

namespace numerous_llm {

class Llama {
 public:
  // The prefill stage.
  Status ContextDecode();

  // The decode stage.
  Status Decode();
};

}  // namespace numerous_llm
