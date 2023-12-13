/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

namespace numerous_llm {

enum InferStage {
  // The context decode stage.
  STAGE_CONTEXT,

  // The decode stage.
  STATE_DECODE,
};

}  // namespace numerous_llm