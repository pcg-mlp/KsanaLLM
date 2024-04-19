/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

namespace ksana_llm {

enum InferStage {
  // The context decode stage.
  STAGE_CONTEXT,

  // The decode stage.
  STATE_DECODE,
};

}  // namespace ksana_llm