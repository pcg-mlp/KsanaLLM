/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

namespace numerous_llm {

enum RetCode {
  // All things ok.
  RET_SUCCESS = 0,

  // The input argument is invalid.
  RET_INVALID_ARGUMENT = 1,

  // something not in above values.
  RET_UNKNOWN = 255,
};

} // namespace numerous_llm
