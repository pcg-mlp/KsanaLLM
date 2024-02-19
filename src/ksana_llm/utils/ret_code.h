/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

namespace ksana_llm {

enum RetCode {
  // All things ok.
  RET_SUCCESS = 0,

  // The input argument is invalid.
  RET_INVALID_ARGUMENT = 1,

  // The segment error.
  RET_SEGMENT_FAULT = 2,

  // Out of memory error.
  RET_OUT_OF_MEMORY = 3,

  // The service is terminated.
  RET_TERMINATED = 4,

  // The sequence len exceed the capacity.
  RET_EXCEED_CAPACITY = 5,

  // handle timeout.
  RET_TIMEOUT = 6,

  // block manager allocate fail
  RET_ALLOCATE_FAIL = 7,

  // block manager free fail
  RET_FREE_FAIL = 8,

  // undefined reference
  RET_UNDEFINED_REFERENCE = 9,

  // iteratioin stopped.
  RET_STOP_ITERATION = 10,

  // The runtime error.
  RET_RUNTIME = 8,

  // something not in above values.
  RET_UNKNOWN = 255,
};

}  // namespace ksana_llm
