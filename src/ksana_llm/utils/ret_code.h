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

  // The input len exceed max input length.
  RET_EXCEED_LENGTH = 6,

  // handle timeout.
  RET_TIMEOUT = 7,

  // block manager allocate fail
  RET_ALLOCATE_FAIL = 8,

  // block manager free fail
  RET_FREE_FAIL = 9,

  // undefined reference
  RET_UNDEFINED_REFERENCE = 10,

  // iteratioin stopped.
  RET_STOP_ITERATION = 11,

  // The runtime error.
  RET_RUNTIME = 12,

  // something not in above values.
  RET_UNKNOWN = 255,
};

}  // namespace ksana_llm
