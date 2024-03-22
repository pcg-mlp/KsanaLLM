/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

namespace ksana_llm {

// All the available data types.
enum DataType {
  TYPE_INVALID,
  TYPE_BOOL,
  TYPE_UINT8,
  TYPE_UINT16,
  TYPE_UINT32,
  TYPE_UINT64,
  TYPE_INT8,
  TYPE_INT16,
  TYPE_INT32,
  TYPE_INT64,
  TYPE_FP16,
  TYPE_FP32,
  TYPE_FP64,
  TYPE_BYTES,
  TYPE_BF16,
  TYPE_FP8_E4M3,
  TYPE_STR,
  TYPE_POINTER,
  TYPE_VOID,
};

}  // namespace ksana_llm
