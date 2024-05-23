/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <stdint.h>

#include "acl/acl.h"

namespace llm_kernels {
namespace ascend {

struct RotaryEmbeddingTilingConfig {
  uint32_t seq_len = 0;
  uint32_t hidden_units_num = 0;
  int rotary_dim = 0;

  int num_heads = 0;
  int num_kv_heads = 0;
  int head_size = 0;
  // 0: for query, 1:for key
  int mode = 0;
};

}  // namespace ascend
}  // namespace llm_kernels