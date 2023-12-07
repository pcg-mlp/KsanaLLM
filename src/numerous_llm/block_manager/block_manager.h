/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "numerous_llm/utils/environment.h"
#include "numerous_llm/utils/status.h"

namespace numerous_llm {

class BlockManager {
 public:
  BlockManager(const BlockManagerConfig& block_manager_config);

  // Get block pointer.
  Status GetBlockPtrs(const std::vector<int>& blocks, std::vector<void*>& addrs);
};

}  // namespace numerous_llm
