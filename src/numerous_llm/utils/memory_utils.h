/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <vector>

#include "numerous_llm/block_manager/block_manager.h"
#include "numerous_llm/utils/singleton.h"

namespace numerous_llm {

// Get block pointer.
template <typename T>
std::vector<T*> GetBlockPtrs(const std::vector<int>& blocks) {
  std::vector<void*> addrs;
  Singleton<BlockManager>::GetInstance()->GetBlockPtrs(blocks, addrs);
  return addrs;
}

}  // namespace numerous_llm
