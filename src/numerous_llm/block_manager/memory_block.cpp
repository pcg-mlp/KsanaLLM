/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/block_manager/memory_block.h"

namespace numerous_llm {

bool KvCacheBlock::IsEmpty() { return token_num == 0; }

bool KvCacheBlock::IsFull() { return token_num == block_size; }

size_t KvCacheBlock::GetEmptySlotNum() { return block_size - token_num; }

int KvCacheBlock::GetLastTokenId() {
  if (!token_ids.empty()) {
    return token_ids.back();
  }
  return -1;
}

std::vector<int> KvCacheBlock::GetTokenIds() { return token_ids; }

} // namespace numerous_llm
