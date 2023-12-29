/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/utils/memory_utils.h"
#include <memory>

namespace numerous_llm {

static BlockManager* g_block_manager = nullptr;

void SetBlockManager(BlockManager* block_manager) { g_block_manager = block_manager; }

BlockManager* GetBlockManager() { return g_block_manager; }

}  // namespace numerous_llm
