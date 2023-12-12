/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/block_manager/block_manager.h"
#include "numerous_llm/utils/singleton.h"

namespace numerous_llm {

BlockManager::BlockManager() {
  BlockManagerConfig block_manager_config;
  Singleton<Environment>::GetInstance()->GetBlockManagerConfig(block_manager_config);
}

}  // namespace numerous_llm
