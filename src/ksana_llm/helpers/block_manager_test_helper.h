/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <filesystem>
#include "ksana_llm/block_manager/block_manager.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/memory_utils.h"
#include "ksana_llm/utils/singleton.h"

namespace ksana_llm {

inline void InitTestBlockManager(Environment* env) {
  BlockManagerConfig block_manager_config;
  env->InitializeBlockManagerConfig();
  env->GetBlockManagerConfig(block_manager_config);

  int tp_para = env->GetTensorParallelSize();
  BlockManager* block_manager = new BlockManager(block_manager_config, std::make_shared<Context>(tp_para, 1));
  SetBlockManager(block_manager);
}

}  // namespace ksana_llm
