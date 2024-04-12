/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/utils/singleton.h"
#include "src/ksana_llm/utils/memory_utils.h"

#include <filesystem>

#include "test.h"

namespace ksana_llm {

TEST(WorkspaceTest, GetWorkspace) {
  std::filesystem::path current_path = __FILE__;
  std::filesystem::path parent_path = current_path.parent_path();
  std::filesystem::path config_path_relate = parent_path / "../../../examples/llama7b/ksana_llm.yaml";
  std::string config_path = std::filesystem::absolute(config_path_relate).string();

  Singleton<Environment>::GetInstance()->ParseConfig(config_path);

  BlockManagerConfig block_manager_config;
  Singleton<Environment>::GetInstance()->GetBlockManagerConfig(block_manager_config);
  BlockManager* block_manager = new BlockManager(block_manager_config, std::make_shared<Context>(1, 1));
  SetBlockManager(block_manager);

  WorkSpaceFunc f = GetWorkSpaceFunc();

  void* ws_addr_1 = nullptr;
  f(1024, &ws_addr_1);

  void* ws_addr_2 = nullptr;
  f(2048, &ws_addr_2);
  EXPECT_NE(ws_addr_1, ws_addr_2);

  void* ws_addr_3 = nullptr;
  f(1536, &ws_addr_3);
  EXPECT_EQ(ws_addr_2, ws_addr_3);
}

}  // namespace ksana_llm
