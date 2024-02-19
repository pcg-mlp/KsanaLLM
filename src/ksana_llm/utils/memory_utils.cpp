/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/utils/memory_utils.h"
#include <memory>
#include "ksana_llm/utils/ret_code.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

static BlockManager* g_block_manager = nullptr;

void SetBlockManager(BlockManager* block_manager) { g_block_manager = block_manager; }

BlockManager* GetBlockManager() { return g_block_manager; }

Status GetDeviceMemoryInfo(size_t* free, size_t* total) {
  CUDA_CHECK(cudaMemGetInfo(free, total));
  return Status();
}

Status GetHostMemoryInfo(size_t* free, size_t* total) {
  constexpr const char* memory_file = "/proc/meminfo";

  bool found_free = false;
  bool found_total = false;

  std::string token;
  std::ifstream file(memory_file);
  while (file >> token) {
    if (token == "MemTotal:") {
      if (file >> *total) {
        found_total = true;
      } else {
        return Status(RET_RUNTIME, "Get total memory failed.");
      }
    } else if (token == "MemAvailable:") {
      if (file >> *free) {
        found_free = true;
      } else {
        return Status(RET_RUNTIME, "Get free memory failed.");
      }
    }

    // Ignore the rest of the line
    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }

  if (found_free && found_total) {
    return Status();
  }

  return Status(RET_RUNTIME, "Get host memory info failed.");
}

}  // namespace ksana_llm
