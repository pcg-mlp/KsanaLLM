/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/utils/memory_utils.h"

#include <memory>

#include "ksana_llm/utils/device_utils.h"
#include "ksana_llm/utils/ret_code.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

static BlockManagerInterface* g_block_manager = nullptr;

void SetBlockManager(BlockManagerInterface* block_manager) { g_block_manager = block_manager; }

BlockManagerInterface* GetBlockManager() { return g_block_manager; }

Status GetDeviceMemoryInfo(MemoryDevice device, size_t* free, size_t* total) {
  MemGetInfo(free, total);
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
    // convert kB to bytes.
    *free *= 1024, *total *= 1024;
    return Status();
  }

  return Status(RET_RUNTIME, "Get host memory info failed.");
}

void GetWorkSpaceImpl(size_t size, void** ws_addr) {
  if (size > 0) {
    static int block_id = -1;
    static size_t block_size = 0;
    if (block_size < size) {
      if (block_id >= 0) {
        GetBlockManager()->FreeContiguous(block_id);
      }
      GetBlockManager()->AllocateContiguous(size, block_id);
      block_size = size;
    }

    GetBlockManager()->GetContiguousPtr(block_id, *ws_addr);
  }
}

WorkSpaceFunc GetWorkSpaceFunc() { return GetWorkSpaceImpl; }

}  // namespace ksana_llm
