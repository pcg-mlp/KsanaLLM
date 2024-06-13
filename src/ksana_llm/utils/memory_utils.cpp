/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/utils/memory_utils.h"

#include <memory>
#include <optional>

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
        *total <<= 10;  // convert kB to bytes.
        found_total = true;
      }
    } else if (token == "MemAvailable:") {
      if (file >> *free) {
        *free <<= 10;  // convert kB to bytes.
        found_free = true;
      }
    }

    // Ignore the rest of the line
    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }

  constexpr const char* memory_limit_file_cgroup_v1 = "/sys/fs/cgroup/memory/memory.limit_in_bytes";
  constexpr const char* memory_limit_file_cgroup_v2 = "/sys/fs/cgroup/memory.max";

  std::optional<size_t> cgroup_mem_total;
  for (const auto memory_file : {memory_limit_file_cgroup_v1, memory_limit_file_cgroup_v2}) {
    std::ifstream file(memory_file);
    size_t value;
    if (file >> value) {
      cgroup_mem_total = value;
    }
  }

  constexpr const char* memory_usage_file_cgroup_v1 = "/sys/fs/cgroup/memory/memory.usage_in_bytes";
  constexpr const char* memory_usage_file_cgroup_v2 = "/sys/fs/cgroup/memory.current";

  std::optional<size_t> cgroup_mem_usage;
  for (const auto memory_file : {memory_usage_file_cgroup_v1, memory_usage_file_cgroup_v2}) {
    std::ifstream file(memory_file);
    size_t value;
    if (file >> value) {
      cgroup_mem_usage = value;
    }
  }

  constexpr const char* memory_stat_file_cgroup_v1 = "/sys/fs/cgroup/memory/memory.stat";
  constexpr const char* memory_stat_file_cgroup_v2 = "/sys/fs/cgroup/memory.stat";

  size_t cgroup_cache = 0;
  for (const auto memory_file : {memory_stat_file_cgroup_v1, memory_stat_file_cgroup_v2}) {
    std::ifstream file(memory_file);
    while (file >> token) {
      if (token == "total_inactive_file") {  // cgroup v1
        file >> cgroup_cache;
      } else if (token == "inactive_file") {  // cgroup v2
        file >> cgroup_cache;
      }
    }
  }

  if (cgroup_mem_total.has_value()) {
    if (found_total) {
      // memory.limit_in_bytes returns a very big number if there is no limit
      *total = std::min(*total, cgroup_mem_total.value());
    } else {
      *total = cgroup_mem_total.value();
      found_total = true;
    }

    if (cgroup_mem_usage.has_value()) {
      // Cache is intentionally excluded from memeory usage in Docker
      // Refer to
      // https://github.com/docker/cli/blob/master/cli/command/container/stats_helpers.go#L227
      if (cgroup_cache < cgroup_mem_usage.value()) {
        cgroup_mem_usage.value() -= cgroup_cache;
      }
      if (cgroup_mem_usage.value() <= *total) {
        *free = *total - cgroup_mem_usage.value();
        found_free = true;
      }
    }
  }

  if (found_free && found_total) {
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
