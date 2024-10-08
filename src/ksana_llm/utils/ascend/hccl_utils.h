/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <hccl/hccl.h>
#include <hccl/hccl_types.h>

#include "fmt/core.h"

#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/ret_code.h"

namespace ksana_llm {

#define HCCL_CHECK(cmd)                                                                                             \
  do {                                                                                                              \
    HcclResult ret = cmd;                                                                                           \
    if (ret != HCCL_SUCCESS) {                                                                                      \
      KLLM_LOG_ERROR << fmt::format("HCCL runtime error code:{} {}:{}", static_cast<int>(ret), __FILE__, __LINE__); \
      exit(RetCode::RET_INVALID_ARGUMENT);                                                                          \
    }                                                                                                               \
  } while (0)

struct HCCLParam {
  int rank{0};
  int world_size{1};

  HcclComm hccl_comm{nullptr};

  HCCLParam() : rank(0), world_size(1), hccl_comm(nullptr) {}
  HCCLParam(int rank, int world_size) : rank(rank), world_size(world_size) {}
  HCCLParam(HCCLParam const& param) : rank(param.rank), world_size(param.world_size), hccl_comm(param.hccl_comm) {}
  std::string toString() {
    return fmt::format("HCCLParam: rank={}, world_size=%{}, hccl_comm={}", rank, world_size,
                       reinterpret_cast<uintptr_t>(hccl_comm));
  }
};

}  // namespace ksana_llm