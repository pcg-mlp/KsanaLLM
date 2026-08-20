/* Copyright 2026 Tencent Inc.  All rights reserved.

   BKCL (Baidu Kunlun Communication Library) error checking utilities.
   Mirrors device/nvidia/nccl_utils.h and device/ascend/hccl_utils.h.
==============================================================================*/
#pragma once

#include <bkcl.h>  // ${XMLIR_DIR}/xccl/include/bkcl.h

#include <cstring>
#include <string>

#include "device/device_types.h"
#include "fmt/core.h"
#include "utils/logger.h"
#include "utils/ret_code.h"

namespace ksana_llm {

// BKCL error code → human-readable string.
// 来源：${XMLIR_DIR}/xccl/include/bkcl.h 里 BKCLResult_t 的 5 个枚举值。
inline const char* GetBKCLErrorString(BKCLResult_t code) {
  switch (code) {
    case BKCL_SUCCESS:
      return "BKCL_SUCCESS";
    case BKCL_RUNTIME_ERROR:
      return "BKCL_RUNTIME_ERROR (kunlun runtime returned unexpected code)";
    case BKCL_SYSTEM_ERROR:
      return "BKCL_SYSTEM_ERROR (no enough memory / no NIC)";
    case BKCL_INTERNAL_ERROR:
      return "BKCL_INTERNAL_ERROR";
    case BKCL_INVALID_ARGUMENT:
      return "BKCL_INVALID_ARGUMENT";
    default:
      return "BKCL_UNKNOWN_ERROR";
  }
}

inline void CheckBKCLError(BKCLResult_t result, const char* call, const char* file, const int line) {
  if (result != BKCL_SUCCESS) {
    KLLM_LOG_ERROR << fmt::format("BKCL error {}: {} at {}:{}", GetBKCLErrorString(result), call, file, line);
    abort();
    exit(RetCode::RET_INVALID_ARGUMENT);
  }
}

#define BKCL_CHECK(val) ::ksana_llm::CheckBKCLError((val), #val, __FILE__, __LINE__)

// ---------------------------------------------------------------------------
// BKCLParam: mirrors NCCLParam in device/nvidia/nccl_utils.h
// ---------------------------------------------------------------------------

struct BKCLParam {
  int rank{0};
  int world_size{1};
  BKCLUniqueId bkcl_uid;
  BKCLContext_t bkcl_comm{nullptr};

  BKCLParam() : rank(0), world_size(1), bkcl_comm(nullptr) { memset(&bkcl_uid, 0, sizeof(bkcl_uid)); }
  BKCLParam(int rank, int world_size) : rank(rank), world_size(world_size), bkcl_comm(nullptr) {
    memset(&bkcl_uid, 0, sizeof(bkcl_uid));
  }
  std::string toString() {
    return fmt::format("BKCLParam: rank={}, world_size={}, bkcl_comm={}", rank, world_size,
                       reinterpret_cast<uintptr_t>(bkcl_comm));
  }
};

inline BKCLResult_t DestroyBKCLParam(BKCLParam& param) {
  BKCLResult_t status = BKCL_SUCCESS;
  if (param.bkcl_comm != nullptr) {
    status = bkcl_destroy_context(param.bkcl_comm);
    param.bkcl_comm = nullptr;
  }
  return status;
}

inline BKCLUniqueId GenerateBKCLUniqueID() {
  BKCLUniqueId uid;
  BKCL_CHECK(bkcl_get_unique_id(&uid));
  return uid;
}

inline BKCLDataType GetBkclDataType(DataType dtype) {
  switch (dtype) {
    case DataType::TYPE_FP32:
      return BKCL_FLOAT;
    case DataType::TYPE_FP16:
      return BKCL_FLOAT16;
    case DataType::TYPE_BF16:
      return BKCL_BFLOAT16;
    case DataType::TYPE_FP64:
      return BKCL_FLOAT64;
    case DataType::TYPE_INT32:
      return BKCL_INT32;
    case DataType::TYPE_INT64:
      return BKCL_INT64;
    case DataType::TYPE_UINT8:
      return BKCL_UINT8;
    default:
      KLLM_LOG_ERROR << "Unsupported DataType for BKCL: " << static_cast<int>(dtype);
      return BKCL_FLOAT16;
  }
}

}  // namespace ksana_llm
