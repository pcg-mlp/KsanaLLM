/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <acl/acl.h>
#include <acl/acl_base.h>
#include <acl/acl_rt.h>

#include "ksana_llm/utils/logger.h"

namespace ksana_llm {

// ref: https://www.hiascend.com/document/detail/zh/canncommercial/5046/windowsversion/windowsug/aclcppdevg_03_0516.html
static const char* GetACLErrorString(aclError error) {
  switch (error) {
    case ACL_SUCCESS:
      return "ACL_SUCCESS";
    case ACL_ERROR_INVALID_PARAM:
      return "ACL_ERROR_INVALID_PARAM";
    case ACL_ERROR_UNINITIALIZE:
      return "ACL_ERROR_UNINITIALIZE";
    case ACL_ERROR_REPEAT_INITIALIZE:
      return "ACL_ERROR_REPEAT_INITIALIZE";
    case ACL_ERROR_INVALID_FILE:
      return "ACL_ERROR_INVALID_FILE";
    case ACL_ERROR_WRITE_FILE:
      return "ACL_ERROR_WRITE_FILE";
    case ACL_ERROR_INVALID_FILE_SIZE:
      return "ACL_ERROR_INVALID_FILE_SIZE";
    case ACL_ERROR_PARSE_FILE:
      return "ACL_ERROR_PARSE_FILE";
    case ACL_ERROR_FILE_MISSING_ATTR:
      return "ACL_ERROR_FILE_MISSING_ATTR";
    case ACL_ERROR_FILE_ATTR_INVALID:
      return "ACL_ERROR_FILE_ATTR_INVALID";
    case ACL_ERROR_INVALID_DUMP_CONFIG:
      return "ACL_ERROR_INVALID_DUMP_CONFIG";
    case ACL_ERROR_INVALID_MODEL_ID:
      return "ACL_ERROR_INVALID_MODEL_ID";
    case ACL_ERROR_DESERIALIZE_MODEL:
      return "ACL_ERROR_DESERIALIZE_MODEL";
    case ACL_ERROR_PARSE_MODEL:
      return "ACL_ERROR_PARSE_MODEL";
    case ACL_ERROR_READ_MODEL_FAILURE:
      return "ACL_ERROR_READ_MODEL_FAILURE";
    case ACL_ERROR_MODEL_SIZE_INVALID:
      return "ACL_ERROR_MODEL_SIZE_INVALID";
    case ACL_ERROR_MODEL_MISSING_ATTR:
      return "ACL_ERROR_MODEL_MISSING_ATTR";
    case ACL_ERROR_MODEL_INPUT_NOT_MATCH:
      return "ACL_ERROR_MODEL_INPUT_NOT_MATCH";
    case ACL_ERROR_MODEL_OUTPUT_NOT_MATCH:
      return "ACL_ERROR_MODEL_OUTPUT_NOT_MATCH";
    case ACL_ERROR_MODEL_NOT_DYNAMIC:
      return "ACL_ERROR_MODEL_NOT_DYNAMIC";
    case ACL_ERROR_OP_TYPE_NOT_MATCH:
      return "ACL_ERROR_OP_TYPE_NOT_MATCH";
    case ACL_ERROR_OP_INPUT_NOT_MATCH:
      return "ACL_ERROR_OP_INPUT_NOT_MATCH";
    case ACL_ERROR_OP_OUTPUT_NOT_MATCH:
      return "ACL_ERROR_OP_OUTPUT_NOT_MATCH";
    case ACL_ERROR_OP_ATTR_NOT_MATCH:
      return "ACL_ERROR_OP_ATTR_NOT_MATCH";
    case ACL_ERROR_OP_NOT_FOUND:
      return "ACL_ERROR_OP_NOT_FOUND";
    case ACL_ERROR_OP_LOAD_FAILED:
      return "ACL_ERROR_OP_LOAD_FAILED";
    case ACL_ERROR_UNSUPPORTED_DATA_TYPE:
      return "ACL_ERROR_UNSUPPORTED_DATA_TYPE";
    case ACL_ERROR_FORMAT_NOT_MATCH:
      return "ACL_ERROR_FORMAT_NOT_MATCH";
    case ACL_ERROR_BIN_SELECTOR_NOT_REGISTERED:
      return "ACL_ERROR_BIN_SELECTOR_NOT_REGISTERED";
    case ACL_ERROR_KERNEL_NOT_FOUND:
      return "ACL_ERROR_KERNEL_NOT_FOUND";
    case ACL_ERROR_BIN_SELECTOR_ALREADY_REGISTERED:
      return "ACL_ERROR_BIN_SELECTOR_ALREADY_REGISTERED";
    case ACL_ERROR_KERNEL_ALREADY_REGISTERED:
      return "ACL_ERROR_KERNEL_ALREADY_REGISTERED";
    case ACL_ERROR_INVALID_QUEUE_ID:
      return "ACL_ERROR_INVALID_QUEUE_ID";
    case ACL_ERROR_REPEAT_SUBSCRIBE:
      return "ACL_ERROR_REPEAT_SUBSCRIBE";
    case ACL_ERROR_STREAM_NOT_SUBSCRIBE:
      return "ACL_ERROR_STREAM_NOT_SUBSCRIBE";
    case ACL_ERROR_WAIT_CALLBACK_TIMEOUT:
      return "ACL_ERROR_WAIT_CALLBACK_TIMEOUT";
    case ACL_ERROR_REPEAT_FINALIZE:
      return "ACL_ERROR_REPEAT_FINALIZE";
    case ACL_ERROR_NOT_STATIC_AIPP:
      return "ACL_ERROR_NOT_STATIC_AIPP";
    case ACL_ERROR_COMPILING_STUB_MODE:
      return "ACL_ERROR_COMPILING_STUB_MODE";
    case ACL_ERROR_GROUP_NOT_SET:
      return "ACL_ERROR_GROUP_NOT_SET";
    case ACL_ERROR_GROUP_NOT_CREATE:
      return "ACL_ERROR_GROUP_NOT_CREATE";
    case ACL_ERROR_DUMP_ALREADY_RUN:
      return "ACL_ERROR_DUMP_ALREADY_RUN";
    case ACL_ERROR_DUMP_NOT_RUN:
      return "ACL_ERROR_DUMP_NOT_RUN";
    case ACL_ERROR_INVALID_MAX_OPQUEUE_NUM_CONFIG:
      return "ACL_ERROR_INVALID_MAX_OPQUEUE_NUM_CONFIG";
    case ACL_ERROR_INVALID_OPP_PATH:
      return "ACL_ERROR_INVALID_OPP_PATH";
    case ACL_ERROR_OP_UNSUPPORTED_DYNAMIC:
      return "ACL_ERROR_OP_UNSUPPORTED_DYNAMIC";
    case ACL_ERROR_RELATIVE_RESOURCE_NOT_CLEARED:
      return "ACL_ERROR_RELATIVE_RESOURCE_NOT_CLEARED";
    case ACL_ERROR_BAD_ALLOC:
      return "ACL_ERROR_BAD_ALLOC";
    case ACL_ERROR_API_NOT_SUPPORT:
      return "ACL_ERROR_API_NOT_SUPPORT";
    case ACL_ERROR_INVALID_DEVICE:
      return "ACL_ERROR_INVALID_DEVICE";
    case ACL_ERROR_MEMORY_ADDRESS_UNALIGNED:
      return "ACL_ERROR_MEMORY_ADDRESS_UNALIGNED";
    case ACL_ERROR_RESOURCE_NOT_MATCH:
      return "ACL_ERROR_RESOURCE_NOT_MATCH";
    case ACL_ERROR_INVALID_RESOURCE_HANDLE:
      return "ACL_ERROR_INVALID_RESOURCE_HANDLE";
    case ACL_ERROR_FEATURE_UNSUPPORTED:
      return "ACL_ERROR_FEATURE_UNSUPPORTED";
    case ACL_ERROR_STORAGE_OVER_LIMIT:
      return "ACL_ERROR_STORAGE_OVER_LIMIT";
    case ACL_ERROR_INTERNAL_ERROR:
      return "ACL_ERROR_INTERNAL_ERROR";
    case ACL_ERROR_FAILURE:
      return "ACL_ERROR_FAILURE";
    case ACL_ERROR_GE_FAILURE:
      return "ACL_ERROR_GE_FAILURE";
    case ACL_ERROR_RT_FAILURE:
      return "ACL_ERROR_RT_FAILURE";
    case ACL_ERROR_DRV_FAILURE:
      return "ACL_ERROR_DRV_FAILURE";
  }
  return "UNKNOWN";
}

template <typename T>
void CheckACLError(T result, const char* func, const char* file, const int line) {
  if (result != ACL_SUCCESS) {
    NLLM_LOG_ERROR << fmt::format("ACL runtime error: {} {}:{}@{}", GetACLErrorString(result), file, line, func);
    abort();
    exit(RetCode::RET_INVALID_ARGUMENT);
  }
}

#define ACL_CHECK(val) CheckACLError((val), #val, __FILE__, __LINE__)

}  // namespace ksana_llm