/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <assert.h>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "acl/acl.h"
#include "acl/acl_op_compiler.h"
#include "aclnn/acl_meta.h"

namespace llm_kernels {
namespace utils {

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stderr, "[ERROR]  " fmt "\n", ##args)
#define FATAL_LOG(fmt, args...)                  \
  fprintf(stderr, "[FATAL]  " fmt "\n", ##args); \
  assert(false)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

#define ACL_CHECK_RET(expr)                                                          \
  do {                                                                               \
    aclError ret = (expr);                                                           \
    if (ret != ACL_SUCCESS) {                                                        \
      FATAL_LOG("Acl return error %s:%d, with ERROR %d\n", __FILE__, __LINE__, ret); \
    }                                                                                \
  } while (0)

#define ACL_CHECK_OP(a, b, op)                                                                                \
  do {                                                                                                        \
    int64_t ret_a = (a);                                                                                      \
    int64_t ret_b = (b);                                                                                      \
    if (!(ret_a op ret_b)) {                                                                                  \
      FATAL_LOG("Acl check %s(%d) %s %s(%d) fail at %s:%d\n", #a, ret_a, #op, #b, ret_b, __FILE__, __LINE__); \
    }                                                                                                         \
  } while (0)

#define ACL_CHECK_EQ(a, b) ACL_CHECK_OP(a, b, ==)
#define ACL_CHECK_GE(a, b) ACL_CHECK_OP(a, b, >=)
#define ACL_CHECK_GT(a, b) ACL_CHECK_OP(a, b, >)
#define ACL_CHECK_LT(a, b) ACL_CHECK_OP(a, b, <)
#define ACL_CHECK_LE(a, b) ACL_CHECK_OP(a, b, <=)

const std::unordered_map<aclDataType, size_t> DT2LONG = {{aclDataType::ACL_FLOAT16, sizeof(uint16_t)},
                                                         {aclDataType::ACL_INT64, sizeof(int64_t)},
                                                         {aclDataType::ACL_FLOAT, sizeof(float)}};

int64_t GetShapeSize(const std::vector<int64_t>& shape);

void CalShapeStrides(const std::vector<int64_t>& shape, std::vector<int64_t>& strides);

void ReallocWorkspace(const uint64_t new_size, uint64_t& old_size, void** ws_dev,
                      aclrtMemMallocPolicy policy = ACL_MEM_MALLOC_NORMAL_ONLY);

void CreateAclTensor(const int64_t hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
                     aclFormat dataFormat, aclTensor** tensor);

void CreateAclTensor(const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType, aclFormat dataFormat,
                     aclTensor** tensor);

void CreateAclTensorWithData(const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
                             aclFormat dataFormat, aclTensor** tensor);

void SaveNpy(const aclTensor* tensor, const std::string& filename, aclrtStream& stream);

std::string GetNumpyTypeDesc(aclDataType dtype);

void PrintTensor(const aclTensor* src, aclrtStream& stream, const char* name = "");

void Random(aclTensor* input_tensor, void* workspace_ptr, uint64_t workspace_size, int64_t from, int64_t to,
            int64_t seed, int64_t offset, aclrtStream& stream);

// Define a function to create kernel workspace.
typedef void (*WorkSpaceFunc)(size_t, void**);

void GetTestWorkSpaceFunc(size_t size, void** ws_addr);

}  // namespace utils
}  // namespace llm_kernels
