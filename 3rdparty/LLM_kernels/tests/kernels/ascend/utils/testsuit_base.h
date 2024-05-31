/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <math.h>
#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "csrc/utils/ascend/common.h"

namespace llm_kernels {
namespace ascend {
namespace test {

class AscendTestSuitBase : public testing::Test {
 public:
  static void SetUpTestCase() { ACL_CHECK_RET(aclInit(nullptr)); }

  void SetUp() override { Init(&context, &stream); }

  void TearDown() override {
    if (is_inited) {
      ACL_CHECK_RET(aclrtSetDevice(device));
      ACL_CHECK_RET(aclrtSynchronizeStream(stream));
      ACL_CHECK_RET(aclrtDestroyStream(stream));
      ACL_CHECK_RET(aclrtDestroyContext(context));
    }
  }

  void Init(aclrtContext* context, aclrtStream* stream) {
    // init acl resource
    ACL_CHECK_RET(aclrtSetDevice(default_device));
    ACL_CHECK_RET(aclrtCreateContext(context, default_device));
    ACL_CHECK_RET(aclrtSetCurrentContext(*context));
    ACL_CHECK_RET(aclrtCreateStream(stream));
    aclrtRunMode runMode;
    ACL_CHECK_RET(aclrtGetRunMode(&runMode));
    is_inited = (runMode == ACL_DEVICE);
  }

 protected:
  int32_t default_device{0};
  int32_t device{-1};
  aclrtStream stream;
  aclrtContext context;
  bool is_inited = false;
};

static const float HALF_FLT_MAX = 65504.F;

}  // namespace test
}  // namespace ascend
}  // namespace llm_kernels