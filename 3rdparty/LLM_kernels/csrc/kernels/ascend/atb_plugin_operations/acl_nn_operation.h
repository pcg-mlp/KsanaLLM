/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <string>
#include "acl_nn_tensor.h"
#include "atb/atb_infer.h"
#include "atb/operation.h"
#include "utils.h"

#include "csrc/utils/ascend/common.h"

namespace llm_kernels {
namespace ascend {

class AclNNOperation : public atb::Operation {
 public:
  explicit AclNNOperation(const std::string &op_name);
  ~AclNNOperation() override;
  std::string GetName() const override;
  atb::Status Setup(const atb::VariantPack &variant_pack, uint64_t &workspace_size, atb::Context *context) override;
  atb::Status Execute(const atb::VariantPack &variant_pack, uint8_t *workspace, uint64_t workspace_size,
                      atb::Context *context) override;
  void DestroyOperation();
  atb::Status CreateAclNNOpCache(const atb::VariantPack &variant_pack);
  virtual int CreateAclNNVariantPack(const atb::VariantPack &variant_pack) = 0;
  virtual int SetAclNNWorkspaceExecutor() = 0;
  virtual int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) = 0;
  std::string op_name_;
  atb::SVector<AclNNTensor> acl_in_tensors;
  atb::SVector<AclNNTensor> acl_out_tensors;
  aclOpExecutor *acl_executor;
  uint64_t workspace_size;
};
}  // namespace ascend
}  // namespace llm_kernels
