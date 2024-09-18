/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include "acl_nn_operation.h"
#include <cstring>
#include <iostream>
#include "utils.h"

namespace llm_kernels {
namespace ascend {

AclNNOperation::AclNNOperation(const std::string &op_name) : op_name_(op_name) {}

AclNNOperation::~AclNNOperation() { this->DestroyOperation(); }

std::string AclNNOperation::GetName() const { return this->op_name_; }

void AclNNOperation::DestroyOperation() {
  if (this->acl_executor == nullptr) {
    return;
  }

  this->acl_executor = nullptr;

  // 清空aclTensor
  for (size_t i = 0; i < this->acl_in_tensors.size(); ++i) {
    ACL_CHECK_RET(aclDestroyTensor(this->acl_in_tensors[i].tensor));
  }
  this->acl_in_tensors.clear();

  for (size_t i = 0; i < this->acl_out_tensors.size(); ++i) {
    ACL_CHECK_RET(aclDestroyTensor(this->acl_out_tensors[i].tensor));
  }
  this->acl_out_tensors.clear();
}

atb::Status AclNNOperation::Setup(const atb::VariantPack &variant_pack, uint64_t &workspace_size,
                                  atb::Context *context) {
  if (context == nullptr) {
    return atb::ERROR_INVALID_PARAM;
  }

  int ret = CreateAclNNOpCache(variant_pack);
  if (ret != 0) {
    return ret;
  }

  ret = SetAclNNWorkspaceExecutor();
  if (ret != 0) {
    return ret;
  }

  workspace_size = this->workspace_size;
  return atb::NO_ERROR;
}

atb::Status AclNNOperation::CreateAclNNOpCache(const atb::VariantPack &variant_pack) {
  int ret = CreateAclNNVariantPack(variant_pack);
  if (ret != 0) {
    return atb::ERROR_CANN_ERROR;
  }
  return atb::NO_ERROR;
}

atb::Status AclNNOperation::Execute(const atb::VariantPack &variant_pack, uint8_t *workspace, uint64_t workspace_size,
                                    atb::Context *context) {
  if (!context) {
    return atb::ERROR_INVALID_PARAM;
  }

  aclrtStream stream = context->GetExecuteStream();
  if (!stream) {
    return atb::ERROR_INVALID_PARAM;
  }

  int ret = ExecuteAclNNOp(workspace, stream);
  if (ret != 0) {
    return atb::ERROR_CANN_ERROR;
  }
  return atb::NO_ERROR;
}

}  // namespace ascend
}  // namespace llm_kernels
