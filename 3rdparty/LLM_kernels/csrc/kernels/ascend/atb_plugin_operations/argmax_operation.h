/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#pragma once

#include "acl_nn_operation.h"
#include "acl_nn_tensor.h"

namespace llm_kernels {
namespace ascend {

struct ArgmaxParam {
  int64_t dim = 0;
};

class ArgmaxOperation : public AclNNOperation {
 public:
  explicit ArgmaxOperation(const std::string &name, ArgmaxParam param);

  ~ArgmaxOperation() override;
  atb::Status InferShape(const atb::SVector<atb::TensorDesc> &in_tensor_descs,
                         atb::SVector<atb::TensorDesc> &out_tensor_descs) const override;
  uint32_t GetInputNum() const override;
  uint32_t GetOutputNum() const override;

 private:
  int CreateAclNNVariantPack(const atb::VariantPack &variant_pack) override;
  int SetAclNNWorkspaceExecutor() override;
  int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;

  int CreateAclNNInTensorVariantPack(const atb::VariantPack &variant_pack);
  int CreateAclNNOutTensorVariantPack(const atb::VariantPack &variant_pack);
  ArgmaxParam param_;
};
}  // namespace ascend
}  // namespace llm_kernels
