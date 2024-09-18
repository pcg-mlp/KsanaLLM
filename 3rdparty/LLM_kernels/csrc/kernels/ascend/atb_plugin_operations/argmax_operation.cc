/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include "argmax_operation.h"
#include <securec.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <sstream>
#include "acl/acl.h"
#include "acl_nn_tensor.h"
#include "aclnnop/aclnn_argmax.h"
#include "utils.h"

namespace llm_kernels {
namespace ascend {

ArgmaxOperation::ArgmaxOperation(const std::string &name, ArgmaxParam param) : AclNNOperation(name), param_(param) {}

ArgmaxOperation::~ArgmaxOperation() {}

atb::Status ArgmaxOperation::InferShape(const atb::SVector<atb::TensorDesc> &in_tensor_descs,
                                        atb::SVector<atb::TensorDesc> &out_tensor_descs) const {
  out_tensor_descs.at(0).format = in_tensor_descs.at(0).format;
  out_tensor_descs.at(0).dtype = ACL_INT32;
  out_tensor_descs.at(0).shape.dimNum = in_tensor_descs.at(0).shape.dimNum - 1;
  for (size_t i = 0; i < out_tensor_descs.at(0).shape.dimNum; i++) {
    out_tensor_descs.at(0).shape.dims[i] =
        (i < this->param_.dim) ? in_tensor_descs.at(0).shape.dims[i] : in_tensor_descs.at(0).shape.dims[i + 1];
  }
  return 0;
}

uint32_t ArgmaxOperation::GetInputNum() const { return NUM1; }

uint32_t ArgmaxOperation::GetOutputNum() const { return NUM1; }

int ArgmaxOperation::CreateAclNNInTensorVariantPack(const atb::VariantPack &variant_pack) {
  this->acl_in_tensors.resize(GetInputNum());
  for (size_t i = 0; i < this->acl_in_tensors.size(); ++i) {
    AclNNTensor aclnn_tensor;
    aclnn_tensor.tensor_idx = i;
    aclnn_tensor.atb_tensor = variant_pack.inTensors.at(i);
    atb::Tensor squeezed_atb_tensor = SqueezeBatchSeq(variant_pack.inTensors.at(i));

    aclnn_tensor.strides = GetCopyTensorStride(squeezed_atb_tensor.desc.shape);
    aclnn_tensor.tensor = aclCreateTensor(squeezed_atb_tensor.desc.shape.dims, squeezed_atb_tensor.desc.shape.dimNum,
                                          squeezed_atb_tensor.desc.dtype, aclnn_tensor.strides.data(), 0,
                                          squeezed_atb_tensor.desc.format, squeezed_atb_tensor.desc.shape.dims,
                                          squeezed_atb_tensor.desc.shape.dimNum, squeezed_atb_tensor.deviceData);

    if (aclnn_tensor.tensor == nullptr) {
      return atb::ERROR_INTERNAL_ERROR;
    }
    this->acl_in_tensors[i] = aclnn_tensor;
  }
  return atb::NO_ERROR;
}

int ArgmaxOperation::CreateAclNNOutTensorVariantPack(const atb::VariantPack &variant_pack) {
  this->acl_out_tensors.resize(GetOutputNum());
  for (size_t i = 0; i < this->acl_out_tensors.size(); ++i) {
    AclNNTensor aclnn_tensor;
    aclnn_tensor.tensor_idx = i;
    aclnn_tensor.atb_tensor = variant_pack.outTensors.at(i);
    atb::Tensor squeezed_atb_tensor = SqueezeBatchSeq(variant_pack.outTensors.at(i));
    aclnn_tensor.strides = GetCopyTensorStride(squeezed_atb_tensor.desc.shape);
    aclnn_tensor.tensor = aclCreateTensor(squeezed_atb_tensor.desc.shape.dims, squeezed_atb_tensor.desc.shape.dimNum,
                                          squeezed_atb_tensor.desc.dtype, aclnn_tensor.strides.data(), 0,
                                          squeezed_atb_tensor.desc.format, squeezed_atb_tensor.desc.shape.dims,
                                          squeezed_atb_tensor.desc.shape.dimNum, squeezed_atb_tensor.deviceData);
    if (aclnn_tensor.tensor == nullptr) {
      return atb::ERROR_INTERNAL_ERROR;
    }
    this->acl_out_tensors[i] = aclnn_tensor;
  }
  return atb::NO_ERROR;
}

atb::Status ArgmaxOperation::CreateAclNNVariantPack(const atb::VariantPack &variant_pack) {
  int ret = 0;
  ret = CreateAclNNInTensorVariantPack(variant_pack);
  if (ret != 0) {
    return ret;
  }

  ret = CreateAclNNOutTensorVariantPack(variant_pack);
  if (ret != 0) {
    return ret;
  }

  return atb::NO_ERROR;
}

int ArgmaxOperation::SetAclNNWorkspaceExecutor() {
  bool keepdim = false;
  int ret = aclnnArgMaxGetWorkspaceSize(this->acl_in_tensors.at(0).tensor, this->param_.dim, keepdim,
                                        this->acl_out_tensors.at(0).tensor, &this->workspace_size, &this->acl_executor);
  return ret;
}

int ArgmaxOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) {
  int ret = aclnnArgMax(workspace, this->workspace_size, this->acl_executor, stream);
  return ret;
}
}  // namespace ascend
}  // namespace llm_kernels
