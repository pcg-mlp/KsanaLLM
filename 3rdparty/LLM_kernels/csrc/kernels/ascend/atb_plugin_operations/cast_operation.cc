/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include "cast_operation.h"
#include <securec.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <sstream>
#include "acl/acl.h"
#include "acl_nn_tensor.h"
#include "aclnnop/aclnn_cast.h"
#include "utils.h"

namespace llm_kernels {
namespace ascend {

CastOperation::CastOperation(const std::string &name, CastParam param) : AclNNOperation(name), param_(param) {}

CastOperation::~CastOperation() {}

atb::Status CastOperation::InferShape(const atb::SVector<atb::TensorDesc> &in_tensor_descs,
                                      atb::SVector<atb::TensorDesc> &out_tensor_descs) const {
  out_tensor_descs.at(0).format = in_tensor_descs.at(0).format;
  out_tensor_descs.at(0).dtype = this->param_.dataType;
  out_tensor_descs.at(0).shape.dimNum = in_tensor_descs.at(0).shape.dimNum;
  for (size_t i = 0; i < in_tensor_descs.at(0).shape.dimNum; i++) {
    out_tensor_descs.at(0).shape.dims[i] = in_tensor_descs.at(0).shape.dims[i];
  }
  return 0;
}

uint32_t CastOperation::GetInputNum() const { return NUM1; }

uint32_t CastOperation::GetOutputNum() const { return NUM1; }

int CastOperation::CreateAclNNInTensorVariantPack(const atb::VariantPack &variant_pack) {
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

int CastOperation::CreateAclNNOutTensorVariantPack(const atb::VariantPack &variant_pack) {
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

atb::Status CastOperation::CreateAclNNVariantPack(const atb::VariantPack &variant_pack) {
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

int CastOperation::SetAclNNWorkspaceExecutor() {
  int ret = aclnnCastGetWorkspaceSize(this->acl_in_tensors.at(0).tensor, this->param_.dataType,
                                      this->acl_out_tensors.at(0).tensor, &this->workspace_size, &this->acl_executor);
  return ret;
}

int CastOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) {
  int ret = aclnnCast(workspace, this->workspace_size, this->acl_executor, stream);
  return ret;
}
}  // namespace ascend
}  // namespace llm_kernels
