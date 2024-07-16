/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#pragma once

#ifdef ENABLE_ACL_ATB

#  include "atb/atb_infer.h"

namespace llm_kernels {
namespace utils {

class ATBOperationExecutor {
 public:
  ATBOperationExecutor(){};

  ~ATBOperationExecutor() { ResetVariantPack(); };

  template <typename OPParamType>
  void Init(const int rank, const OPParamType& param) {
    rank_ = rank;
    ATB_CHECK_RET(atb::CreateOperation(param, &operation_));
  }

  void ResetVariantPack();

  void SetInputTensor(void* addr_ptr, const std::vector<size_t> shape, const aclDataType dtype);

  void SetOutputTensor(void* addr_ptr, const std::vector<size_t> shape, const aclDataType dtype);

  void Run(atb::Context* context, void (*ws_func)(size_t, void**));

 private:
  int rank_{0};
  atb::Operation* operation_{nullptr};
  void* workspace_ptr_{nullptr};
  size_t workspace_size_{0};
  uint64_t in_tensor_num_{0};
  uint64_t out_tensor_num_{0};
  atb::VariantPack variant_pack_;
};

}  // namespace utils
}  // namespace llm_kernels

#endif