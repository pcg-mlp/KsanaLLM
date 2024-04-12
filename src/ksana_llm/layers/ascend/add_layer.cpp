/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/add_layer.h"

#include "csrc/kernels/ascend/elementwise/elementwise.h"
#include "csrc/utils/ascend/common.h"
#include "ksana_llm/utils/ascend/acl_utils.h"

namespace ksana_llm {

Status AddLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  auto a = reinterpret_cast<const void*>(input_tensors[0].GetPtr<void>());
  auto b = reinterpret_cast<const void*>(input_tensors[1].GetPtr<void>());
  if (input_tensors[0].shape[0] == input_tensors[1].shape[0]) {
    size_t seq_len = input_tensors[0].shape[0];
    size_t hidden_size = input_tensors[0].shape[1];
    std::vector<int64_t> add_shape = {1, seq_len, hidden_size};

    uint16_t one_in_fp16 = 0b11110000000000;
    aclScalar* add_alpha = aclCreateScalar(reinterpret_cast<void*>(&one_in_fp16), aclDataType::ACL_FLOAT16);

    void* add_out_buf = output_tensors[0].GetPtr<void>();
    aclTensor* add_output = nullptr;
    llm_kernels::utils::CreateAclTensorWithData(add_shape, &add_out_buf, aclDataType::ACL_FLOAT16,
                                                aclFormat::ACL_FORMAT_ND, &add_output);

    void* a_ptr = reinterpret_cast<void*>(input_tensors[0].GetPtr<void>());
    void* b_ptr = reinterpret_cast<void*>(input_tensors[1].GetPtr<void>());
    aclTensor* input_a = nullptr;
    aclTensor* input_b = nullptr;
    llm_kernels::utils::CreateAclTensorWithData(add_shape, &a_ptr, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND,
                                                &input_a);
    llm_kernels::utils::CreateAclTensorWithData(add_shape, &b_ptr, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND,
                                                &input_b);
    llm_kernels::ascend::Add(input_a, input_b, add_alpha, &add_output, context_->GetComputeStreams()[rank_].Get(),
                             GetWorkSpaceFunc());

    ACL_CHECK(aclDestroyTensor(input_a));
    ACL_CHECK(aclDestroyTensor(input_b));
    ACL_CHECK(aclDestroyTensor(add_output));
  } else {
    return Status(RET_SEGMENT_FAULT, "add bias not implemented");
  }
  output_tensors[0].shape = input_tensors[0].shape;
  output_tensors[0].dtype = input_tensors[0].dtype;
  return Status();
}
}  // namespace ksana_llm
