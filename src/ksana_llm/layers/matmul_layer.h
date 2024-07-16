/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/layers/base_layer.h"
#ifdef ENABLE_ACL
#  include "3rdparty/LLM_kernels/csrc/utils/ascend/common.h"
#  ifdef ENABLE_ACL_ATB
#    include "3rdparty/LLM_kernels/csrc/utils/ascend/atb_executor.h"
#  endif  // ENABLE_ACL_ATB
#endif

namespace ksana_llm {

template <typename T>
class MatMulLayer : public BaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context, int rank) override;

  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) override;

 private:
#ifdef ENABLE_ACL
  // ACLNNMatmulComputeType(int8_t, Calculation Input): Integer type on the Host side, determines which calculation
  // logic the Cube unit uses for operations. The data type supports INT8, and the supported enumeration values are as
  // follows: 0: KEEP_DTYPE - Keep the input data type for calculation. When the input is FLOAT, the Cube calculation
  // unit of the Atlas training series products does not support it. An error will occur if 0 is selected. 1:
  // ALLOW_FP32_DOWN_PRECISION - Allow the input data to be downcast for calculation. When the input is FLOAT, the Atlas
  // training series products convert it to FLOAT16 for calculation, and the Atlas A2 training series products convert
  // it to HFLOAT32 for calculation. 2: USE_FP16 - Allow conversion to the data type FLOAT16 for calculation. When the
  // input data type is FLOAT, it is converted to FLOAT16 for calculation. 3: USE_HF32 - Allow conversion to the data
  // type HFLOAT32 for calculation. When the input is FLOAT, the Cube calculation unit of the Atlas training series
  // products does not support it. An error will occur if 3 is selected. The Atlas A2 training series products convert
  // it to HFLOAT32 for calculation.
  llm_kernels::utils::ACLNNMatmulComputeType aclnn_mm_type_{llm_kernels::utils::ACLNNMatmulComputeType::KEEP_DTYPE};
  aclDataType aclnn_dtype_;

#  ifdef ENABLE_ACL_ATB
  llm_kernels::utils::ATBOperationExecutor atb_op_executor_;
#  endif  // ENABLE_ACL_ATB
#endif    // ENABLE_ACL
};

}  // namespace ksana_llm
