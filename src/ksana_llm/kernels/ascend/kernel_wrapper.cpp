/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/kernels/ascend/kernel_wrapper.h"

#include "atb/infer_op_params.h"

#include "3rdparty/LLM_kernels/csrc/utils/ascend/common.h"
#include "csrc/kernels/ascend/argmax/argmax.h"
#include "csrc/kernels/ascend/embedding/embedding.h"
#include "csrc/kernels/ascend/permute/permute.h"
#include "csrc/utils/ascend/common.h"

#include "ksana_llm/kernels/argmax.h"
#include "ksana_llm/kernels/cast.h"
#include "ksana_llm/utils/ascend/acl_utils.h"

namespace ksana_llm {

aclDataType CastDataTypeToAclDataType(const DataType dtype) {
  switch (dtype) {
    case DataType::TYPE_FP16:
      return aclDataType::ACL_FLOAT16;
    case DataType::TYPE_FP32:
      return aclDataType::ACL_FLOAT;
    default:
      return aclDataType::ACL_FLOAT;
  }
}

void LookupEmbedding(const aclTensor* input_ids, const aclTensor* embedding_table, const aclTensor* position_table,
                     aclTensor* output, aclrtStream stream, WorkSpaceFunc ws_func) {
  llm_kernels::ascend::LookupEmbedding(input_ids, embedding_table, position_table, output, stream, ws_func);
}

Status CastInplace(Tensor& tensor, const DataType target_dtype, Stream& stream, void* workspace_ptr) {
  if (tensor.dtype == DataType::TYPE_BF16 || target_dtype == DataType::TYPE_BF16) {
    KLLM_THROW(
        fmt::format("Invalid cast compute type bfloat16, only support float16 to float32 or float32 to float16. "
                    "Tensor type is {}, target type is {}.",
                    tensor.dtype, target_dtype));
  } else if (tensor.dtype == target_dtype) {
    // No need to convert
  } else {
    KLLM_THROW(fmt::format("CastInplace from type {} to {} is not yet implement", tensor.dtype, target_dtype));
  }
  tensor.dtype = target_dtype;
  return Status();
}

Status Permute(Tensor& input_tensor, Tensor& output_tensor, const std::vector<size_t>& permutation, Stream& stream,
               void* workspace_ptr) {
  std::vector<size_t> input_shape = input_tensor.shape;
  std::vector<size_t> output_shape = input_tensor.shape;
  atb::infer::TransposeParam param;
  for (size_t i = 0; i < permutation.size(); ++i) {
    param.perm.push_back(static_cast<int32_t>(permutation[i]));
    output_shape[i] = input_shape[permutation[i]];
  }
  llm_kernels::utils::ATBOperationExecutor atb_op_executor;
  int32_t rank = GetBlockManager()->GetDeviceId();
  atb_op_executor.Init(rank, param);
  output_tensor.dtype = input_tensor.dtype;
  output_tensor.shape = output_shape;
  reinterpret_cast<atb::Context*>(GetRuntimeContext(rank))->SetExecuteStream(stream.Get());
  atb_op_executor.ResetVariantPack();
  atb_op_executor.SetInputTensor(input_tensor.GetPtr<void>(), input_tensor.shape,
                                 static_cast<aclDataType>(input_tensor.dtype));
  atb_op_executor.SetOutputTensor(output_tensor.GetPtr<void>(), output_tensor.shape,
                                  static_cast<aclDataType>(output_tensor.dtype));
  atb_op_executor.Run(reinterpret_cast<atb::Context*>(GetRuntimeContext(rank)), GetWorkSpaceFunc());
  StreamSynchronize(stream);
  return Status();
}

template <typename T>
Status ArgMaxATBExecutor<T>::Init(const int rank, const size_t max_batch_size) {
  llm_kernels::ascend::ArgmaxParam argmax_param;
  argmax_param.dim = 1;
  atb::Operation* argmax_op = new llm_kernels::ascend::ArgmaxOperation("argmax_exec_op_1", argmax_param);

  llm_kernels::ascend::CastParam cast_param;
  cast_param.dataType = aclDataType::ACL_UINT32;
  atb::Operation* cast_op = new llm_kernels::ascend::CastOperation("argmax_exec_op_2", cast_param);

  atb_argmax_op_executor_.SetOperation(argmax_op);
  atb_cast_op_executor_.SetOperation(cast_op);
  STATUS_CHECK_FAILURE(CreateTensor(internal_tensor_, {max_batch_size}, DataType::TYPE_INT32, rank, MEMORY_DEVICE));
  return Status();
}

template <typename T>
Status ArgMaxATBExecutor<T>::Run(const int rank, const T* input, const uint32_t* ids_offset, const int32_t batch_size,
                                 const int32_t vocab_size, uint32_t* result, Stream& stream) {
  // NOTE(karlluo): get argmax and output int32 type
  atb_argmax_op_executor_.ResetVariantPack();
  atb_argmax_op_executor_.SetInputTensor(reinterpret_cast<void*>(const_cast<T*>(input)),
                                         {static_cast<uint32_t>(batch_size), static_cast<uint32_t>(vocab_size)},
                                         static_cast<aclDataType>(TYPE_FP32));
  atb_argmax_op_executor_.SetOutputTensor(internal_tensor_.GetPtr<void>(), {static_cast<uint32_t>(batch_size)},
                                          static_cast<aclDataType>(TYPE_INT32));
  atb_argmax_op_executor_.Run(reinterpret_cast<atb::Context*>(GetRuntimeContext(rank)), GetWorkSpaceFunc());
  // NOTE(karlluo): cast int32 to uint32 type
  atb_cast_op_executor_.ResetVariantPack();
  atb_cast_op_executor_.SetInputTensor(internal_tensor_.GetPtr<void>(), {static_cast<uint32_t>(batch_size)},
                                       static_cast<aclDataType>(TYPE_INT32));
  atb_cast_op_executor_.SetOutputTensor(result, {static_cast<uint32_t>(batch_size)},
                                        static_cast<aclDataType>(TYPE_UINT32));
  atb_cast_op_executor_.Run(reinterpret_cast<atb::Context*>(GetRuntimeContext(rank)), GetWorkSpaceFunc());
  return Status();
}

template class ArgMaxATBExecutor<float>;

template <typename T>
Status ArgMax(const T* input, const uint32_t* ids_offset, const int32_t batch_size, const int32_t vocab_size,
              uint32_t* result, Stream& stream, void* buffer_ptr) {
  if (ids_offset != nullptr) {
    KLLM_THROW("Not supported ids offset.");
  }
  if (std::is_same<T, float>::value) {
    int32_t rank = GetBlockManager()->GetDeviceId();
    reinterpret_cast<atb::Context*>(GetRuntimeContext(rank))->SetExecuteStream(stream.Get());
    ArgMaxATBExecutor<T>* arg_max_atb_executor_ptr = reinterpret_cast<ArgMaxATBExecutor<T>*>(buffer_ptr);
    arg_max_atb_executor_ptr->Run(rank, input, ids_offset, batch_size, vocab_size, result, stream);
  } else {
    KLLM_THROW("Not supported argmax data type.");
  }
  return Status();
}

#define INSTANTIATE_ARG_MAX(T)                                                                 \
  template Status ArgMax(const T* input, const uint32_t* ids_offset, const int32_t batch_size, \
                         const int32_t vocab_size, uint32_t* result, Stream& stream, void* buffer_ptr);

INSTANTIATE_ARG_MAX(float);
INSTANTIATE_ARG_MAX(float16);

#undef INSTANTIATE_ARG_MAX

}  // namespace ksana_llm
