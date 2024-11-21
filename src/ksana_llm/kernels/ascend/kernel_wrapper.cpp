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
    case DataType::TYPE_BF16:
      return aclDataType::ACL_BF16;
    case DataType::TYPE_FP32:
      return aclDataType::ACL_FLOAT;
    default:
      return aclDataType::ACL_FLOAT;
  }
}

void LookupEmbedding(const aclTensor* input_ids, const aclTensor* embedding_table, const aclTensor* position_table,
                     aclTensor* output, aclrtStream stream, WorkSpaceFunc ws_func) {
  uint64_t ws_size = 0ull;
  void* workspace = nullptr;
  aclOpExecutor* executor = nullptr;
  ACL_CHECK_RET(aclnnEmbeddingGetWorkspaceSize(embedding_table, input_ids, output, &ws_size, &executor));
  ws_func(ws_size, &workspace);
  ACL_CHECK_RET(aclnnEmbedding(workspace, ws_size, executor, stream));
}

Status CastInplace(Tensor& tensor, const DataType target_dtype, Stream& stream, void* workspace_ptr) {
  if (tensor.dtype != target_dtype) {
    llm_kernels::utils::ATBOperationExecutor atb_cast_op_executor;
    llm_kernels::ascend::CastParam cast_param;
    // NOTE(karlluo): there is no inplace in ATB, we have prepare a buffer simulate inplace operation
    cast_param.dataType = static_cast<aclDataType>(target_dtype);
    atb::Operation* cast_op = new llm_kernels::ascend::CastOperation(
        fmt::format("Cast{}To{}Inplace", GetTypeString(tensor.dtype), GetTypeString(target_dtype)), cast_param);
    atb_cast_op_executor.SetOperation(cast_op);
    int32_t rank = GetBlockManager()->GetDeviceId();
    reinterpret_cast<atb::Context*>(GetRuntimeContext(rank))->SetExecuteStream(stream.Get());
    int block_id = -1;
    GetBlockManager()->AllocateContiguous(tensor.GetTotalBytes(), block_id);
    Tensor tmp_tensor(MemoryDevice::MEMORY_DEVICE, target_dtype, tensor.shape, block_id);
    atb_cast_op_executor.ResetVariantPack();
    atb_cast_op_executor.SetInputTensor(tensor.GetPtr<void>(), tensor.shape, static_cast<aclDataType>(tensor.dtype));
    atb_cast_op_executor.SetOutputTensor(tmp_tensor.GetPtr<void>(), tensor.shape,
                                         static_cast<aclDataType>(target_dtype));
    atb_cast_op_executor.Run(reinterpret_cast<atb::Context*>(GetRuntimeContext(rank)), GetWorkSpaceFunc());
    StreamSynchronize(stream);
    Memcpy(tensor.GetPtr<void>(), tmp_tensor.GetPtr<void>(), tensor.GetTotalBytes(), MEMCPY_DEVICE_TO_DEVICE);
    GetBlockManager()->FreeContiguous(block_id);
    tensor.dtype = target_dtype;
  } else {
    // NOTE(karlluo): dtype same will skip cast
    KLLM_LOG_DEBUG << fmt::format("Cast{}To{}Inplace is ignore", GetTypeString(tensor.dtype),
                                  GetTypeString(target_dtype));
  }
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
#ifdef ENABLE_BFLOAT16
INSTANTIATE_ARG_MAX(bfloat16);
#endif

#undef INSTANTIATE_ARG_MAX

Status TransLayout(Tensor& tensor, Stream& stream) {
  constexpr size_t TRANSDATA_IN_DIMS = 3ul;
  constexpr size_t TRANSDATA_TRUNK = 16ul;
  std::vector<size_t> src_shape = tensor.shape;
  std::vector<size_t> in_tensor_shape = src_shape;
  // NOTE(karlluo): transdata need 3 dims input
  if (src_shape.size() < TRANSDATA_IN_DIMS) {
    in_tensor_shape.clear();
    in_tensor_shape.resize(TRANSDATA_IN_DIMS, 1);
    for (size_t dim_idx = TRANSDATA_IN_DIMS - src_shape.size(), src_dim_idx = 0;
         dim_idx < TRANSDATA_IN_DIMS && src_dim_idx < src_shape.size(); ++dim_idx, ++src_dim_idx) {
      in_tensor_shape[dim_idx] = src_shape[src_dim_idx];
    }
  } else if (src_shape.size() > TRANSDATA_IN_DIMS) {
    in_tensor_shape.clear();
    in_tensor_shape.resize(TRANSDATA_IN_DIMS, 1);
    in_tensor_shape[0] = src_shape[0];
    in_tensor_shape[1] = src_shape[1];
    size_t final_dim = 1ul;
    for (size_t dim_idx = 2ul; dim_idx < src_shape.size(); ++dim_idx) {
      final_dim *= src_shape[dim_idx];
    }
    in_tensor_shape[2] = final_dim;
  }

  if (in_tensor_shape.at(2) % TRANSDATA_TRUNK != 0) {
    return Status();
  }

  std::vector<size_t> out_tensor_shape = {in_tensor_shape.at(0), in_tensor_shape.at(2) / TRANSDATA_TRUNK,
                                          in_tensor_shape.at(1), TRANSDATA_TRUNK};
  const aclFormat src_format = GetACLFormat(tensor.data_format);
  aclFormat target_format = aclFormat::ACL_FORMAT_FRACTAL_NZ;
  atb::infer::TransdataParam::TransdataType transdata_type =
      atb::infer::TransdataParam::TransdataType::ND_TO_FRACTAL_NZ;
  atb::infer::TransdataParam transdata_param;
  if (src_format == aclFormat::ACL_FORMAT_FRACTAL_NZ) {
    // for nz to nd
    // NOTE(karlluo): nz to nd only support shape size > 2
    // ref:
    // https://www.hiascend.com/document/detail/zh/canncommercial/80RC22/apiref/ascendtbapi/ascendtb_01_0083.html
    target_format = aclFormat::ACL_FORMAT_ND;
    transdata_type = atb::infer::TransdataParam::TransdataType::FRACTAL_NZ_TO_ND;
    transdata_param.outCrops[0] = tensor.shape[in_tensor_shape.size() - 2];
    transdata_param.outCrops[1] = tensor.shape[in_tensor_shape.size() - 1] * tensor.shape[in_tensor_shape.size() - 3];
  }
  void* workspace_ptr;
  int workspace_block_id;
  GetBlockManager()->AllocateContiguous(tensor.GetTotalBytes(), workspace_block_id);
  GetBlockManager()->GetContiguousPtr(workspace_block_id, workspace_ptr);
  transdata_param.transdataType = transdata_type;
  llm_kernels::utils::ATBOperationExecutor atb_op_executor;
  int32_t rank = GetBlockManager()->GetDeviceId();
  atb_op_executor.Init(rank, transdata_param);
  atb_op_executor.SetInputTensor(tensor.GetPtr<void>(), in_tensor_shape, static_cast<aclDataType>(tensor.dtype),
                                 src_format);
  atb_op_executor.SetOutputTensor(workspace_ptr, out_tensor_shape, static_cast<aclDataType>(tensor.dtype),
                                  target_format);
  atb_op_executor.Run(reinterpret_cast<atb::Context*>(GetRuntimeContext(rank)), GetWorkSpaceFunc());
  tensor.data_format = GetTensorFormat(target_format);
  tensor.shape = out_tensor_shape;
  MemcpyAsync(tensor.GetPtr<void>(), workspace_ptr, tensor.GetTotalBytes(), MEMCPY_DEVICE_TO_DEVICE, stream);
  StreamSynchronize(stream);
  GetBlockManager()->FreeContiguous(workspace_block_id);
  return Status();
}

}  // namespace ksana_llm
