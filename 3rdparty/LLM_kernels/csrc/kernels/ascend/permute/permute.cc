/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include "csrc/kernels/ascend/permute/permute.h"
#include <type_traits>

#include "aclrtlaunch_InvokePermuteKernel.h"

#include "csrc/utils/ascend/common.h"
#include "csrc/utils/ascend/tiling_data_types.h"

namespace llm_kernels {
namespace ascend {

// The max used ai core number.
constexpr uint32_t MAX_USED_CORE_NUM = 24;

// The min block size of permute.
constexpr uint32_t MIN_PERMUTE_BLOCK_SIZE = 16 * 16;

void Permute(const aclTensor* permute_input, void** permute_input_tensor_addr_ptr, aclTensor** permute_output,
             const std::vector<int64_t>& dims, aclrtStream& stream, void (*ws_func)(size_t, void**)) {
  int64_t* input_t_shape_ptr = nullptr;
  uint64_t input_t_dims_num = 0;
  ACL_CHECK_RET(aclGetViewShape(permute_input, &input_t_shape_ptr, &input_t_dims_num));
  std::vector<int64_t> input_t_shape(input_t_dims_num);
  for (uint64_t i = 0; i < input_t_dims_num; ++i) {
    input_t_shape[i] = input_t_shape_ptr[i];
  }
  std::vector<int64_t> input_t_strides;
  utils::CalShapeStrides(input_t_shape, input_t_strides);

  std::vector<int64_t> output_t_shape(input_t_dims_num, 0);
  std::vector<int64_t> output_t_strides(input_t_shape.size(), 1);
  std::copy(input_t_shape.begin(), input_t_shape.end(), output_t_shape.begin());
  for (uint64_t i = 0; i < dims.size(); ++i) {
    output_t_shape[i] = input_t_shape[dims[i]];
    output_t_strides[i] = input_t_strides[dims[i]];
  }
  aclDataType acl_dtype;
  ACL_CHECK_RET(aclGetDataType(permute_input, &acl_dtype));
  *permute_output = aclCreateTensor(output_t_shape.data(), output_t_shape.size(), acl_dtype, output_t_strides.data(), 0,
                                    aclFormat::ACL_FORMAT_ND, output_t_shape.data(), output_t_shape.size(),
                                    *permute_input_tensor_addr_ptr);

  ACL_CHECK_RET(aclrtSynchronizeStream(stream));
}

template <typename T>
PermuteKernelWrapper<T>::PermuteKernelWrapper() {
  tiling_size_ = sizeof(PermuteTilingData);
  ACL_CHECK_RET(aclrtMalloc(&tiling_buffer_gm_, tiling_size_, ACL_MEM_MALLOC_HUGE_FIRST));

  // TODO: Get block num from device info.
  tiling_data_.used_core_num = MAX_USED_CORE_NUM;

  size_t usr_workspace_size = 1024 << 2;
  size_t sys_workspace_size = 1024 << 14;
  ACL_CHECK_RET(aclrtMalloc(&workspace_gm_, usr_workspace_size + sys_workspace_size, ACL_MEM_MALLOC_HUGE_FIRST));
}

template <typename T>
PermuteKernelWrapper<T>::~PermuteKernelWrapper() {
  ACL_CHECK_RET(aclrtFree(tiling_buffer_gm_));
  ACL_CHECK_RET(aclrtFree(workspace_gm_));
}

template <typename T>
void PermuteKernelWrapper<T>::CopyTilingToDevice(aclrtStream stream) {
  ACL_CHECK_RET(aclrtMemcpyAsync(tiling_buffer_gm_, tiling_size_, &tiling_data_, tiling_size_,
                                 aclrtMemcpyKind::ACL_MEMCPY_HOST_TO_DEVICE, stream));
  ACL_CHECK_RET(aclrtSynchronizeStream(stream));
}

template <typename T>
void PermuteKernelWrapper<T>::Forward(void* output, void* input, const std::vector<uint64_t>& shape,
                          const std::vector<uint64_t> new_indexes, aclrtStream stream) {
  // The kernel support at most 6 dimension now.
  if (shape.size() > 6 && new_indexes.size() > 6) {
    return;
  }

  std::vector<uint64_t> input_shape = shape;
  while (input_shape.size() < 6) {
    input_shape.insert(input_shape.begin(), 1);
  }

  std::vector<uint32_t> strides;
  strides.resize(input_shape.size(), 1);
  for (int64_t i = input_shape.size() - 2; i >= 0; i--) {
    strides[i] = input_shape[i + 1] * strides[i + 1];
  }

  // NOTE(karlluo): for Huawei kernel not support dynamic vector.
  // extern indexes to 6 dims to fill static const length vector.
  std::vector<uint64_t> output_new_indexes = new_indexes;
  int fill_dim = 6 - output_new_indexes.size();
  for (size_t i = 0; i < output_new_indexes.size(); ++i) {
    output_new_indexes[i] = output_new_indexes[i] + fill_dim;
  }

  fill_dim = fill_dim - 1;
  while (fill_dim >= 0) {
    output_new_indexes.insert(output_new_indexes.begin(), fill_dim);
    fill_dim = fill_dim - 1;
  }

  std::vector<int64_t> new_shape;
  for (auto i : output_new_indexes) {
    new_shape.push_back(input_shape[i]);
  }

  std::vector<uint32_t> new_strides;
  new_strides.resize(new_shape.size(), 1);
  for (int64_t i = new_shape.size() - 2; i >= 0; i--) {
    new_strides[i] = new_shape[i + 1] * new_strides[i + 1];
  }

  int64_t shape_size = 1;
  for (auto i : input_shape) {
    shape_size *= i;
  }

  tiling_data_.dim0 = input_shape[0];
  tiling_data_.dim1 = input_shape[1];
  tiling_data_.dim2 = input_shape[2];
  tiling_data_.dim3 = input_shape[3];
  tiling_data_.dim4 = input_shape[4];
  tiling_data_.dim5 = input_shape[5];

  tiling_data_.stride0 = strides[0];
  tiling_data_.stride1 = strides[1];
  tiling_data_.stride2 = strides[2];
  tiling_data_.stride3 = strides[3];
  tiling_data_.stride4 = strides[4];
  tiling_data_.stride5 = strides[5];

  tiling_data_.new_idx0 = output_new_indexes[0];
  tiling_data_.new_idx1 = output_new_indexes[1];
  tiling_data_.new_idx2 = output_new_indexes[2];
  tiling_data_.new_idx3 = output_new_indexes[3];
  tiling_data_.new_idx4 = output_new_indexes[4];
  tiling_data_.new_idx5 = output_new_indexes[5];

  tiling_data_.new_stride0 = new_strides[0];
  tiling_data_.new_stride1 = new_strides[1];
  tiling_data_.new_stride2 = new_strides[2];
  tiling_data_.new_stride3 = new_strides[3];
  tiling_data_.new_stride4 = new_strides[4];
  tiling_data_.new_stride5 = new_strides[5];

  tiling_data_.total_length = shape_size;

  if (shape_size / MIN_PERMUTE_BLOCK_SIZE >= MAX_USED_CORE_NUM) {
    tiling_data_.used_core_num = MAX_USED_CORE_NUM;
  } else {
    tiling_data_.used_core_num = (shape_size + MIN_PERMUTE_BLOCK_SIZE - 1) / MIN_PERMUTE_BLOCK_SIZE;
  }

  tiling_data_.used_core_num = 1;
  tiling_data_.block_length = (shape_size + tiling_data_.used_core_num - 1) / tiling_data_.used_core_num;

  if (sizeof(T) == 2) {
    tiling_data_.tiling_key = static_cast<uint32_t>(TilingDataType::FLOAT16);
  } else if (sizeof(T) == 4) {
    tiling_data_.tiling_key = static_cast<uint32_t>(TilingDataType::FLOAT32);
  }

  CopyTilingToDevice(stream);

  ACLRT_LAUNCH_KERNEL(InvokePermuteKernel)
  (tiling_data_.used_core_num, stream, input, output, workspace_gm_, tiling_buffer_gm_);
}

template class PermuteKernelWrapper<aclFloat16>;
template class PermuteKernelWrapper<float>;

}  // namespace ascend
}  // namespace llm_kernels
   //
