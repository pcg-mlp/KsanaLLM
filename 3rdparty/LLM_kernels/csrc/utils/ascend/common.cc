/**
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include "common.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>

#include "aclnnop/aclnn_cast.h"
#include "aclnnop/aclnn_copy.h"
#include "aclnnop/aclnn_random.h"

namespace llm_kernels {
namespace utils {

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shape_size = 1;
  for (auto i : shape) {
    shape_size *= i;
  }
  return shape_size;
}

void CalShapeStrides(const std::vector<int64_t>& shape, std::vector<int64_t>& strides) {
  strides.resize(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
}

void ReallocWorkspace(const uint64_t new_size, uint64_t& old_size, void** ws_dev, aclrtMemMallocPolicy policy) {
  if (new_size > old_size) {
    if (old_size > 0) {
      ACL_CHECK_RET(aclrtFree(*ws_dev));
      *ws_dev = nullptr;
    }
    old_size = new_size;
    ACL_CHECK_RET(aclrtMalloc(ws_dev, new_size, policy));
  }
}

void CreateAclTensor(const int64_t hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
                     aclFormat dataFormat, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * DT2LONG.at(dataType);
  ACL_CHECK_RET(aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK_RET(aclrtMemcpy(*deviceAddr, size, &hostData, size, ACL_MEMCPY_HOST_TO_DEVICE));
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, dataFormat, shape.data(),
                            shape.size(), *deviceAddr);
}

void CreateAclTensor(const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType, aclFormat dataFormat,
                     aclTensor** tensor) {
  auto size = GetShapeSize(shape) * DT2LONG.at(dataType);
  ACL_CHECK_RET(aclrtMalloc(deviceAddr, size, ACL_MEM_TYPE_HIGH_BAND_WIDTH));
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, dataFormat, shape.data(),
                            shape.size(), *deviceAddr);
}

// TODO: void** deviceAddr,aclTensor** tensor,aclDataType dataType, aclFormat dataFormat,
void CreateAclTensorWithData(const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
                             aclFormat dataFormat, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * DT2LONG.at(dataType);
  std::vector<int64_t> strides;
  CalShapeStrides(shape, strides);
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, dataFormat, shape.data(),
                            shape.size(), *deviceAddr);
}

std::string GetNumpyTypeDesc(aclDataType dtype) {
  switch (dtype) {
    case aclDataType::ACL_BF16:
      std::cerr << "GetNumpyTypeDesc(Bfloat16) returns an invalid type 'x' since Numpy doesn't "
                   "support bfloat16 as of now, it will be properly extended if numpy supports. "
                   "Please refer for the discussions https://github.com/numpy/numpy/issues/19808."
                << std::endl;
      return "x";
    case aclDataType::ACL_BOOL:
      return "?";
    case aclDataType::ACL_UINT8:
      return "u1";
    case aclDataType::ACL_UINT16:
      return "u2";
    case aclDataType::ACL_UINT32:
      return "u4";
    case aclDataType::ACL_UINT64:
      return "u8";
    case aclDataType::ACL_INT8:
      return "i1";
    case aclDataType::ACL_INT16:
      return "i2";
    case aclDataType::ACL_INT32:
      return "i4";
    case aclDataType::ACL_INT64:
      return "i8";
    case aclDataType::ACL_FLOAT16:
      return "f2";
    case aclDataType::ACL_FLOAT:
      return "f4";
    case aclDataType::ACL_DOUBLE:
      return "f8";
    default:
      return "x";
  }
}

void SaveNpy(const aclTensor* tensor, const std::string& filename, aclrtStream& stream) {
  aclDataType acl_dtype;
  ACL_CHECK_RET(aclGetDataType(tensor, &acl_dtype));
  int64_t* shape = nullptr;
  uint64_t dims = 0;
  ACL_CHECK_RET(aclGetViewShape(tensor, &shape, &dims));
  size_t elem_nums = dims == 0ul ? 0ul : 1ul;
  std::vector<int64_t> tensor_shape(dims);
  for (uint64_t i = 0; i < dims; ++i) {
    elem_nums *= shape[i];
    tensor_shape[i] = shape[i];
  }
  aclTensor* tmp_tensor;
  void* tmp_tensor_ptr = nullptr;
  CreateAclTensor(tensor_shape, &tmp_tensor_ptr, acl_dtype, aclFormat::ACL_FORMAT_ND, &tmp_tensor);
  uint64_t new_workspace_size = 0ul;
  aclOpExecutor* executor = nullptr;
  ACL_CHECK_RET(aclnnInplaceCopyGetWorkspaceSize(tmp_tensor, tensor, &new_workspace_size, &executor));
  void* workspace_ptr = nullptr;
  if (new_workspace_size > 0) {
    ACL_CHECK_RET(aclrtMalloc(&workspace_ptr, new_workspace_size, ACL_MEM_TYPE_HIGH_BAND_WIDTH));
  }
  ACL_CHECK_RET(aclnnInplaceCopy(workspace_ptr, new_workspace_size, executor, stream));
  ACL_CHECK_RET(aclrtSynchronizeStream(stream));

  // copy
  void* host_buffer_ptr = nullptr;
  ACL_CHECK_RET(aclrtMallocHost(&host_buffer_ptr, elem_nums * aclDataTypeSize(acl_dtype)));
  ACL_CHECK_RET(aclrtMemcpy(tmp_tensor_ptr, elem_nums * aclDataTypeSize(acl_dtype), host_buffer_ptr,
                            elem_nums * aclDataTypeSize(acl_dtype), ACL_MEMCPY_DEVICE_TO_HOST));

  const char magic[] =
      "\x93"
      "NUMPY";
  const uint8_t npy_major = 1;
  const uint8_t npy_minor = 0;

  std::stringstream header_stream;
  header_stream << "{'descr': '" << GetNumpyTypeDesc(acl_dtype) << "', 'fortran_order': False, 'shape': (";
  for (size_t i = 0; i < tensor_shape.size(); ++i) {
    header_stream << tensor_shape[i];
    if (i + 1 < tensor_shape.size() || tensor_shape.size() == 1) {
      header_stream << ", ";
    }
  }
  header_stream << ")}";
  int32_t base_length = 6 + 4 + header_stream.str().size();
  int32_t pad_length = 16 * ((base_length + 1 + 15) / 16);  // Take ceiling of base_length + 1 (for '\n' ending)
  for (int32_t i = 0; i < pad_length - base_length; ++i) {
    header_stream << ((i == pad_length - base_length - 1) ? "\n" : "\x20");
  }
  std::string header = header_stream.str();
  const uint16_t header_len = header.size();

  FILE* f_ptr = fopen(filename.c_str(), "wb");
  if (f_ptr == nullptr) {
    std::cerr << "Unable to open " << filename << " for writing." << std::endl;
  }
  fwrite(magic, sizeof(char), sizeof(magic) - 1, f_ptr);
  fwrite(&npy_major, sizeof(uint8_t), 1, f_ptr);
  fwrite(&npy_minor, sizeof(uint8_t), 1, f_ptr);
  fwrite(&header_len, sizeof(uint16_t), 1, f_ptr);
  fwrite(header.c_str(), sizeof(char), header_len, f_ptr);
  fwrite(host_buffer_ptr, aclDataTypeSize(acl_dtype), elem_nums, f_ptr);
  fclose(f_ptr);
  ACL_CHECK_RET(aclrtFreeHost(host_buffer_ptr));
  ACL_CHECK_RET(aclDestroyTensor(tmp_tensor));
  ACL_CHECK_RET(aclrtFree(tmp_tensor_ptr));
  if (workspace_ptr != nullptr) {
    ACL_CHECK_RET(aclrtFree(workspace_ptr));
  }
}

void PrintTensor(const aclTensor* src, aclrtStream& stream, const char* name) {
  void* ws_dev = nullptr;
  aclDataType src_dtype;
  aclDataType dst_dtype = aclDataType::ACL_FLOAT;
  ACL_CHECK_RET(aclGetDataType(src, &src_dtype));
  int64_t* src_shape = nullptr;
  uint64_t dim = 0;
  ACL_CHECK_RET(aclGetViewShape(src, &src_shape, &dim));
  std::vector<int64_t> shape(dim);
  LOG_PRINT("%s tensor shape: [", name);
  for (uint64_t i = 0; i < dim; ++i) {
    shape[i] = src_shape[i];
    std::cout << shape[i] << ",";
  }
  std::cout << "] dtype: " << src_dtype << std::endl;

  aclTensor* dst = nullptr;
  void* dstDev = nullptr;
  CreateAclTensor(shape, &dstDev, dst_dtype, aclFormat::ACL_FORMAT_ND, &dst);

  uint64_t ws_size = 0;
  aclOpExecutor* executor = nullptr;
  ACL_CHECK_RET(aclnnCastGetWorkspaceSize(src, dst_dtype, dst, &ws_size, &executor));
  if (ws_size > 0) {
    ACL_CHECK_RET(aclrtMalloc(&ws_dev, ws_size, ACL_MEM_MALLOC_NORMAL_ONLY));
  }
  ACL_CHECK_RET(aclnnCast(ws_dev, ws_size, executor, stream));
  ACL_CHECK_RET(aclrtSynchronizeStream(stream));

  auto elemsize = GetShapeSize(shape);
  std::vector<float> fp32(elemsize);
  ACL_CHECK_RET(
      aclrtMemcpy(fp32.data(), elemsize * sizeof(float), dstDev, elemsize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
  aclDestroyTensor(dst);
  aclrtFree(dstDev);
  dstDev = nullptr;
  if (ws_dev) {
    aclrtFree(ws_dev);
    ws_dev = nullptr;
  }
  size_t n = 10;
  LOG_PRINT("%s last seq, first %d:", name, n);
  for (size_t i = 0; i < std::min(n, fp32.size()); ++i) {
    std::cout << fp32[fp32.size() - 1 - n + i] << ",";
  }
  std::cout << std::endl;
}

void Random(aclTensor* input_tensor, void* workspace_ptr, uint64_t workspace_size, int64_t from, int64_t to,
            int64_t seed, int64_t offset, aclrtStream& stream) {
  aclOpExecutor* executor = nullptr;
  uint64_t new_workspace_size = 0ul;
  ACL_CHECK_RET(
      aclnnInplaceRandomGetWorkspaceSize(input_tensor, from, to, seed, offset, &new_workspace_size, &executor));
  ACL_CHECK_RET(workspace_size >= new_workspace_size);
  ACL_CHECK_RET(aclnnInplaceRandom(workspace_ptr, new_workspace_size, executor, stream));
  ACL_CHECK_RET(aclrtSynchronizeStream(stream));
}

void GetTestWorkSpaceFunc(size_t size, void** ws_addr) {
  static void* cached_addr = nullptr;
  static size_t cached_size = 0;

  if (size > 0) {
    if (cached_size < size) {
      if (cached_addr != nullptr) {
        aclrtFree(cached_addr);
      }
      ACL_CHECK_RET(aclrtMalloc(&cached_addr, size, ACL_MEM_MALLOC_HUGE_FIRST));
      cached_size = size;
    }

    *ws_addr = cached_addr;
  }
}

}  // namespace utils
}  // namespace llm_kernels
