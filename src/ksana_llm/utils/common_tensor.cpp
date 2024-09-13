/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/utils/common_tensor.h"

#include <numeric>
#include <string>

#include "ksana_llm/utils/device_utils.h"
#ifdef ENABLE_ACL
#  include "3rdparty/LLM_kernels/csrc/utils/ascend/common.h"
#endif

namespace ksana_llm {

void ParseNpyIntro(FILE*& f_ptr, uint32_t& header_len, uint32_t& start_data) {
  const char magic[] =
      "\x93"
      "NUMPY";
  char magic_test[sizeof(magic)] = "\0";

  size_t n_elems = fread(reinterpret_cast<void*>(magic_test), sizeof(char), sizeof(magic) - 1, f_ptr);
  if (n_elems != sizeof(magic) - 1 || std::string(magic) != std::string(magic_test)) {
    KLLM_THROW("Could read magic token in NPY file");
  }

  uint8_t npy_major = 0;
  uint8_t npy_minor = 0;
  n_elems = fread(reinterpret_cast<void*>(&npy_major), sizeof(uint8_t), 1, f_ptr);
  n_elems += fread(reinterpret_cast<void*>(&npy_minor), sizeof(uint8_t), 1, f_ptr);

  if (npy_major == 1) {
    uint16_t header_len_u16 = 0;
    n_elems = fread(reinterpret_cast<void*>(&header_len_u16), sizeof(uint16_t), 1, f_ptr);
    header_len = header_len_u16;
  } else if (npy_major == 2) {
    uint32_t header_len_u32 = 0;
    n_elems = fread(reinterpret_cast<void*>(&header_len_u32), sizeof(uint32_t), 1, f_ptr);
    header_len = header_len_u32;
  } else {
    KLLM_THROW(fmt::format("Unsupported npy version: {}", std::to_string(npy_major)));
  }

  start_data = 8 + 2 * npy_major + header_len;
}

void ParseNpyHeader(FILE*& f_ptr, uint32_t header_len, std::vector<size_t>& shape) {
  char* header_c = reinterpret_cast<char*>(malloc(header_len * sizeof(char)));
  size_t n_elems = fread(reinterpret_cast<void*>(header_c), sizeof(char), header_len, f_ptr);
  if (n_elems != header_len) {
    free(header_c);
    KLLM_THROW(
        fmt::format("Tensor total elements number {} is not equal with NPY's elements number {}", n_elems, header_len));
  }
  std::string header(header_c, header_len);
  free(header_c);

  size_t start, end;
  start = header.find("'descr'") + 7;
  start = header.find("'", start);
  end = header.find("'", start + 1);

  start = header.find("'fortran_order'") + 15;
  start = header.find(":", start);
  end = header.find(",", start + 1);
  if (header.substr(start + 1, end - start - 1).find("False") == std::string::npos) {
    KLLM_THROW("Unsupported value for fortran_order while reading npy file");
  }

  start = header.find("'shape'") + 7;
  start = header.find("(", start);
  end = header.find(")", start + 1);

  std::istringstream shape_stream(header.substr(start + 1, end - start - 1));
  std::string token;

  shape.clear();
  while (std::getline(shape_stream, token, ',')) {
    if (token.find_first_not_of(' ') == std::string::npos) {
      break;
    }
    shape.push_back(std::stoul(token));
  }
}

template <int T>
TensorT<T>::TensorT() : device(MEMORY_HOST), dtype(TYPE_INVALID), shape({}) {}

template <int T>
TensorT<T>::TensorT(const MemoryDevice device, const DataType dtype, const std::vector<size_t> shape, int block_id,
                    const std::vector<int64_t>& strides, DataFormat data_format)
    : device(device), dtype(dtype), shape(shape), block_id(block_id), strides(strides), data_format(data_format) {
  InitializeDeviceTensor();
}

template <int T>
TensorT<T>::TensorT(const MemoryDevice device, const DataType dtype, const std::vector<size_t> shape, void* refer_ptr,
                    const std::vector<int64_t>& strides, DataFormat data_format)
    : device(device), dtype(dtype), shape(shape), refer_ptr(refer_ptr), strides(strides), data_format(data_format) {}

template <int T>
size_t TensorT<T>::GetElementNumber() const {
  return std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1), std::multiplies<size_t>());
}

template <int T>
size_t TensorT<T>::GetTotalBytes() const {
  return GetElementNumber() * GetTypeSize(dtype);
}

template <int T>
std::string TensorT<T>::DeviceToString() const {
  static const std::unordered_map<MemoryDevice, std::string> mem_to_string{{MEMORY_HOST, "host"},
                                                                           {MEMORY_DEVICE, "device"}};
  return mem_to_string.at(device);
}

template <int T>
std::string TensorT<T>::ToString() const {
  std::string memtype_str = DeviceToString();

  static const std::unordered_map<DataType, std::string> type_to_string{
      {TYPE_BOOL, "BOOL"},     {TYPE_UINT8, "UINT8"},     {TYPE_UINT16, "UINT16"},   {TYPE_UINT32, "UINT32"},
      {TYPE_UINT64, "UINT64"}, {TYPE_INT8, "INT8"},       {TYPE_INT16, "INT16"},     {TYPE_INT32, "INT32"},
      {TYPE_INT64, "INT64"},   {TYPE_BF16, "BF16"},       {TYPE_FP16, "FP16"},       {TYPE_FP32, "FP32"},
      {TYPE_FP64, "FP64"},     {TYPE_BYTES, "BYTES"},     {TYPE_INVALID, "INVALID"}, {TYPE_FP8_E4M3, "E4M3"},
      {TYPE_VOID, "VOID"},     {TYPE_POINTER, "POINTER"}, {TYPE_FP8_E5M2, "E5M2"},
  };

  DataType tensor_dtype = GetDeviceTensorDataType();
  if (tensor_dtype == TYPE_INVALID) {
    tensor_dtype = dtype;
  }

  return FormatStr("Tensor[where=%s, dtype=%s, shape=%s, block=%s]", memtype_str.c_str(),
                   type_to_string.at(tensor_dtype).c_str(), Vector2Str(shape).c_str(),
                   std::to_string(block_id).c_str());
}

template <int T>
std::string TensorT<T>::GetNumpyType() const {
  static const std::unordered_map<DataType, std::string> type_map{
      {TYPE_INVALID, "x"}, {TYPE_BOOL, "?"},    {TYPE_BYTES, "b"},   {TYPE_UINT8, "u1"},
      {TYPE_UINT16, "u2"}, {TYPE_UINT32, "u4"}, {TYPE_UINT64, "u8"}, {TYPE_POINTER, "u8"},
      {TYPE_INT8, "i1"},   {TYPE_INT16, "i2"},  {TYPE_INT32, "i4"},  {TYPE_INT64, "i8"},
      {TYPE_FP16, "f2"},   {TYPE_BF16, "f2"},   {TYPE_FP32, "f4"},   {TYPE_FP64, "f8"}};
  return type_map.count(dtype) ? type_map.at(dtype) : "x";
}

template <int T>
void TensorT<T>::SaveToFile(const std::string& file_path) {
  KLLM_LOG_DEBUG << fmt::format("Save {} To File {}", ToString(), file_path);

  size_t total_size = GetTotalBytes();
  void* cpu_data = malloc(total_size);
  void* tensor_data_ptr = GetPtr<void>();
  auto memcpy_type = (device == MEMORY_DEVICE) ? MEMCPY_DEVICE_TO_HOST : MEMCPY_HOST_TO_HOST;
  DeviceSynchronize();
  Memcpy(cpu_data, tensor_data_ptr, total_size, memcpy_type);

  std::filesystem::path dir_path = std::filesystem::path(file_path).parent_path();
  if (!std::filesystem::exists(dir_path)) {
    KLLM_LOG_WARNING << fmt::format("Do not exists the saved path {}", dir_path.string());
    std::filesystem::create_directories(dir_path);
  }

  std::ofstream file(file_path, std::ios::binary);
  if (!file.is_open()) {
    KLLM_LOG_ERROR << fmt::format("Could not open file {}", file_path);
    return;
  }
  // Header of numpy file
  file << "\x93NUMPY";
  uint8_t major_version = 1;
  uint8_t minor_version = 0;
  file.write(reinterpret_cast<const char*>(&major_version), sizeof(uint8_t));
  file.write(reinterpret_cast<const char*>(&minor_version), sizeof(uint8_t));
  std::stringstream header_stream;
  header_stream << "{'descr': '" << GetNumpyType() << "', 'fortran_order': False, 'shape': (";
  for (size_t i = 0; i < shape.size(); ++i) {
    header_stream << shape[i];
    if (shape.size() == 1 || i < shape.size() - 1) {
      header_stream << ",";
    }
  }
  // header_stream << "1600,";
  header_stream << ")}";
  int base_length = 6 + 4 + header_stream.str().size();
  int pad_length = 16 * ((base_length + 1 + 15) / 16);
  for (int i = 0; i < pad_length - base_length; ++i) {
    header_stream << ((i == pad_length - base_length - 1) ? "\n" : "\x20");
  }
  std::string header = header_stream.str();
  const uint16_t header_len = header.size();
  file.write(reinterpret_cast<const char*>(&header_len), sizeof(uint16_t));
  file << header;

  // Tensor Data
  file.write(reinterpret_cast<const char*>(cpu_data), total_size);
  file.close();
}

template <int T>
void TensorT<T>::LoadFromFile(const std::string& file_path) {
  KLLM_LOG_DEBUG << fmt::format("Load {} To Tensor {}", file_path, ToString());

  std::vector<size_t> file_data_shape;
  FILE* f_ptr = fopen(file_path.c_str(), "rb");
  if (f_ptr == nullptr) {
    throw std::runtime_error("Could not open file " + file_path);
  }
  uint32_t header_len, start_data;
  ParseNpyIntro(f_ptr, header_len, start_data);
  ParseNpyHeader(f_ptr, header_len, file_data_shape);

  const size_t file_elems_num =
      std::accumulate(file_data_shape.begin(), file_data_shape.end(), 1, std::multiplies<size_t>());

  size_t data_size = file_elems_num * GetTypeSize(dtype);

  if (data_size > GetTotalBytes()) {
    KLLM_THROW(fmt::format("LoadFromFile {} {} Bytes is more than tensor's total {} Bytes.", file_path, data_size,
                           GetTotalBytes()));
  }

  void* file_host_data_ptr = malloc(data_size);
  size_t n_elems = fread(file_host_data_ptr, GetTypeSize(dtype), file_elems_num, f_ptr);
  if (n_elems != file_elems_num) {
    KLLM_THROW(fmt::format("LoadFromFile {} to tensor failed.", file_path));
  }
  void* tensor_data_ptr = GetPtr<void>();
  auto memcpy_type = (device == MEMORY_DEVICE) ? MEMCPY_HOST_TO_DEVICE : MEMCPY_HOST_TO_HOST;
  DeviceSynchronize();
  Memcpy(tensor_data_ptr, file_host_data_ptr, data_size, memcpy_type);
  DeviceSynchronize();

  free(file_host_data_ptr);
  fclose(f_ptr);
}

template class TensorT<ACTIVE_DEVICE_TYPE>;

template float* TensorT<ACTIVE_DEVICE_TYPE>::GetPtr<float>() const;
template int* TensorT<ACTIVE_DEVICE_TYPE>::GetPtr<int>() const;
template int8_t* TensorT<ACTIVE_DEVICE_TYPE>::GetPtr<int8_t>() const;
template char* TensorT<ACTIVE_DEVICE_TYPE>::GetPtr<char>() const;
template void* TensorT<ACTIVE_DEVICE_TYPE>::GetPtr<void>() const;
template void** TensorT<ACTIVE_DEVICE_TYPE>::GetPtr<void*>() const;

}  // namespace ksana_llm
