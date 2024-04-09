/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/utils/common_tensor.h"

#include <numeric>
#include <string>

#include "ksana_llm/utils/device_utils.h"

namespace ksana_llm {

template <int T>
TensorT<T>::TensorT() : device(MEMORY_HOST), dtype(TYPE_INVALID), shape({}) {}

template <int T>
TensorT<T>::TensorT(const MemoryDevice device, const DataType dtype, const std::vector<size_t> shape, int block_id,
                    const std::vector<int64_t>& strides, DataFormat data_format)
    : device(device), dtype(dtype), shape(shape), block_id(block_id), strides(strides), data_format(data_format) {
  InitializeDeviceTensor();
}

template <int T>
size_t TensorT<T>::GetElementNumber() const {
  if (shape.empty()) {
    return 0;
  }
  return std::accumulate(shape.begin(), shape.end(), (size_t)1, std::multiplies<size_t>());
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
      {TYPE_VOID, "VOID"},     {TYPE_POINTER, "POINTER"},
  };
  return FormatStr("Tensor[where=%s, dtype=%s, shape=%s, block=%s]", memtype_str.c_str(),
                   type_to_string.at(dtype).c_str(), Vector2Str(shape).c_str(), std::to_string(block_id).c_str());
}

template <int T>
const int TensorT<T>::GetBlockId() const {
  return block_id;
}

template <int T>
std::string TensorT<T>::GetNumpyType() const {
  static const std::unordered_map<DataType, std::string> type_map{
      {TYPE_INVALID, "x"}, {TYPE_BOOL, "?"},    {TYPE_BYTES, "b"}, {TYPE_UINT8, "u1"}, {TYPE_UINT16, "u2"},
      {TYPE_UINT32, "u4"}, {TYPE_UINT64, "u8"}, {TYPE_INT8, "i1"}, {TYPE_INT16, "i2"}, {TYPE_INT32, "i4"},
      {TYPE_INT64, "i8"},  {TYPE_FP16, "f2"},   {TYPE_FP32, "f4"}, {TYPE_FP64, "f8"}};
  return type_map.count(dtype) ? type_map.at(dtype) : "x";
}

template <int T>
void TensorT<T>::SaveToFile(const std::string& file_path) {
  NLLM_LOG_DEBUG << fmt::format("Save {} To File {}", ToString(), file_path);

  size_t total_size = GetTotalBytes();
  void* cpu_data = malloc(total_size);
  void* tensor_data_ptr = GetPtr<void>();
  auto memcpy_type = (device == MEMORY_DEVICE) ? MEMCPY_DEVICE_TO_HOST : MEMCPY_HOST_TO_HOST;
  DeviceSynchronize();
  Memcpy(cpu_data, tensor_data_ptr, total_size, memcpy_type);

  std::filesystem::path dir_path = std::filesystem::path(file_path).parent_path();
  if (!std::filesystem::exists(dir_path)) {
    NLLM_LOG_WARNING << fmt::format("Do not exists the saved path {}", dir_path.string());
    std::filesystem::create_directories(dir_path);
  }

  std::ofstream file(file_path, std::ios::binary);
  if (!file.is_open()) {
    NLLM_LOG_ERROR << fmt::format("Could not open file {}", file_path);
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

template class TensorT<ACTIVE_DEVICE_TYPE>;

template float* TensorT<ACTIVE_DEVICE_TYPE>::GetPtr<float>() const;
template int* TensorT<ACTIVE_DEVICE_TYPE>::GetPtr<int>() const;
template int8_t* TensorT<ACTIVE_DEVICE_TYPE>::GetPtr<int8_t>() const;
template char* TensorT<ACTIVE_DEVICE_TYPE>::GetPtr<char>() const;
template void* TensorT<ACTIVE_DEVICE_TYPE>::GetPtr<void>() const;
template void** TensorT<ACTIVE_DEVICE_TYPE>::GetPtr<void*>() const;

}  // namespace ksana_llm
