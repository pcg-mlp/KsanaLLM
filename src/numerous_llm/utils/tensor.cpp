/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/utils/tensor.h"

#include <numeric>

namespace numerous_llm {

Tensor::Tensor() : device(MEMORY_CPU), storage(STORAGE_CONTIGUOUS), dtype(TYPE_INVALID), shape({}), blocks({}) {}

Tensor::Tensor(const MemoryDevice _device, const StorageType _storage, const DataType _dtype,
               const std::vector<size_t> _shape, const std::vector<int> _blocks)
    : device(_device), storage(_storage), dtype(_dtype), shape(_shape), blocks(_blocks) {}

size_t Tensor::GetElementNumber() const {
  if (blocks.size() == 0 || shape.size() == 0) {
    return 0;
  }
  return std::accumulate(shape.begin(), shape.end(), (size_t)1, std::multiplies<size_t>());
}

size_t Tensor::GetTotalBytes() const { return GetElementNumber() * Tensor::GetTypeSize(dtype); }

std::string Tensor::DeviceToString() const {
  static const std::unordered_map<MemoryDevice, std::string> mem_to_string{
      {MEMORY_CPU, "CPU"}, {MEMORY_CPU_PINNED, "CPU_PINNED"}, {MEMORY_GPU, "GPU"}};
  return mem_to_string.at(device);
}

std::string Tensor::ToString() const {
  std::string memtype_str = DeviceToString();

  static const std::unordered_map<DataType, std::string> type_to_string{
      {TYPE_BOOL, "BOOL"},     {TYPE_UINT8, "UINT8"},     {TYPE_UINT16, "UINT16"},   {TYPE_UINT32, "UINT32"},
      {TYPE_UINT64, "UINT64"}, {TYPE_INT8, "INT8"},       {TYPE_INT16, "INT16"},     {TYPE_INT32, "INT32"},
      {TYPE_INT64, "INT64"},   {TYPE_BF16, "BF16"},       {TYPE_FP16, "FP16"},       {TYPE_FP32, "FP32"},
      {TYPE_FP64, "FP64"},     {TYPE_BYTES, "BYTES"},     {TYPE_INVALID, "INVALID"}, {TYPE_FP8_E4M3, "E4M3"},
      {TYPE_VOID, "VOID"},     {TYPE_POINTER, "POINTER"},
  };
  return FormatStr("Tensor[where=%s, dtype=%s, shape=%s, blocks=%s]", memtype_str.c_str(),
                   type_to_string.at(dtype).c_str(), Vector2Str(shape).c_str(), Vector2Str(blocks).c_str());
}

std::string Tensor::GetNumpyType() const {
  static const std::unordered_map<DataType, std::string> type_map{
      {TYPE_INVALID, "x"}, {TYPE_BOOL, "?"},    {TYPE_BYTES, "b"}, {TYPE_UINT8, "u1"}, {TYPE_UINT16, "u2"},
      {TYPE_UINT32, "u4"}, {TYPE_UINT64, "u8"}, {TYPE_INT8, "i1"}, {TYPE_INT16, "i2"}, {TYPE_INT32, "i4"},
      {TYPE_INT64, "i8"},  {TYPE_FP16, "f2"},   {TYPE_FP32, "f4"}, {TYPE_FP64, "f8"}};
  return type_map.count(dtype) ? type_map.at(dtype) : "x";
}

size_t Tensor::GetTypeSize(DataType dtype) {
  static const std::unordered_map<DataType, size_t> type_map{{TYPE_BOOL, sizeof(bool)},
                                                             {TYPE_BYTES, sizeof(char)},
                                                             {TYPE_UINT8, sizeof(uint8_t)},
                                                             {TYPE_UINT16, sizeof(uint16_t)},
                                                             {TYPE_UINT32, sizeof(uint32_t)},
                                                             {TYPE_UINT64, sizeof(uint64_t)},
                                                             {TYPE_INT8, sizeof(int8_t)},
                                                             {TYPE_INT16, sizeof(int16_t)},
                                                             {TYPE_INT32, sizeof(int32_t)},
                                                             {TYPE_INT64, sizeof(int64_t)},
#ifdef ENABLE_BF16
                                                             {TYPE_BF16, sizeof(__nv_bfloat16)},
#endif
#ifdef ENABLE_FP8
                                                             {TYPE_FP8_E4M3, sizeof(__nv_fp8_e4m3)},
#endif
                                                             {TYPE_FP16, sizeof(half)},
                                                             {TYPE_FP32, sizeof(float)},
                                                             {TYPE_FP64, sizeof(double)},
                                                             {TYPE_POINTER, sizeof(void*)}};
  return type_map.at(dtype);
}

void Tensor::SaveToFile(const std::string& file_path) {
  cudaDeviceSynchronize();
  NLLM_LOG_INFO << fmt::format("Save {} To File {}", ToString(), file_path);
  size_t total_size = GetTotalBytes();
  void* cpu_data = malloc(total_size);
  void* tensor_data_ptr = GetPtr<void>();
  auto memcpy_type = device == MEMORY_GPU ? cudaMemcpyDeviceToHost : cudaMemcpyHostToHost;
  printf("addr = %p, type = %d, size = %d\n", tensor_data_ptr, memcpy_type == cudaMemcpyDeviceToHost, total_size);

  cudaError_t ret = cudaMemcpy(cpu_data, tensor_data_ptr, total_size, memcpy_type);
  if (ret != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(ret) << std::endl;
    exit(1);
  }
  printf("第一个值  %d\n", ((char*)cpu_data)[0]);
  printf("第二个值  %d\n", ((char*)cpu_data)[1]);

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

TensorMap::TensorMap(const std::unordered_map<std::string, Tensor>& tensor_map) {
  for (auto& kv : tensor_map) {
    if (IsValid(kv.second)) {
      Insert(kv.first, kv.second);
    } else {
      NLLM_LOG_DEBUG << FormatStr("%s is not a valid tensor, skipping insert into TensorMap", kv.first.c_str());
    }
  }
}

TensorMap::TensorMap(const std::vector<Tensor>& tensor_map) {
  for (size_t i = 0; i < tensor_map.size(); i++) {
    Insert(std::to_string(i), tensor_map[i]);
  }
}

TensorMap::TensorMap(std::initializer_list<std::pair<std::string, Tensor>> tensor_map) {
  for (auto& pair : tensor_map) {
    if (IsValid(pair.second)) {
      Insert(pair.first, pair.second);
    } else {
      NLLM_LOG_DEBUG << FormatStr("%s is not a valid tensor, skipping insert into TensorMap", pair.first.c_str());
    }
  }
}

TensorMap::~TensorMap() { tensor_map_.clear(); }

std::vector<std::string> TensorMap::GetKeys() const {
  std::vector<std::string> key_names;
  for (auto& kv : tensor_map_) {
    key_names.push_back(kv.first);
  }
  return key_names;
}

std::string TensorMap::ToString() {
  std::stringstream ss;
  ss << "{";
  std::vector<std::string> key_names = GetKeys();
  for (size_t i = 0; i < tensor_map_.size(); ++i) {
    ss << key_names[i] << ": " << Get(key_names[i]).ToString();
    if (i < tensor_map_.size() - 1) {
      ss << ", ";
    }
  }
  ss << "}";
  return ss.str();
}

}  // namespace numerous_llm
