/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include "ksana_llm/block_manager/block_manager.h"
#include "ksana_llm/utils/dtypes.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/string_utils.h"
#include "ksana_llm/utils/memory_utils.h"

namespace ksana_llm {

// The storage type, contigous or segmented.
enum StorageType { STORAGE_CONTIGUOUS, STORAGE_SEGMENTED };

struct Tensor {
  MemoryDevice device;
  StorageType storage;
  DataType dtype;
  std::vector<size_t> shape;

  // Use block id instead of physical address, so that the blockmanager could do defragmentation easily.
  // Get the real physical address through blockmanager.
  // For contigous tensor, it must have only one block.
  std::vector<int> blocks;

  Tensor();
  Tensor(const MemoryDevice _device, const StorageType _storage, const DataType _dtype,
         const std::vector<size_t> _shape, const std::vector<int> _blocks);

  size_t GetElementNumber() const;
  size_t GetTotalBytes() const;

  std::string ToString() const;
  std::string DeviceToString() const;

  static size_t GetTypeSize(DataType dtype);

  // Get the block ids.
  inline const std::vector<int>& GetBlockIds() const { return blocks; }

  // Get pointer of block
  template <typename T>
  inline std::vector<T*> GetPtrs() const {
    if (GetTensorType<T>() != dtype) {
      NLLM_LOG_DEBUG << "GetPtrs and dtype not matched.";
    }
    NLLM_CHECK_WITH_INFO(!blocks.empty(), "No available blocks");
    return GetBlockPtrs<T>(blocks);
  }

  template <typename T>
  inline T* GetPtr() const {
    if (GetTensorType<T>() != dtype) {
      // NLLM_LOG_DEBUG << "GetPtr and dtype not matched.";
    }
    NLLM_CHECK_WITH_INFO(!blocks.empty(), "No available blocks");
    return GetContiguousPtr<T>(blocks.front());
  }

  void SaveToFile(const std::string& file_path);

  std::string GetNumpyType() const;
};

// A container used to store multiple named tensors.
class TensorMap {
 public:
  TensorMap() = default;
  TensorMap(const std::unordered_map<std::string, Tensor>& tensor_map);
  TensorMap(const std::vector<Tensor>& tensor_map);
  TensorMap(std::initializer_list<std::pair<std::string, Tensor>> tensor_map);
  ~TensorMap();

  inline size_t GetSize() const { return tensor_map_.size(); }

  inline bool IsExist(const std::string& key) const { return tensor_map_.find(key) != tensor_map_.end(); }

  std::vector<std::string> GetKeys() const;

  inline void Insert(const std::string& key, const Tensor& value) {
    NLLM_CHECK_WITH_INFO(!IsExist(key), FormatStr("Duplicated key %s", key.c_str()));
    NLLM_CHECK_WITH_INFO(IsValid(value), FormatStr("A none tensor or nullptr is not allowed (key is %s)", key.c_str()));
    tensor_map_.insert({key, value});
  }

  inline void InsertIfValid(const std::string& key, const Tensor& value) {
    if (IsValid(value)) {
      Insert({key, value});
    }
  }

  inline void Insert(std::pair<std::string, Tensor> p) { tensor_map_.insert(p); }

  inline Tensor& Get(const std::string& key) {
    NLLM_CHECK_WITH_INFO(IsExist(key), FormatStr("Cannot find a tensor of name %s in the tensor map (keys: %s)",
                                                 key.c_str(), Vector2Str(GetKeys()).c_str()));
    return tensor_map_.at(key);
  }

  inline Tensor Get(const std::string& key) const {
    NLLM_CHECK_WITH_INFO(IsExist(key), FormatStr("Cannot find a tensor of name %s in the tensor map (keys: %s)",
                                                 key.c_str(), Vector2Str(GetKeys()).c_str()));
    return tensor_map_.at(key);
  }

  inline const std::vector<int>& GetBlockIds(const std::string& key) const {
    NLLM_CHECK_WITH_INFO(IsExist(key), FormatStr("Cannot find a tensor of name %s in the tensor map (keys: %s)",
                                                 key.c_str(), Vector2Str(GetKeys()).c_str()));
    return tensor_map_.at(key).GetBlockIds();
  }

  template <typename T>
  inline std::vector<T*> GetPtrs(const std::string& key) const {
    NLLM_CHECK_WITH_INFO(IsExist(key), FormatStr("Cannot find a tensor of name %s in the tensor map (keys: %s)",
                                                 key.c_str(), Vector2Str(GetKeys()).c_str()));
    return tensor_map_.at(key).GetPtrs<T>();
  }

  template <typename T>
  inline T* GetPtr(const std::string& key) const {
    NLLM_CHECK_WITH_INFO(IsExist(key), FormatStr("Cannot find a tensor of name %s in the tensor map (keys: %s)",
                                                 key.c_str(), Vector2Str(GetKeys()).c_str()));
    return tensor_map_.at(key).GetPtr<T>();
  }

  inline std::unordered_map<std::string, Tensor> GetMap() const { return tensor_map_; }

  inline std::unordered_map<std::string, Tensor>::iterator Begin() { return tensor_map_.begin(); }
  inline std::unordered_map<std::string, Tensor>::iterator End() { return tensor_map_.end(); }

  std::string ToString();

 private:
  std::unordered_map<std::string, Tensor> tensor_map_;

  // Check whether a tensor is valid.
  inline bool IsValid(const Tensor& tensor) { return tensor.GetElementNumber() > 0 && !tensor.blocks.empty(); }
};

template <typename T>
Status CreateTensor(Tensor& tensor, const size_t total_bytes, const int rank,
                    const MemoryDevice memory_device = MEMORY_GPU,
                    const StorageType storage_type = STORAGE_CONTIGUOUS);

Status DestroyTensor(Tensor& tensor, const int rank);

Status CreateTensor(Tensor& tensor, const std::vector<size_t> shape, const DataType dtype, const int rank,
                    const MemoryDevice memory_device, const StorageType storage_type);

}  // namespace ksana_llm
