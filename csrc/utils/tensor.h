/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "device/device_types.h"
#include "device/device_utils.h"
#include "utils/logger.h"
#include "utils/memory_utils.h"
#include "utils/status.h"
#include "utils/string_utils.h"

namespace ksana_llm {

// Host 内存来源枚举，用于区分释放路径
// 仅当 location == LOCATION_HOST 且 data_ptr 由 HostMemoryPool::Allocate() 获得时为 POOL
// 其他情况（device、shared buffer、直分配 host）为 DIRECT
enum class HostMemorySource {
  DIRECT,  // 直接分配（HostAlloc/HostFree）
  POOL     // 内存池分配（HostMemoryPool）
};

// The tensor define, only support contigous memory layout.
class Tensor {
 public:
  // Initialize a empty tensor.
  explicit Tensor(const std::string& name = "");

  // Initialize the tensor, if data_ptr is not null, it will be used as data buffer.
  Tensor(MemoryLocation location, DataType dtype, const std::vector<size_t>& shape, void* data_ptr, Stream* stream,
         const std::string& name = "");

  Tensor(MemoryLocation location, DataType dtype, const std::vector<size_t>& shape, const std::string& name = "")
      : Tensor(location, dtype, shape, nullptr, nullptr, name) {}

  Tensor(MemoryLocation location, DataType dtype, const std::vector<size_t>& shape, void* data_ptr,
         const std::string& name = "")
      : Tensor(location, dtype, shape, data_ptr, nullptr, name) {}

  ~Tensor();

  Tensor(const Tensor& other);

  Tensor& operator=(const Tensor& other);

 public:
  // Release and reallocate buffer memory.
  void ReallocateMemory(MemoryLocation location, DataType dtype, const std::vector<size_t>& shape);

  // Whether two tensor is equal.
  bool Equal(const Tensor& other) const;

  // Get the element number of this tensor.
  size_t GetElementNumber() const;

  // Get the byte of this tensor dtype.
  size_t GetDTypeSize() const;

  // Get the total bytes of this tensor.
  size_t GetTotalBytes() const;

  // Get a view of the current tensor that shares the underlying storage, possibly with a different shape and offset
  Tensor GetView(const std::vector<size_t>& shape, const size_t offset = 0) const;

  // Get pointer of block
  template <typename T>
  inline T* GetPtr(bool check_empty = true) const {
    return reinterpret_cast<T*>(GetPtrImpl(check_empty));
  }

  // Get tensor meta in string.
  std::string ToString() const;

  // Save to npy format file
  void SaveToNpyFile(const std::string& file_path);

  // Load tensor from npy file.
  void LoadFromNpyFile(const std::string& file_path);

 public:
  // Identify a tensor, for debug and readable.
  std::string name;

  // The memory location, host or device.
  MemoryLocation location = MemoryLocation::LOCATION_UNKNOWN;

  // The data type of current tensor.
  DataType dtype = DataType::TYPE_INVALID;

  // The shape of current tensor.
  std::vector<size_t> shape;

  // The data format, for ascend only now.
  DataFormat data_format = DataFormat::FORMAT_DEFAULT;

 public:
  // 获取 host 内存来源
  HostMemorySource GetHostMemorySource() const { return host_memory_source_; }

  // 便捷方法：判断是否来自内存池
  bool IsHostFromPool() const { return host_memory_source_ == HostMemorySource::POOL; }

  // 获取 host 内存来源的字符串表示（用于日志和调试）
  std::string GetHostMemorySourceString() const {
    return host_memory_source_ == HostMemorySource::POOL ? "pool" : "direct";
  }

 private:
  // Get location in string.
  std::string GetLocationString() const;

  // Assign every members.
  void AssignMembers(const Tensor& other);

  // The implementation of GetPtr.
  uint8_t* GetPtrImpl(bool check_empty) const;

  // Implement of allocation and release.
  void AcquireImpl();
  void ReleaseImpl();

 private:
  // The underlying memory address.
  void* data_ptr = nullptr;

  // Whether the data buffer is shared with others, the shared buffer will not be free.
  bool is_shared_buffer_ = false;

  std::shared_ptr<int> reference_ = nullptr;

  // Host 内存来源标记，用于区分释放路径
  // 仅当 location == LOCATION_HOST 时有意义
  HostMemorySource host_memory_source_ = HostMemorySource::DIRECT;

  // NOTE(karlluo): for NVIDIA GPU ref
  // https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-1/
  // for Huawei NPU ref
  // https://support.enflame-tech.com/onlinedoc_dev_3.3/_static/topsplatform_html/3-guide
  // /programing_guide/content/source/memory_model.html
  // create device memory space with stream and memory pool as extra memory management.
  Stream* stream_{nullptr};

 public:
  // TODO(yancyliu): The following number should be removed later.
  // ////////////////////////////////////////////////////////////////
  Tensor* scales = nullptr;
  Tensor* zeros = nullptr;

  // g_idx indicates the scales row number corresponding to each row of weight
  Tensor* g_idx = nullptr;
  // perm is converted from g_idx, perm=torch.argsort(g_idx), perm is used in marlin backend to support gptq-desc
  Tensor* perm = nullptr;

  Tensor* input_scales = nullptr;
  Tensor* weight_scales = nullptr;
  Tensor* weight_global_scale = nullptr;
  Tensor* pre_quant_scales = nullptr;

  // input_alpha is used as a coefficient for input
  Tensor* input_alpha = nullptr;
  float alpha = 0.f;

  void Fill(float f);
};

}  // namespace ksana_llm
