/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "configure/context.h"
#include "configure/environment.h"
#include "ipc/core/ipc_func_types.h"
#include "ipc/core/ipc_func_utils.h"
#include "utils/waiter.h"

#ifdef ENABLE_CUDA
#  include <cuda_runtime.h>
#  include <array>
#endif

namespace ksana_llm {

// [0]: custom_all_reduce_signal
// [1]: custom_all_reduce_input
// [2]: trt_all_reduce_buffer0
// [3]: trt_all_reduce_buffer1
// [4]: trt_all_reduce_buffer2
// [5]: trt_all_reduce_flag
// [6]: trt_all_reduce_workspace
inline constexpr size_t kCustomIpcHandleCount = 7;

class InitCustomCommunicationRequest : public IpcRequest {
 public:
  virtual ~InitCustomCommunicationRequest() {}
  virtual Status Serialize(void* ptr, size_t* bytes) const override;
  virtual Status Deserialize(const void* ptr) override;

#ifdef ENABLE_CUDA
  std::array<std::vector<cudaIpcMemHandle_t>, kCustomIpcHandleCount> ipc_handles{};
#endif
};

class InitCustomCommunicationResponse : public IpcResponse {
 public:
  virtual ~InitCustomCommunicationResponse() {}
  virtual Status Serialize(void* ptr, size_t* bytes) const override;
  virtual Status Deserialize(const void* ptr) override;

#ifdef ENABLE_CUDA
  std::array<cudaIpcMemHandle_t, kCustomIpcHandleCount> ipc_handles{};
#endif
};

class InitCustomCommunicationCallable : public IpcCallable {
 public:
  virtual ~InitCustomCommunicationCallable() {}
  virtual Status Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) override;

  void SetContext(std::shared_ptr<Context> context);

 private:
  std::shared_ptr<Context> context_;
};

class InitCustomCommunicationCallback : public IpcCallback, public IpcWaiter {
 public:
  virtual ~InitCustomCommunicationCallback() {}
  virtual Status Call(const IpcResponse* response, IpcContext* context) override;

#ifdef ENABLE_CUDA
  void SetCustomCommunicationIpcHandles(std::vector<cudaIpcMemHandle_t>* ipc_handles);
#endif

 private:
#ifdef ENABLE_CUDA
  std::vector<cudaIpcMemHandle_t>* ipc_handles_;
#endif
};

}  // namespace ksana_llm
