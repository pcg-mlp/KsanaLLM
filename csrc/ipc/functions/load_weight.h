/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ipc/core/ipc_func_types.h"
#include "ipc/core/ipc_func_utils.h"
#include "runtime/weight_instance_inferface.h"

namespace ksana_llm {

class LoadWeightRequest : public IpcRequest {
 public:
  virtual ~LoadWeightRequest() {}
};

class LoadWeightResponse : public IpcResponse {
 public:
  virtual ~LoadWeightResponse() {}
  virtual Status Serialize(void* ptr, size_t* bytes) const override;
  virtual Status Deserialize(const void* ptr) override;

  // The loading result, succ or failure.
  Status result_status;
};

class LoadWeightCallable : public IpcCallable {
 public:
  virtual ~LoadWeightCallable() {}
  virtual Status Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) override;

  void SetWeightInstance(std::shared_ptr<WeightInstanceInterface> model_weight);

 private:
  std::shared_ptr<WeightInstanceInterface> model_weight_ = nullptr;
};

class LoadWeightCallback : public IpcCallback, public IpcWaiter {
 public:
  virtual ~LoadWeightCallback() {}
  virtual Status Call(const IpcResponse* response, IpcContext* context) override;

  // The final result stat of loading weight.
  Status load_status;
};

}  // namespace ksana_llm
