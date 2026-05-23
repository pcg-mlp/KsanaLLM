/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "cache_manager/cache_layout/base_cache_layout.h"
#include "configure/environment.h"
#include "configure/schedule_config_parser.h"
#include "ipc/core/ipc_func_types.h"
#include "ipc/core/ipc_func_utils.h"
#include "utils/waiter.h"

namespace ksana_llm {

class ResetPipelineConfigRequest : public IpcRequest {
 public:
  virtual ~ResetPipelineConfigRequest() {}
  virtual Status Serialize(void* ptr, size_t* bytes) const override;
  virtual Status Deserialize(const void* ptr) override;

  // The pipeline config.
  PipelineConfig pipeline_config;
};

class ResetPipelineConfigResponse : public IpcResponse {
 public:
  virtual ~ResetPipelineConfigResponse() {}
};

class ResetPipelineConfigCallable : public IpcCallable {
 public:
  virtual ~ResetPipelineConfigCallable() {}
  virtual Status Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) override;

  void SetEnvironment(std::shared_ptr<Environment> env);

  void SetBlockCacheLayout(std::shared_ptr<BaseModelCacheLayout> block_cache_layout);

 private:
  std::shared_ptr<Environment> env_ = nullptr;

  // The model-level cache layout (holds all block cache layouts keyed by type).
  std::shared_ptr<BaseModelCacheLayout> block_cache_layout_ = nullptr;
};

class ResetPipelineConfigCallback : public IpcCallback, public IpcWaiter {
 public:
  virtual ~ResetPipelineConfigCallback() {}
  virtual Status Call(const IpcResponse* response, IpcContext* context) override;
};

}  // namespace ksana_llm
