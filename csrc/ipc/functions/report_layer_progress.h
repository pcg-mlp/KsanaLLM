/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ipc/core/ipc_func_types.h"
#include "utils/blocking_queue.h"

namespace ksana_llm {

// 层进度信息
struct LayerProgressInfo {
  int device_id;
  int layer_index;

  LayerProgressInfo() : device_id(-1), layer_index(-1) {}
  LayerProgressInfo(int dev_id, int layer_idx) : device_id(dev_id), layer_index(layer_idx) {}
};

class ReportLayerProgressRequest : public IpcRequest {
 public:
  virtual ~ReportLayerProgressRequest() {}
  virtual Status Serialize(void* ptr, size_t* bytes) const override;
  virtual Status Deserialize(const void* ptr) override;

  int device_id;
  int layer_index;
};

class ReportLayerProgressResponse : public IpcResponse {
 public:
  virtual ~ReportLayerProgressResponse() {}
};

class ReportLayerProgressCallable : public IpcCallable {
 public:
  virtual ~ReportLayerProgressCallable() {}
  virtual Status Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) override;

  void SetRecvQueue(std::shared_ptr<BlockingQueue<LayerProgressInfo>> queue);

 private:
  std::shared_ptr<BlockingQueue<LayerProgressInfo>> layer_progress_queue_;
};

class ReportLayerProgressCallback : public IpcCallback {
 public:
  virtual ~ReportLayerProgressCallback() {}
};

}  // namespace ksana_llm