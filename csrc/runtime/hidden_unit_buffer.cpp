/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "runtime/hidden_unit_buffer.h"

#include <stdexcept>

#include "configure/environment.h"
#include "device/device_types.h"
#include "device/device_utils.h"
#include "profiler/profile_event.h"
#include "utils/memory_utils.h"
#include "utils/singleton.h"
#include "utils/status.h"

namespace ksana_llm {

using namespace std::placeholders;

static HiddenUnitBufferHelper* s_hidden_unit_helper = nullptr;

HiddenUnitBufferHelper::HiddenUnitBufferHelper(std::shared_ptr<Environment> env) {
  env_ = env;
  InitializeHiddenUnitConfig();

  // Used for allocate packet buffer only.
  s_hidden_unit_helper = this;
}

Packet* HiddenUnitBufferHelper::CreateHostBufferPacket() {
  if (s_hidden_unit_helper) {
    return s_hidden_unit_helper->CreateHostBuffer();
  }
  throw std::runtime_error("HiddenUnitBufferHelper is not initialized.");
}

size_t HiddenUnitBufferHelper::GetHostBufferPacketSize(Packet* packet) {
  if (s_hidden_unit_helper) {
    return s_hidden_unit_helper->GetHostBufferSize(packet);
  }
  throw std::runtime_error("HiddenUnitBufferHelper is not initialized.");
}

void HiddenUnitBufferHelper::InitializeHiddenUnitConfig() {
  ModelConfig model_config;
  Status status = env_->GetModelConfig(model_config);
  if (!status.OK()) {
    KLLM_LOG_ERROR << "GetModelConfig failed. status: " << status.ToString();
    return;
  }

  PipelineConfig pipeline_config;
  env_->GetPipelineConfig(pipeline_config);

  RuntimeConfig runtime_config;
  Singleton<Environment>::GetInstance()->GetRuntimeConfig(runtime_config);

  weight_type_ = model_config.weight_data_type;
  tensor_para_size_ = runtime_config.parallel_basic_config.tensor_parallel_size;
  max_token_num_ = runtime_config.max_step_token_num;
  hidden_unit_size_ = model_config.hidden_units * model_config.mhc_config.hc_mult;

  KLLM_LOG_INFO << "HiddenUnitBufferPool::InitializeHiddenUnitConfig weight_type:" << weight_type_
                << ", tensor_para_size:" << tensor_para_size_ << ", max_token_num:" << max_token_num_
                << ", hidden_unit_size:" << hidden_unit_size_;
}

Packet* HiddenUnitBufferHelper::CreateHostBuffer() {
  size_t extra_size = max_token_num_ * hidden_unit_size_ * GetTypeSize(weight_type_);
  size_t packet_body_size = sizeof(HiddenUnitHostBuffer) + extra_size;
  Packet* packet = reinterpret_cast<Packet*>(malloc(sizeof(Packet) + packet_body_size));
  if (packet == nullptr) {
    KLLM_LOG_ERROR << "CreateHostBuffer error, allocate memory failed.";
    return packet;
  }

  packet->type = PacketType::DATA_REQ_HIDDEN_UNIT;
  packet->size = packet_body_size;

  HiddenUnitHostBuffer* hidden_unit = reinterpret_cast<HiddenUnitHostBuffer*>(packet->body);

  hidden_unit->shape_dims[0] = max_token_num_;
  hidden_unit->shape_dims[1] = hidden_unit_size_;

  return packet;
}

HiddenUnitDeviceBuffer* HiddenUnitBufferHelper::CreateDeviceBuffer() {
  int rank;
  GetDevice(&rank);

  HiddenUnitDeviceBuffer* hidden_unit_buffer = new HiddenUnitDeviceBuffer();
  hidden_unit_buffer->tensor =
      Tensor(MemoryLocation::LOCATION_DEVICE, weight_type_, {max_token_num_, hidden_unit_size_});
#ifdef ENABLE_ACL
  hidden_unit_buffer->prefill_tensor =
      Tensor(MemoryLocation::LOCATION_DEVICE, weight_type_, {max_token_num_, hidden_unit_size_});
#endif

#ifdef ENABLE_ACL
  hidden_unit_buffer->prefill_enabled = false;
  hidden_unit_buffer->decode_enabled = false;
#endif

  KLLM_LOG_DEBUG << "HiddenUnitBufferHelper::CreateDeviceBuffer, shape:"
                 << Vector2Str(std::vector<size_t>(hidden_unit_buffer->tensor.shape))
                 << ", max_token_num:" << max_token_num_ << ", hidden_unit_size:" << hidden_unit_size_;

  return hidden_unit_buffer;
}

Status HiddenUnitBufferHelper::ConvertHostBufferToDevice(HiddenUnitDeviceBuffer* hidden_unit_dev,
                                                         HiddenUnitHostBuffer* hidden_unit_host) {
  size_t buffer_bytes = hidden_unit_host->shape_dims[0] * hidden_unit_host->shape_dims[1] * GetTypeSize(weight_type_);
  std::vector<size_t> buffer_shape = {hidden_unit_host->shape_dims[0], hidden_unit_host->shape_dims[1]};

#ifdef ENABLE_ACL
  size_t prefill_buffer_bytes =
      hidden_unit_host->prefill_shape_dims[0] * hidden_unit_host->prefill_shape_dims[1] * GetTypeSize(weight_type_);
  std::vector<size_t> prefill_buffer_shape = {hidden_unit_host->prefill_shape_dims[0],
                                              hidden_unit_host->prefill_shape_dims[1]};
  hidden_unit_dev->decode_enabled = buffer_bytes > 0;
  hidden_unit_dev->prefill_enabled = prefill_buffer_bytes > 0;
#endif

  if (buffer_bytes > 0) {
    Memcpy(hidden_unit_dev->tensor.GetPtr<void>(), hidden_unit_host->data, buffer_bytes, MEMCPY_HOST_TO_DEVICE);
    hidden_unit_dev->tensor.shape = buffer_shape;
  }
#ifdef ENABLE_ACL
  if (prefill_buffer_bytes > 0) {
    Memcpy(hidden_unit_dev->prefill_tensor.GetPtr<void>(), hidden_unit_host->data + buffer_bytes, prefill_buffer_bytes,
           MEMCPY_HOST_TO_DEVICE);
    hidden_unit_dev->prefill_tensor.shape = prefill_buffer_shape;
  }
#endif

  return Status();
}

Status HiddenUnitBufferHelper::ConvertDeviceBufferToHost(HiddenUnitHostBuffer* hidden_unit_host,
                                                         HiddenUnitDeviceBuffer* hidden_unit_dev) {
  std::vector<size_t> buffer_shape = hidden_unit_dev->tensor.shape;
  hidden_unit_host->shape_dims[0] = buffer_shape[0];
  hidden_unit_host->shape_dims[1] = buffer_shape[1];
#ifdef ENABLE_ACL
  if (!hidden_unit_dev->decode_enabled) {
    hidden_unit_host->shape_dims[0] = 0;
    hidden_unit_host->shape_dims[1] = 0;
  }
  std::vector<size_t> prefill_buffer_shape = hidden_unit_dev->prefill_tensor.shape;
  hidden_unit_host->prefill_shape_dims[0] = prefill_buffer_shape[0];
  hidden_unit_host->prefill_shape_dims[1] = prefill_buffer_shape[1];

  if (!hidden_unit_dev->prefill_enabled) {
    hidden_unit_host->prefill_shape_dims[0] = 0;
    hidden_unit_host->prefill_shape_dims[1] = 0;
  }
#endif

  size_t buffer_bytes = hidden_unit_host->shape_dims[0] * hidden_unit_host->shape_dims[1] * GetTypeSize(weight_type_);
#ifdef ENABLE_ACL
  size_t prefill_buffer_bytes =
      hidden_unit_host->prefill_shape_dims[0] * hidden_unit_host->prefill_shape_dims[1] * GetTypeSize(weight_type_);
#endif
  if (buffer_bytes > 0) {
    Memcpy(hidden_unit_host->data, hidden_unit_dev->tensor.GetPtr<void>(), buffer_bytes, MEMCPY_DEVICE_TO_HOST);
  }
#ifdef ENABLE_ACL
  if (prefill_buffer_bytes > 0) {
    Memcpy(hidden_unit_host->data + buffer_bytes, hidden_unit_dev->prefill_tensor.GetPtr<void>(), prefill_buffer_bytes,
           MEMCPY_DEVICE_TO_HOST);
  }
#endif

  return Status();
}

size_t HiddenUnitBufferHelper::GetHostBufferSize(Packet* packet) {
  HiddenUnitHostBuffer* hidden_unit = reinterpret_cast<HiddenUnitHostBuffer*>(packet->body);
  size_t host_packet_size = sizeof(HiddenUnitHostBuffer) +
                            (hidden_unit->shape_dims[0] * hidden_unit->shape_dims[1] * GetTypeSize(weight_type_));
#ifdef ENABLE_ACL
  host_packet_size +=
      (hidden_unit->prefill_shape_dims[0] * hidden_unit->prefill_shape_dims[1] * GetTypeSize(weight_type_));
#endif
  return host_packet_size;
}

void HiddenUnitBufferHelper::ResetDeviceBuffer(HiddenUnitDeviceBuffer* hidden_unit_buffer) {
#ifdef ENABLE_ACL
  hidden_unit_buffer->prefill_enabled = false;
  hidden_unit_buffer->decode_enabled = false;
#endif
}

}  // namespace ksana_llm
