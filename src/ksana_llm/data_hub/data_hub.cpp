/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/data_hub/data_hub.h"

#include <cstddef>
#include <cstring>
#include <unordered_map>
#include <vector>
#include "ksana_llm/data_hub/schedule_output.h"
#include "ksana_llm/distributed/control_message.h"
#include "ksana_llm/distributed/packet_type.h"
#include "ksana_llm/distributed/raw_packet.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

// object pool of schedule output buffer.
ScheduleOutputPool* g_schedule_output_pool = nullptr;

// object pool of hidden unit buffer.
HiddenUnitBufferPool* g_hidden_unit_buffer_pool = nullptr;

// The current device hidden unit buffer.
HiddenUnitDeviceBuffer* g_hidden_unit_buffer = nullptr;

std::unordered_map<std::string, std::shared_ptr<ModelInstance>> g_model_instances;

void InitializeScheduleOutputPool() { g_schedule_output_pool = new ScheduleOutputPool(); }

void InitializeHiddenUnitBufferPool() { g_hidden_unit_buffer_pool = new HiddenUnitBufferPool(); }

void DestroyScheduleOutputPool() {
  if (g_schedule_output_pool) {
    delete g_schedule_output_pool;
    g_schedule_output_pool = nullptr;
  }
}

void DestroyHiddenUnitBufferPool() {
  if (g_hidden_unit_buffer_pool) {
    delete g_hidden_unit_buffer_pool;
    g_hidden_unit_buffer_pool = nullptr;
  }
}

void SetCurrentHiddenUnitBuffer(HiddenUnitDeviceBuffer* hidden_unit_buffer) {
  g_hidden_unit_buffer = hidden_unit_buffer;
}

HiddenUnitDeviceBuffer* GetCurrentHiddenUnitBuffer() { return g_hidden_unit_buffer; }

ScheduleOutputPool* GetScheduleOutputPool() { return g_schedule_output_pool; }

HiddenUnitBufferPool* GetHiddenUnitBufferPool() { return g_hidden_unit_buffer_pool; }

Status BroadcastScheduleOutput(ScheduleOutput* schedule_output) {
  GetScheduleOutputPool()->PutToSendQueue(schedule_output);
  return Status();
}

Status SendHiddenUnits(HiddenUnitDeviceBuffer* hidden_unit_buffer) {
  GetHiddenUnitBufferPool()->PutToSendQueue(hidden_unit_buffer);
  return Status();
}

Status SetModelInstance(const std::string model_name, std::shared_ptr<ModelInstance> model_instance) {
  g_model_instances[model_name] = model_instance;
  return Status();
}

std::shared_ptr<ModelInstance> GetModelInstance(const std::string& model_name) {
  if (g_model_instances.find(model_name) == g_model_instances.end()) {
    return nullptr;
  }
  return g_model_instances[model_name];
}

void DestroyModelInstance() {
  g_model_instances.clear();
}

}  // namespace ksana_llm
