/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <vector>
#include "ksana_llm/data_hub/hidden_unit_buffer.h"
#include "ksana_llm/data_hub/schedule_output.h"
#include "ksana_llm/distributed/packet_type.h"
#include "ksana_llm/distributed/raw_packet.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

// Initialize and destroy the data hub.
void InitializeScheduleOutputPool();
void InitializeHiddenUnitBufferPool();

void DestroyScheduleOutputPool();
void DestroyHiddenUnitBufferPool();

// Get the object pool of schedule output.
ScheduleOutputPool* GetScheduleOutputPool();

// Get the object pool of hidden unit buffer.
HiddenUnitBufferPool* GetHiddenUnitBufferPool();

// Set and get current device buffer for compute thread.
void SetCurrentHiddenUnitBuffer(HiddenUnitDeviceBuffer* hidden_unit_buffer);
HiddenUnitDeviceBuffer* GetCurrentHiddenUnitBuffer();

// Broadcast to all workers.
Status BroadcastScheduleOutput(ScheduleOutput* schedule_output);

// Send hidden_units to downstream.
Status SendHiddenUnits(HiddenUnitDeviceBuffer* hidden_unit_buffer);

// Get and set model instance.
Status SetModelInstance(const std::string model_name, std::shared_ptr<ModelInstance> model_instance);
std::shared_ptr<ModelInstance> GetModelInstance(const std::string& model_name);
void DestroyModelInstance();

}  // namespace ksana_llm
