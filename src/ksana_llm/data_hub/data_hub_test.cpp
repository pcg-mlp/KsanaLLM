/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <vector>
#include "include/gtest/gtest.h"
#include "ksana_llm/data_hub/data_hub.h"
#include "ksana_llm/distributed/raw_packet.h"
#include "ksana_llm/runtime/infer_request.h"
#include "ksana_llm/runtime/infer_stage.h"
#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/singleton.h"
#include "test.h"

#include "ksana_llm/helpers/block_manager_test_helper.h"
#include "ksana_llm/helpers/environment_test_helper.h"

using namespace ksana_llm;

// For data verification.
constexpr size_t SCHEDULE_ID = 235;

class DataHubTest : public testing::Test {
 protected:
  void SetUp() override {
    std::string config_file = GetTestConfigFile();
    Singleton<Environment>::GetInstance()->ParseConfig(config_file);
    InitTestBlockManager(Singleton<Environment>::GetInstance().get());
  }

  void TearDown() override {}
};

TEST_F(DataHubTest, TestDataHub) {
  InitializeScheduleOutputPool();
  InitializeHiddenUnitBufferPool();

  EXPECT_TRUE(GetScheduleOutputPool() != nullptr);
  EXPECT_TRUE(GetHiddenUnitBufferPool() != nullptr);

  // Get device buffer, set and get.
  HiddenUnitDeviceBuffer* dev_hidden_unit = GetHiddenUnitBufferPool()->GetDeviceBuffer();
  dev_hidden_unit->schedule_id = SCHEDULE_ID;
  SetCurrentHiddenUnitBuffer(dev_hidden_unit);
  HiddenUnitDeviceBuffer* cur_dev_hidden_unit = GetCurrentHiddenUnitBuffer();
  EXPECT_EQ(cur_dev_hidden_unit->schedule_id, SCHEDULE_ID);

  // Get schedule output.
  ScheduleOutput* schedule_output = GetScheduleOutputPool()->GetScheduleOutput();
  schedule_output->schedule_id = SCHEDULE_ID;

  // Broadcast schedule output.
  BroadcastScheduleOutput(schedule_output);

  // get from send queue.
  Packet* send_schedule_output_packet = GetScheduleOutputPool()->GetFromSendQueue();

  ScheduleOutput* send_schedule_output = new ScheduleOutput();
  ScheduleOutputParser::DeserializeScheduleOutput(send_schedule_output_packet->body, send_schedule_output);

  EXPECT_EQ(send_schedule_output->schedule_id, SCHEDULE_ID);

  // Send hidden unit.
  SendHiddenUnits(cur_dev_hidden_unit);

  // get from conv queue
  Packet* send_packet = GetHiddenUnitBufferPool()->GetFromSendQueue();
  HiddenUnitHostBuffer* send_host_hidden_unit = reinterpret_cast<HiddenUnitHostBuffer*>(send_packet->body);
  EXPECT_EQ(send_host_hidden_unit->schedule_id, SCHEDULE_ID);

  GetHiddenUnitBufferPool()->FreeDeviceBuffer(dev_hidden_unit);
  GetScheduleOutputPool()->FreeScheduleOutput(schedule_output);

  DestroyScheduleOutputPool();
  DestroyHiddenUnitBufferPool();
}
