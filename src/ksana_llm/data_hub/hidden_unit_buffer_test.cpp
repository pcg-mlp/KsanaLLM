/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <vector>
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

class HiddenUnitBufferTest : public testing::Test {
 protected:
  void SetUp() override {
    std::string config_file = GetTestConfigFile();
    Singleton<Environment>::GetInstance()->ParseConfig(config_file);
    InitTestBlockManager(Singleton<Environment>::GetInstance().get());
  }

  void TearDown() override {}

  void InitBufferSize() {
    std::unordered_map<std::string, ModelConfig> model_configs;
    Singleton<Environment>::GetInstance()->GetModelConfigs(model_configs);
    if (model_configs.empty()) {
      throw std::runtime_error("No model_config provided.");
    }

    ModelConfig model_config = model_configs.begin()->second;

    weight_type_ = model_config.weight_data_type;
    tensor_para_size_ = model_config.tensor_para_size;
    max_token_num_ = model_config.max_scheduler_token_num;
    hidden_unit_size_ = model_config.size_per_head * model_config.head_num;
  }

  void SetHiddenUnitBuffer(HiddenUnitHostBuffer* host_hidden_unit, size_t dim0, size_t dim1) {
    host_hidden_unit->shape_dims[0] = dim0;
    host_hidden_unit->shape_dims[1] = dim1;

    if (weight_type_ == DataType::TYPE_FP16) {
      size_t buffer_size =
          host_hidden_unit->shape_dims[0] * host_hidden_unit->shape_dims[1] * GetTypeSize(weight_type_);

      for (size_t i = 0; i < host_hidden_unit->tensor_parallel; ++i) {
#ifdef ENABLE_CUDA
        std::vector<half> vec;
        for (size_t j = 0; j < dim0 * dim1; ++j) {
          vec.push_back(1.0 * (j + 1) * (i + 1));
        }
#endif

#ifdef ENABLE_ACL
        std::vector<aclFloat16> vec;
        for (size_t j = 0; j < dim0 * dim1; ++j) {
          vec.push_back(aclFloatToFloat16(1.0 * (j + 1) * (i + 1)));
        }
#endif
        memcpy(host_hidden_unit->data + (i * buffer_size), vec.data(), buffer_size);
      }
    }
  }

  bool CheckHiddenUnitBuffer(HiddenUnitHostBuffer* src_host_hidden_unit, HiddenUnitHostBuffer* dst_host_hidden_unit) {
    if (src_host_hidden_unit->tensor_parallel != dst_host_hidden_unit->tensor_parallel) {
      return false;
    }

    if (src_host_hidden_unit->shape_dims[0] != dst_host_hidden_unit->shape_dims[0] ||
        src_host_hidden_unit->shape_dims[1] != dst_host_hidden_unit->shape_dims[1]) {
      return false;
    }

    size_t buffer_element_num = src_host_hidden_unit->shape_dims[0] * src_host_hidden_unit->shape_dims[1];

    for (size_t i = 0; i < src_host_hidden_unit->tensor_parallel; ++i) {
      for (size_t j = 0; j < (src_host_hidden_unit->shape_dims[0] * src_host_hidden_unit->shape_dims[1]); ++j) {
#ifdef ENABLE_CUDA
        if (src_host_hidden_unit->data[i * buffer_element_num + j] !=
            dst_host_hidden_unit->data[i * buffer_element_num + j]) {
          return false;
        }
#endif

#ifdef ENABLE_ACL
        if (aclFloat16ToFloat(src_host_hidden_unit->data[i * buffer_element_num + j]) !=
            aclFloat16ToFloat(dst_host_hidden_unit->data[i * buffer_element_num + j])) {
          return false;
        }
#endif
      }
    }

    return true;
  }

 protected:
  DataType weight_type_;
  size_t max_token_num_;
  size_t tensor_para_size_;
  size_t hidden_unit_size_;
};

TEST_F(HiddenUnitBufferTest, TestConvert) {
  InitializeHiddenUnitBufferPool();
  InitBufferSize();

  // Get a host buffer.
  Packet* packet = GetHiddenUnitBufferPool()->GetHostBuffer();
  EXPECT_TRUE(packet != nullptr);

  HiddenUnitHostBuffer* host_hidden_unit = reinterpret_cast<HiddenUnitHostBuffer*>(packet->body);

  EXPECT_EQ(host_hidden_unit->shape_dims[0], max_token_num_);
  EXPECT_EQ(host_hidden_unit->shape_dims[1], hidden_unit_size_);
  EXPECT_EQ(host_hidden_unit->tensor_parallel, tensor_para_size_);

  // Get a device buffer.
  HiddenUnitDeviceBuffer* dev_hidden_unit = GetHiddenUnitBufferPool()->GetDeviceBuffer();
  EXPECT_EQ(dev_hidden_unit->tensors.size(), tensor_para_size_);
  EXPECT_EQ(dev_hidden_unit->tensors[0].shape[0], max_token_num_);
  EXPECT_EQ(dev_hidden_unit->tensors[0].shape[1], hidden_unit_size_);

  // Set value.
  SetHiddenUnitBuffer(host_hidden_unit, 4, 3);
  EXPECT_EQ(host_hidden_unit->shape_dims[0], 4);
  EXPECT_EQ(host_hidden_unit->shape_dims[1], 3);

  // Covert to device.
  GetHiddenUnitBufferPool()->ConvertHostBufferToDevice(dev_hidden_unit, host_hidden_unit);
  EXPECT_EQ(host_hidden_unit->shape_dims[0], dev_hidden_unit->tensors[0].shape[0]);
  EXPECT_EQ(host_hidden_unit->shape_dims[1], dev_hidden_unit->tensors[0].shape[1]);

  // Convert back to host.
  Packet* new_packet = GetHiddenUnitBufferPool()->GetHostBuffer();
  EXPECT_TRUE(new_packet != nullptr);
  HiddenUnitHostBuffer* new_host_hidden_unit = reinterpret_cast<HiddenUnitHostBuffer*>(new_packet->body);
  GetHiddenUnitBufferPool()->ConvertDeviceBufferToHost(new_host_hidden_unit, dev_hidden_unit);
  EXPECT_EQ(host_hidden_unit->shape_dims[0], new_host_hidden_unit->shape_dims[0]);
  EXPECT_EQ(host_hidden_unit->shape_dims[1], new_host_hidden_unit->shape_dims[1]);

  // Check value.
  EXPECT_TRUE(CheckHiddenUnitBuffer(host_hidden_unit, new_host_hidden_unit));

  // Free buffer.
  GetHiddenUnitBufferPool()->FreeHostBuffer(packet);
  GetHiddenUnitBufferPool()->FreeHostBuffer(new_packet);
  GetHiddenUnitBufferPool()->FreeDeviceBuffer(dev_hidden_unit);

  DestroyHiddenUnitBufferPool();
}

TEST_F(HiddenUnitBufferTest, TestHiddenUnitBufferPool) {
  InitializeHiddenUnitBufferPool();
  InitBufferSize();

  // Get a host buffer.
  Packet* packet = GetHiddenUnitBufferPool()->GetHostBuffer();
  EXPECT_TRUE(packet != nullptr);

  // Assign a id.
  HiddenUnitHostBuffer* host_hidden_unit = reinterpret_cast<HiddenUnitHostBuffer*>(packet->body);
  host_hidden_unit->schedule_id = 235;

  // Put to recv queue and get it.
  GetHiddenUnitBufferPool()->PutToHostRecvQueue(packet);
  Packet* recv_packet = GetHiddenUnitBufferPool()->GetFromHostRecvQueue();
  HiddenUnitHostBuffer* recv_host_hidden_unit = reinterpret_cast<HiddenUnitHostBuffer*>(recv_packet->body);
  EXPECT_EQ(host_hidden_unit->schedule_id, recv_host_hidden_unit->schedule_id);

  // Get a device buffer and converted from a host buffer.
  HiddenUnitDeviceBuffer* dev_hidden_unit = GetHiddenUnitBufferPool()->GetDeviceBuffer();
  GetHiddenUnitBufferPool()->ConvertHostBufferToDevice(dev_hidden_unit, recv_host_hidden_unit);
  EXPECT_EQ(host_hidden_unit->schedule_id, dev_hidden_unit->schedule_id);

  // Put to send queue and get it.
  GetHiddenUnitBufferPool()->PutToSendQueue(dev_hidden_unit);
  Packet* send_packet = GetHiddenUnitBufferPool()->GetFromSendQueue();
  HiddenUnitHostBuffer* send_host_hidden_unit = reinterpret_cast<HiddenUnitHostBuffer*>(send_packet->body);
  EXPECT_EQ(host_hidden_unit->schedule_id, send_host_hidden_unit->schedule_id);

  // Free buffers.
  GetHiddenUnitBufferPool()->FreeHostBuffer(packet);
  GetHiddenUnitBufferPool()->FreeDeviceBuffer(dev_hidden_unit);

  DestroyHiddenUnitBufferPool();
}
