/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "connector/communicator/nvida/nccl_ipc_proxy.h"

#ifdef ENABLE_CUDA
#  include <gtest/gtest.h>
#  include <memory>
#  include <vector>

#  include "ipc/ipc_func_wrapper.h"
#  include "utils/status.h"

namespace ksana_llm {

class NcclIpcProxyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create a real IpcFuncWrapper for basic testing
    // Note: Full integration tests would require actual IPC setup
    ipc_wrapper_ = std::make_shared<IpcFuncWrapper>(1, true, 0);
    nccl_ipc_proxy_ = std::make_unique<NcclIpcProxy>(ipc_wrapper_);
  }

  std::shared_ptr<IpcFuncWrapper> ipc_wrapper_;
  std::unique_ptr<NcclIpcProxy> nccl_ipc_proxy_;
};

// Test basic construction and null pointer handling
TEST_F(NcclIpcProxyTest, Construction) { EXPECT_NE(nccl_ipc_proxy_, nullptr); }

// Test error handling when IpcFuncWrapper is not set
TEST_F(NcclIpcProxyTest, NullIpcWrapper) {
  NcclIpcProxy proxy_with_null(nullptr);

  ncclUniqueId comm_id = {0};
  Status status = proxy_with_null.InitializeCommGroup("test", comm_id, 0, 2, 0);
  EXPECT_FALSE(status.OK());
  EXPECT_EQ(status.GetCode(), RetCode::RET_INTERNAL_UNKNOWN_ERROR);
}

// Test Send with null wrapper
TEST_F(NcclIpcProxyTest, Send_NullWrapper) {
  NcclIpcProxy proxy_with_null(nullptr);
  std::vector<uint8_t> data = {1, 2, 3, 4};

  Status status = proxy_with_null.Send("test_group", 1, data, DataType::TYPE_FP32, 0);
  EXPECT_FALSE(status.OK());
  EXPECT_EQ(status.GetCode(), RetCode::RET_INTERNAL_UNKNOWN_ERROR);
}

// Test Recv with null wrapper
TEST_F(NcclIpcProxyTest, Recv_NullWrapper) {
  NcclIpcProxy proxy_with_null(nullptr);
  std::vector<uint8_t> data;

  Status status = proxy_with_null.Recv("test_group", 1, 100, DataType::TYPE_FP32, 0, data);
  EXPECT_FALSE(status.OK());
  EXPECT_EQ(status.GetCode(), RetCode::RET_INTERNAL_UNKNOWN_ERROR);
}

// Test SendGroup with null wrapper
TEST_F(NcclIpcProxyTest, SendGroup_NullWrapper) {
  NcclIpcProxy proxy_with_null(nullptr);
  std::vector<int> block_ids = {0, 1};
  std::vector<int64_t> offsets = {0, 100};
  std::vector<size_t> counts = {100, 100};

  Status status = proxy_with_null.SendGroup("test_group", 1, block_ids, offsets, counts, DataType::TYPE_FP32, 0);
  EXPECT_FALSE(status.OK());
  EXPECT_EQ(status.GetCode(), RetCode::RET_INTERNAL_UNKNOWN_ERROR);
}

// Test RecvGroup with null wrapper
TEST_F(NcclIpcProxyTest, RecvGroup_NullWrapper) {
  NcclIpcProxy proxy_with_null(nullptr);
  std::vector<int> block_ids = {0, 1};
  std::vector<int64_t> offsets = {0, 100};
  std::vector<size_t> counts = {100, 100};

  Status status = proxy_with_null.RecvGroup("test_group", 1, block_ids, offsets, counts, DataType::TYPE_FP32, 0);
  EXPECT_FALSE(status.OK());
  EXPECT_EQ(status.GetCode(), RetCode::RET_INTERNAL_UNKNOWN_ERROR);
}

// Test GetCommGroupInfo with null wrapper
TEST_F(NcclIpcProxyTest, GetCommGroupInfo_NullWrapper) {
  NcclIpcProxy proxy_with_null(nullptr);
  int rank = -1;
  int world_size = -1;

  Status status = proxy_with_null.GetCommGroupInfo("test_group", 0, rank, world_size);
  EXPECT_FALSE(status.OK());
  EXPECT_EQ(status.GetCode(), RetCode::RET_INTERNAL_UNKNOWN_ERROR);
}

}  // namespace ksana_llm

#endif  // ENABLE_CUDA