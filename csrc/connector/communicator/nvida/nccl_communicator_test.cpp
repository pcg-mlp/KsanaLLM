/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "connector/communicator/nvida/nccl_communicator.h"
#include <memory>
#include <string>
#include <vector>
#include "base64.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace ksana_llm {

#ifdef ENABLE_CUDA

// Mock class for NcclIpcProxy
class MockNcclIpcProxy : public NcclIpcProxy {
 public:
  MockNcclIpcProxy() : NcclIpcProxy(nullptr) {}

  MOCK_METHOD(Status, InitializeCommGroup,
              (const std::string& group_key, const ncclUniqueId& comm_id, int rank, int world_size, int executor_id),
              (override));
  MOCK_METHOD(Status, Send,
              (const std::string& group_key, int peer_rank, const std::vector<uint8_t>& data, DataType dtype,
               int executor_id),
              (override));
  MOCK_METHOD(Status, Recv,
              (const std::string& group_key, int peer_rank, size_t count, DataType dtype, int executor_id,
               std::vector<uint8_t>& data),
              (override));
  MOCK_METHOD(Status, SendGroup,
              (const std::string& group_key, int peer_rank, const std::vector<int>& block_ids,
               const std::vector<int64_t>& offsets, const std::vector<size_t>& counts, DataType dtype, int executor_id),
              (override));
  MOCK_METHOD(Status, RecvGroup,
              (const std::string& group_key, int peer_rank, const std::vector<int>& block_ids,
               const std::vector<int64_t>& offsets, const std::vector<size_t>& counts, DataType dtype, int executor_id),
              (override));
  MOCK_METHOD(Status, GetCommGroupInfo, (const std::string& group_key, int executor_id, int& rank, int& world_size),
              (override));
};

// Test fixture for NcclCommunicator
class NcclCommunicatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create test configuration
    config_.node_rank = 0;
    config_.world_size = 2;
    config_.device_count = 2;
    config_.group_role = GroupRole::PREFILL;

    // Create communicator
    communicator_ = std::make_unique<NcclCommunicator>(config_);
    communicator_->Initialize();

    // Create mock IPC proxy
    mock_ipc_proxy_ = std::make_shared<MockNcclIpcProxy>();
  }

  void TearDown() override {
    communicator_->Shutdown();
    communicator_.reset();
    mock_ipc_proxy_.reset();
  }

  // Helper to create a test comm group
  void SetupTestCommGroup(const std::string& group_key) {
    auto comm_group = std::make_unique<NcclCommGroup>();
    comm_group->device_count_ = config_.device_count;

    // Generate a fake NCCL ID
    ncclUniqueId fake_id;
    memset(&fake_id, 0, sizeof(ncclUniqueId));
    comm_group->comm_id = fake_id;
    comm_group->comm_id_str = base64::to_base64(std::string(reinterpret_cast<char*>(&fake_id), sizeof(ncclUniqueId)));

    communicator_->SetCommGroupForTest(group_key, std::move(comm_group));
  }

  ConnectorConfig config_;
  std::unique_ptr<NcclCommunicator> communicator_;
  std::shared_ptr<MockNcclIpcProxy> mock_ipc_proxy_;
};

// Test InitializeNcclCommsOnExecutors - no IPC proxy set
TEST_F(NcclCommunicatorTest, InitializeNcclCommsOnExecutors_NoIpcProxy) {
  const std::string group_key = "test_group";
  SetupTestCommGroup(group_key);

  ncclUniqueId fake_id;
  memset(&fake_id, 0, sizeof(ncclUniqueId));

  Status status = communicator_->InitializeNcclCommsOnExecutors(group_key, fake_id);
  EXPECT_FALSE(status.OK());
  EXPECT_EQ(status.GetCode(), RetCode::RET_INTERNAL_UNKNOWN_ERROR);
  EXPECT_TRUE(status.GetMessage().find("NCCL IPC proxy not set") != std::string::npos);
}

// Test InitializeNcclCommsOnExecutors - group not found
TEST_F(NcclCommunicatorTest, InitializeNcclCommsOnExecutors_GroupNotFound) {
  // Set up IPC proxy
  communicator_->SetNcclIpcProxy(mock_ipc_proxy_);

  ncclUniqueId fake_id;
  memset(&fake_id, 0, sizeof(ncclUniqueId));

  Status status = communicator_->InitializeNcclCommsOnExecutors("non_existent_group", fake_id);
  EXPECT_FALSE(status.OK());
  EXPECT_EQ(status.GetCode(), RetCode::RET_INVALID_ARGUMENT);
  EXPECT_TRUE(status.GetMessage().find("Communication group not found") != std::string::npos);
}

// Test InitializeNcclCommsOnExecutors - successful initialization
TEST_F(NcclCommunicatorTest, InitializeNcclCommsOnExecutors_Success) {
  const std::string group_key = "test_group";
  SetupTestCommGroup(group_key);

  // Set up IPC proxy
  communicator_->SetNcclIpcProxy(mock_ipc_proxy_);

  // Expect InitializeCommGroup to be called for each device
  EXPECT_CALL(*mock_ipc_proxy_, InitializeCommGroup(testing::_, testing::_, testing::_, testing::_, testing::_))
      .Times(config_.device_count)
      .WillRepeatedly(testing::Return(Status()));

  ncclUniqueId fake_id;
  memset(&fake_id, 0, sizeof(ncclUniqueId));

  Status status = communicator_->InitializeNcclCommsOnExecutors(group_key, fake_id);
  EXPECT_TRUE(status.OK());

  // Verify comm group state after initialization
  auto* comm_group = communicator_->GetCommunicatorGroup(group_key);
  ASSERT_NE(comm_group, nullptr);
  EXPECT_EQ(comm_group->world_size_, 2 * config_.world_size);  // global_world_size = 2 * world_size
  EXPECT_EQ(comm_group->device_ranks_.size(), static_cast<size_t>(config_.device_count));
}

// Test InitializeNcclCommsOnExecutors - IPC call failure
TEST_F(NcclCommunicatorTest, InitializeNcclCommsOnExecutors_IpcFailure) {
  const std::string group_key = "test_group";
  SetupTestCommGroup(group_key);

  // Set up IPC proxy
  communicator_->SetNcclIpcProxy(mock_ipc_proxy_);

  // Expect InitializeCommGroup to fail
  EXPECT_CALL(*mock_ipc_proxy_, InitializeCommGroup(testing::_, testing::_, testing::_, testing::_, testing::_))
      .WillRepeatedly(testing::Return(Status(RetCode::RET_RUNTIME_FAILED, "IPC call failed")));

  ncclUniqueId fake_id;
  memset(&fake_id, 0, sizeof(ncclUniqueId));

  Status status = communicator_->InitializeNcclCommsOnExecutors(group_key, fake_id);
  EXPECT_FALSE(status.OK());
  EXPECT_EQ(status.GetCode(), RetCode::RET_RUNTIME_FAILED);
}

// Test InitializeNcclCommsOnExecutors - verify CalcNcclRanks is called correctly
TEST_F(NcclCommunicatorTest, InitializeNcclCommsOnExecutors_CalcNcclRanksVerification) {
  // Verify CalcNcclRanks produces correct results for PREFILL role
  auto ranks = communicator_->CalcNcclRanks(GroupRole::PREFILL, 0, 2, 2, 0);
  // For PREFILL (role=0): cur_rank = 0*2 + 0*2 + 0 = 0
  // peer_rank = 1*2 + 0*2 + 0 = 2
  EXPECT_EQ(ranks.cur_rank, 0);
  EXPECT_EQ(ranks.peer_rank, 2);

  // Verify for device 1
  auto ranks1 = communicator_->CalcNcclRanks(GroupRole::PREFILL, 0, 2, 2, 1);
  EXPECT_EQ(ranks1.cur_rank, 1);
  EXPECT_EQ(ranks1.peer_rank, 3);

  // Verify for DECODE role
  auto ranks_decode = communicator_->CalcNcclRanks(GroupRole::DECODE, 0, 2, 2, 0);
  EXPECT_EQ(ranks_decode.cur_rank, 2);
  EXPECT_EQ(ranks_decode.peer_rank, 0);
}

// Test InitializeNcclCommsOnExecutors - verify correct ranks are passed to IPC
TEST_F(NcclCommunicatorTest, InitializeNcclCommsOnExecutors_VerifyRanksPassedToIpc) {
  const std::string group_key = "test_group";
  SetupTestCommGroup(group_key);

  // Set up IPC proxy
  communicator_->SetNcclIpcProxy(mock_ipc_proxy_);

  // Calculate expected global world size
  const int expected_global_world_size = 2 * config_.world_size;

  // Expect InitializeCommGroup to be called with correct parameters
  // For device 0: cur_rank = 0, executor_id = 0
  // For device 1: cur_rank = 1, executor_id = 1
  EXPECT_CALL(*mock_ipc_proxy_, InitializeCommGroup(group_key, testing::_, 0, expected_global_world_size, 0))
      .WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_ipc_proxy_, InitializeCommGroup(group_key, testing::_, 1, expected_global_world_size, 1))
      .WillOnce(testing::Return(Status()));

  ncclUniqueId fake_id;
  memset(&fake_id, 0, sizeof(ncclUniqueId));

  Status status = communicator_->InitializeNcclCommsOnExecutors(group_key, fake_id);
  EXPECT_TRUE(status.OK());
}

#endif  // ENABLE_CUDA

}  // namespace ksana_llm
