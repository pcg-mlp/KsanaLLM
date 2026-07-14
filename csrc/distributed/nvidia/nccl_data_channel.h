/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <thread>

#include "configure/context.h"
#include "runtime/hidden_unit_buffer.h"
#include "utils/blocking_queue.h"
#include "utils/waiter.h"

namespace ksana_llm {

// A fast data channel implement using NCCL.
class NcclDataChannel {
 public:
  explicit NcclDataChannel(std::shared_ptr<Context> context);
  ~NcclDataChannel();

  Status InitializeNcclCluster(ncclUniqueId nccl_unique_id);
  Status DestroyNcclCluster();

  // Send & recv data.
  Status SendData(HiddenUnitDeviceBuffer* hidden_unit_dev, const std::vector<size_t>& shape, DataType data_type);
  Status RecvData(HiddenUnitDeviceBuffer* hidden_unit_dev, const std::vector<size_t>& shape, DataType data_type);

 protected:
  // Convert data type to nccl data type.
#ifdef ENABLE_CUDA
  Status GetNcclDataType(DataType dtype, ncclDataType_t& nccl_dtype);
#endif

 private:
  int device_id_;

#ifdef ENABLE_CUDA
  ncclUniqueId nccl_unique_id_;

  // The communicator for current device.
  ncclComm_t communicator_;

  DistributedCommunicationType comm_type_ = DistributedCommunicationType::DEFAULT;
#endif

  // The upstream and downstream device ranks.
  std::vector<int> upstream_dev_ranks_;
  std::vector<int> downstream_dev_ranks_;

  // The context.
  std::shared_ptr<Context> context_ = nullptr;

  // Whether channel is terminated.
  bool terminated_ = false;
};

}  // namespace ksana_llm
