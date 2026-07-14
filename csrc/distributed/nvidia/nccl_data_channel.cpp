/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "distributed/nvidia/nccl_data_channel.h"

#include <nccl.h>
#include <cstring>
#include <ios>

#include "configure/context.h"
#include "utils/singleton.h"

#include "profiler/profile_event.h"

namespace ksana_llm {

NcclDataChannel::NcclDataChannel(std::shared_ptr<Context> context) { context_ = context; }

NcclDataChannel::~NcclDataChannel() { terminated_ = true; }

Status NcclDataChannel::InitializeNcclCluster(ncclUniqueId nccl_unique_id) {
  nccl_unique_id_ = nccl_unique_id;
  device_id_ = context_->device->GetDeviceId();

  int pp_size = context_->global->GetWorldSize();
  int node_rank = context_->global->GetNodeRank();
  int tp_size = context_->global->GetTensorParallelSize();
  KLLM_LOG_INFO << "Initialize nccl communicators, pp_size:" << pp_size << ", node_rank:" << node_rank
                << ", tp_size:" << tp_size << ", device_id:" << device_id_;

  int cur_dev_rank = node_rank * tp_size + device_id_;
  KLLM_LOG_INFO << "Initialize nccl communicator, dev_id:" << device_id_ << ", cur rank:" << cur_dev_rank;
  NCCL_CHECK(ncclCommInitRank(&communicator_, pp_size * tp_size, nccl_unique_id_, cur_dev_rank));

  // Build upstream and downstream node ranks.
  int upstream_node_rank = node_rank - 1;
  if (upstream_node_rank < 0) {
    upstream_node_rank = -1;
  }

  int downstream_node_rank = node_rank + 1;
  if (downstream_node_rank > pp_size - 1) {
    downstream_node_rank = -1;
  }

  // Build upstream and downstream device ranks.
  // NOTE: device ranks contain all devices of current node, used for for nccl scatter mode.
  for (int dev_id = 0; dev_id < tp_size; ++dev_id) {
    upstream_dev_ranks_.push_back(upstream_node_rank * tp_size + dev_id);
    downstream_dev_ranks_.push_back(downstream_node_rank * tp_size + dev_id);
  }
  KLLM_LOG_INFO << "Initialize nccl upstream node:" << upstream_node_rank
                << ", upstream ranks:" << Vector2Str(upstream_dev_ranks_);
  KLLM_LOG_INFO << "Initialize nccl downstream node:" << downstream_node_rank
                << ", downstream ranks:" << Vector2Str(downstream_dev_ranks_);

  return Status();
}

Status NcclDataChannel::DestroyNcclCluster() {
  NCCL_CHECK(ncclCommDestroy(communicator_));
  return Status();
}

Status NcclDataChannel::GetNcclDataType(DataType dtype, ncclDataType_t& nccl_dtype) {
  switch (dtype) {
    case DataType::TYPE_BYTES:
    case DataType::TYPE_BOOL: {
      nccl_dtype = ncclDataType_t::ncclChar;
      return Status();
    }
    case DataType::TYPE_INT8: {
      nccl_dtype = ncclDataType_t::ncclInt8;
      return Status();
    }
    case DataType::TYPE_UINT8: {
      nccl_dtype = ncclDataType_t::ncclUint8;
      return Status();
    }
    case DataType::TYPE_UINT32: {
      nccl_dtype = ncclDataType_t::ncclUint32;
      return Status();
    }
    case DataType::TYPE_UINT64: {
      nccl_dtype = ncclDataType_t::ncclUint64;
      return Status();
    }
    case DataType::TYPE_INT32: {
      nccl_dtype = ncclDataType_t::ncclInt32;
      return Status();
    }
    case DataType::TYPE_INT64: {
      nccl_dtype = ncclDataType_t::ncclInt64;
      return Status();
    }
    case DataType::TYPE_BF16: {
      nccl_dtype = ncclDataType_t::ncclBfloat16;
      return Status();
    }
    case DataType::TYPE_FP16: {
      nccl_dtype = ncclDataType_t::ncclFloat16;
      return Status();
    }
    case DataType::TYPE_FP32: {
      nccl_dtype = ncclDataType_t::ncclFloat32;
      return Status();
    }
    case DataType::TYPE_FP64: {
      nccl_dtype = ncclDataType_t::ncclFloat64;
      return Status();
    }
    default: {
      return Status(RET_INVALID_ARGUMENT, FormatStr("Not supported dtype %d", dtype));
    }
  }
  return Status();
}

Status NcclDataChannel::SendData(HiddenUnitDeviceBuffer* hidden_unit_dev, const std::vector<size_t>& shape,
                                 DataType data_type) {
  Tensor& tensor = hidden_unit_dev->tensor;

  ncclDataType_t nccl_dtype;
  GetNcclDataType(data_type, nccl_dtype);

  int64_t count = 1;
  for (auto i : shape) {
    count *= i;
  }

  if (comm_type_ == DistributedCommunicationType::SCATTER) {
    if (device_id_ != 0) {
      return Status();
    }

    ncclGroupStart();
    for (size_t dev_id = 0; dev_id < context_->global->GetTensorParallelSize(); ++dev_id) {
      NCCL_CHECK(ncclSend(tensor.GetPtr<void>(), count, nccl_dtype, downstream_dev_ranks_[dev_id], communicator_,
                          context_->device->GetDistributedSendCommStream().Get()));
    }
    ncclGroupEnd();
  } else {
    NCCL_CHECK(ncclSend(tensor.GetPtr<void>(), count, nccl_dtype, downstream_dev_ranks_[device_id_], communicator_,
                        context_->device->GetDistributedSendCommStream().Get()));
  }
  StreamSynchronize(context_->device->GetDistributedSendCommStream());
  return Status();
}

Status NcclDataChannel::RecvData(HiddenUnitDeviceBuffer* hidden_unit_dev, const std::vector<size_t>& shape,
                                 DataType data_type) {
  Tensor& tensor = hidden_unit_dev->tensor;
  tensor.shape = shape;
  tensor.dtype = data_type;

  ncclDataType_t nccl_dtype;
  GetNcclDataType(data_type, nccl_dtype);

  int64_t count = 1;
  for (auto i : shape) {
    count *= i;
  }

  int upstream_dev_rank =
      comm_type_ == DistributedCommunicationType::SCATTER ? upstream_dev_ranks_[0] : upstream_dev_ranks_[device_id_];

  NCCL_CHECK(ncclRecv(tensor.GetPtr<void>(), count, nccl_dtype, upstream_dev_rank, communicator_,
                      context_->device->GetDistributedRecvCommStream().Get()));

  // Reset tensor shape.
  tensor.shape = shape;
  tensor.dtype = data_type;

  StreamSynchronize(context_->device->GetDistributedRecvCommStream());

  return Status();
}

}  // namespace ksana_llm
