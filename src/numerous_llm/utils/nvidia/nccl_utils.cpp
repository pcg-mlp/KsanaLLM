/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "src/numerous_llm/utils/nvidia/nccl_utils.h"

namespace numerous_llm {

ncclResult_t DestroyNCCLParam(NCCLParam& param) {
  ncclResult_t status = ncclSuccess;
  if (param.nccl_comm != nullptr) {
    status = ncclCommDestroy(param.nccl_comm);
    param.nccl_comm = nullptr;
  }
  return status;
}

ncclUniqueId GenerateNCCLUniqueID() {
  ncclUniqueId nccl_uid;
  NCCL_CHECK(ncclGetUniqueId(&nccl_uid));
  return nccl_uid;
}

ncclResult_t InitNCCLParam(NCCLParam& param, const int world_size, const int rank_id, const ncclUniqueId nccl_uid) {
  ncclResult_t status = ncclSuccess;
  if (world_size == 1) {
    param.world_size = 1;
    param.rank = 0;
    return status;
  }

  status = ncclCommInitRank(&(param.nccl_comm), world_size, nccl_uid, rank_id);
  param.rank = rank_id;
  param.nccl_uid = nccl_uid;
  param.world_size = world_size;

  return status;
}

}  // namespace numerous_llm