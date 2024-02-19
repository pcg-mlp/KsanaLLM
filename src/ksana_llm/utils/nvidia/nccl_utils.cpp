/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "src/ksana_llm/utils/nvidia/nccl_utils.h"

namespace ksana_llm {

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

}  // namespace ksana_llm