/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/utils/tensor.h"

#include <numeric>

namespace ksana_llm {

Status DestroyTensor(Tensor& tensor, const int rank) {
  if (tensor.block_id < 0) {
    return Status();
  }

  Status status;
  GetBlockManager()->SetDeviceId(rank);
  if (GetBlockManager()->IsContiguousUsed(tensor.block_id)) {
    status = GetBlockManager()->FreeContiguous(tensor.block_id);
  } else {
    status = Status();
  }

  tensor.block_id = -1;
  return status;
}

Status CreateTensor(Tensor& tensor, const std::vector<size_t> shape, const DataType dtype, const int rank,
                    const MemoryDevice memory_device) {
  if (shape.empty()) {
    return Status();
  }
  size_t total_bytes = std::accumulate(shape.begin(), shape.end(), 1ul, std::multiplies<size_t>()) * GetTypeSize(dtype);

  int block_id;
  GetBlockManager()->SetDeviceId(rank);
  if (memory_device == MemoryDevice::MEMORY_DEVICE) {
    GetBlockManager()->AllocateContiguous(total_bytes, block_id);
  } else {
    GetBlockManager()->AllocateHostContiguous(total_bytes, block_id);
  }
  tensor = Tensor(memory_device, dtype, shape, block_id);
  return Status();
}

}  // namespace ksana_llm
