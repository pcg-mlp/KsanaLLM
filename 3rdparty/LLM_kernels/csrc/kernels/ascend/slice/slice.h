/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <unordered_map>
#include "acl/acl.h"
#include "acl/acl_op_compiler.h"
#include "aclnn/acl_meta.h"

#include "csrc/kernels/ascend/slice/slice_tiling.h"

#ifdef ENABLE_ACL_ATB
#  include "atb/atb_infer.h"
#endif

namespace llm_kernels {
namespace ascend {

void Slice(const aclTensor* input, const int sliceDim, const int sliceStart, const int sliceEnd, const int sliceStep,
           aclTensor** output, aclrtStream& stream, void (*ws_func)(size_t, void**));

template <typename T>
class Slice2 {
 public:
  Slice2();
  ~Slice2();

  // Slice fragments from input, and concat it to output.
  void Forward(void* output, void* input, void* tiling, int block_dim, aclrtStream stream);
  void Forward(void* output, void* input, int start, int length, int step, int times, aclrtStream stream);

  // Cache tiling
  void CacheTiling(void* dev, size_t key, int start, int length, int step, int times, aclrtStream stream);

  // Return device pointer of tiling struct.
  void* GetTilingData(size_t key, int& block_dim);

  size_t GetTilingSize() const { return tiling_size_; }

 private:
  // Copy the tiling data from host to global memory.
  void CopyTilingToDevice(aclrtStream stream);

  // Generate tiling for input shape and indexes.
  void GenerateTiling(int start, int length, int step, int times, SliceTilingData& tiling_data);

  // The tiling data for current request.
  SliceTilingData tiling_data_;

  // The tiling buffer on global memory
  void* tiling_buffer_gm_;

  // The size of tiling data.
  size_t tiling_size_;

  // The tiling cache
  std::unordered_map<size_t, void*> tiling_cache_;

  // The tiling used cores.
  std::unordered_map<size_t, int> tiling_cores_;
};

#ifdef ENABLE_ACL_ATB
void CreateSplitQKVATBOperation(const uint32_t total_token_num, const uint32_t head_size, const uint32_t kv_head_size,
                                const uint32_t head_dim, atb::Operation** operation, const std::string& op_name = "");
#endif

}  // namespace ascend
}  // namespace llm_kernels
