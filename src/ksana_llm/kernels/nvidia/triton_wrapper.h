/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <string>

#include "ksana_llm/utils/status.h"

namespace ksana_llm {

struct TritonKernel {
  int shm_size;
  int grid_x;
  int grid_y;
  int num_warps;
  std::string kernel_name;
  CUmodule module;
  CUfunction kernel;
};

CUresult LoadTritonKernelFromFile(const std::string& ptx_file_path, TritonKernel& triton_kernel);

CUresult InvokeTritonKernel(TritonKernel& triton_kernel, dim3& grid, dim3& block, void* args[], cudaStream_t stream,
                            void** extra_opt);

}  // namespace ksana_llm