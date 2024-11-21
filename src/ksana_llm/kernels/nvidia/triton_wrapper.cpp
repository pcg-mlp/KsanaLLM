/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/kernels/nvidia/triton_wrapper.h"

#include <filesystem>

#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/nvidia/cuda_utils.h"

namespace fs = std::filesystem;

namespace ksana_llm {

CUresult LoadTritonKernelFromFile(const std::string& file_path, TritonKernel& triton_kernel) {
  fs::path file_full_path = fs::absolute(fs::path(file_path));

  CUDA_CHECK_RETURN(cuModuleLoad(&triton_kernel.module, file_full_path.c_str()));
  CUDA_CHECK_RETURN(
      cuModuleGetFunction(&triton_kernel.kernel, triton_kernel.module, triton_kernel.kernel_name.c_str()));
  CUDA_CHECK_RETURN(cuFuncSetAttribute(triton_kernel.kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                       triton_kernel.shm_size));

  return CUDA_SUCCESS;
}

CUresult InvokeTritonKernel(TritonKernel& triton_kernel, dim3& grid, dim3& block, void* args[], cudaStream_t stream,
                            void** extra_opt) {
  CUDA_CHECK_RETURN(cuLaunchKernel(triton_kernel.kernel, grid.x, grid.y,
                                   grid.z,                     // Grid dimensions
                                   block.x, block.y, block.z,  // Block dimensions
                                   triton_kernel.shm_size,     // Shared memory size
                                   stream,                     // Stream
                                   args,                       // Kernel parameters
                                   nullptr));                  // Extra options
  return CUDA_SUCCESS;
}

}  // namespace ksana_llm