# Copyright 2026 Tencent Inc.  All rights reserved.

# set kernels target
file(GLOB_RECURSE FLASH_SPARSE_MLA_SRCS *.cu)
list(FILTER FLASH_SPARSE_MLA_SRCS EXCLUDE REGEX ".*test.cu")

if(ENABLE_FLASH_MLA STREQUAL "TRUE")
  set(CUDA_FLASH_MLA_FLAGS "-gencode arch=compute_90a,code=sm_90a -DENABLE_FLASH_MLA")
  add_definitions(-DENABLE_FLASH_MLA)
else()
  set(CUDA_FLASH_MLA_FLAGS "-gencode arch=compute_86,code=sm_86")
  list(FILTER FLASH_SPARSE_MLA_SRCS EXCLUDE REGEX ".*/flash_sparse_mla/kernels/.*")
endif()

# Create flash_sparse_mla library
add_library(llm_kernels_nvidia_kernel_flash_sparse_mla STATIC ${DEEPSEEK_FLASH_MLA_SOURCES} ${FLASH_SPARSE_MLA_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_flash_sparse_mla PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_flash_sparse_mla PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(llm_kernels_nvidia_kernel_flash_sparse_mla PROPERTIES LINKER_LANGUAGE CUDA)

# Define custom CUDA flags
set(CUSTOM_CUDA_FLAGS
"-O3 -std=c++20 -DNDEBUG -D_USE_MATH_DEFINES -Wno-deprecated-declarations \
-U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ \
-U__CUDA_NO_HALF2_OPERATORS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__ \
--expt-relaxed-constexpr --expt-extended-lambda --use_fast_math \
--ptxas-options=-v,--register-usage-level=10 \
${CUDA_FLASH_MLA_FLAGS} --threads 16")

# Convert the string to a list of arguments
separate_arguments(CUSTOM_CUDA_FLAGS_LIST UNIX_COMMAND "${CUSTOM_CUDA_FLAGS}")

target_compile_options(llm_kernels_nvidia_kernel_flash_sparse_mla PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:${CUSTOM_CUDA_FLAGS_LIST}>)

# get cutlass
get_target_property(CUTLASS_INCLUDE_DIRS cutlass_c3x INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(llm_kernels_nvidia_kernel_flash_sparse_mla PRIVATE
  ${CUTLASS_INCLUDE_DIRS}
  ${DEEPSEEK_FLASH_MLA_INCLUDES})

target_link_libraries(llm_kernels_nvidia_kernel_flash_sparse_mla PUBLIC -lcublas -lcudart -lcublasLt
  llm_kernels_nvidia_utils)
target_link_libraries(llm_kernels_nvidia_kernel_flash_sparse_mla PRIVATE cutlass_c3x ${TORCH_LIBRARIES})

# for test
file(GLOB_RECURSE FLASH_SPARSE_MLA_TEST_SRCS *test.cu)
if(FLASH_SPARSE_MLA_TEST_SRCS AND (ENABLE_FLASH_MLA STREQUAL "TRUE"))
  cc_test(llm_kernels_nvidia_kernel_flash_sparse_mla_test SRCS ${FLASH_SPARSE_MLA_TEST_SRCS} DEPS
    llm_kernels_nvidia_kernel_flash_sparse_mla)
  if(WITH_TESTING)
    target_include_directories(llm_kernels_nvidia_kernel_flash_sparse_mla_test PRIVATE
      ${CUTLASS_INCLUDE_DIRS}
      ${DEEPSEEK_FLASH_MLA_INCLUDES})
  endif()
endif()
