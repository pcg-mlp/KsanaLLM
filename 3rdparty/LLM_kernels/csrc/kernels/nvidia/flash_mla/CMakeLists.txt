# Copyright 2024 Tencent Inc.  All rights reserved.

# set kernels target
file(GLOB_RECURSE FLASH_MLA_SRCS *.cu)
list(FILTER FLASH_MLA_SRCS EXCLUDE REGEX ".*test.cu")

if(ENABLE_FLASH_MLA STREQUAL "TRUE")
  set(CUDA_FLASH_MLA_FLAGS "-DENABLE_FLASH_MLA")
  add_definitions(-DENABLE_FLASH_MLA)
  message(STATUS "Flash MLA enabled for SM90a GPUs")
else()
  # When flash_mla is not enabled, we still need to create the library target
  # but compile with a compatible SM arch to avoid errors
  set(CUDA_FLASH_MLA_FLAGS "")
  message(STATUS "Flash MLA disabled (requires SM90a)")
endif()

# Create flash_mla library
add_library(llm_kernels_nvidia_kernel_flash_mla STATIC ${FLASH_MLA_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_flash_mla PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_flash_mla PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(llm_kernels_nvidia_kernel_flash_mla PROPERTIES LINKER_LANGUAGE CUDA)

# Override CUDA_ARCHITECTURES for this target to avoid compiling with all global SMs
if(ENABLE_FLASH_MLA STREQUAL "TRUE")
  set_target_properties(llm_kernels_nvidia_kernel_flash_mla PROPERTIES CUDA_ARCHITECTURES "90a")
endif()

# Define custom CUDA flags
set(CUSTOM_CUDA_FLAGS
"-O3 -std=c++17 -DNDEBUG -D_USE_MATH_DEFINES -Wno-deprecated-declarations \
-U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ \
-U__CUDA_NO_HALF2_OPERATORS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__ \
--expt-relaxed-constexpr --expt-extended-lambda --use_fast_math \
--ptxas-options=-v,--register-usage-level=10 \
${CUDA_FLASH_MLA_FLAGS} --threads 16")

# Convert the string to a list of arguments
separate_arguments(CUSTOM_CUDA_FLAGS_LIST UNIX_COMMAND "${CUSTOM_CUDA_FLAGS}")

target_compile_options(llm_kernels_nvidia_kernel_flash_mla PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:${CUSTOM_CUDA_FLAGS_LIST}>)

# get cutlass
get_target_property(CUTLASS_INCLUDE_DIRS cutlass_c3x INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(llm_kernels_nvidia_kernel_flash_mla PRIVATE
  ${CUTLASS_INCLUDE_DIRS})

target_link_libraries(llm_kernels_nvidia_kernel_flash_mla PUBLIC -lcublas -lcudart -lcublasLt
  llm_kernels_nvidia_utils)
target_link_libraries(llm_kernels_nvidia_kernel_flash_mla PRIVATE cutlass_c3x)

# for test
file(GLOB_RECURSE FLASH_MLA_TEST_SRCS *test.cu)
if(FLASH_MLA_TEST_SRCS AND (ENABLE_FLASH_MLA STREQUAL "TRUE"))
  cc_test(llm_kernels_nvidia_kernel_flash_mla_test SRCS ${FLASH_MLA_TEST_SRCS} DEPS
    llm_kernels_nvidia_kernel_flash_mla)
endif()
