# Copyright 2026 Tencent Inc. All rights reserved.
#
# Humming JIT tests.

file(GLOB HUMMING_TEST_SRCS *.cu)
if(HUMMING_TEST_SRCS)
  cc_test(llm_kernels_nvidia_kernel_humming_test
    SRCS ${HUMMING_TEST_SRCS}
    DEPS llm_kernels_nvidia_utils
         llm_kernels_nvidia_kernel_humming_jit
         llm_kernels_nvidia_kernel_humming_launcher
         llm_kernels_nvidia_kernel_humming_ops)
endif()
