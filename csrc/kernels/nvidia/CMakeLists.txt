
file(GLOB_RECURSE TRITON_SRCS *.cpp)
list(FILTER TRITON_SRCS EXCLUDE REGEX ".*test.cpp")

find_package(CUDAToolkit REQUIRED)

add_library(llm_kernels_nvidia_kernel_triton STATIC ${TRITON_SRCS})
target_link_libraries(llm_kernels_nvidia_kernel_triton PUBLIC utils ${TORCH_LIBRARIES} -lcuda -lcudart ksana_llm_libs
    llm_kernels_nvidia_utils llm_kernels_nvidia_kernel_moe_utils
    ${CUDA_LIBRARIES})

# file(GLOB_RECURSE KERNEL_WRAPPER_TEST_SRCS kernel_wrapper_test.cpp)
# cc_test(kernel_wrapper_test SRCS ${KERNEL_WRAPPER_TEST_SRCS} DEPS
#     llm_kernels_nvidia_utils llm_kernels_nvidia_kernel_triton LINKS ${TORCH_LIBRARIES})

# file(GLOB_RECURSE GROUPED_TOPK_TEST_SRCS grouped_topk_test.cpp)
# cc_test(grouped_topk_test SRCS ${GROUPED_TOPK_TEST_SRCS} DEPS
#     llm_kernels_nvidia_utils llm_kernels_nvidia_kernel_triton LINKS ${TORCH_LIBRARIES})

file(GLOB_RECURSE FUSED_MOE_TEST_SRCS fused_moe_test.cpp)
cc_test(fused_moe_test SRCS ${FUSED_MOE_TEST_SRCS} DEPS
    llm_kernels_nvidia_utils llm_kernels_nvidia_kernel_triton)

# file(GLOB_RECURSE FUSED_MOE_GPTQ_AWQ_TEST_SRCS fused_moe_gptq_awq_test.cpp)
# cc_test(fused_moe_gptq_awq_test SRCS ${FUSED_MOE_GPTQ_AWQ_TEST_SRCS} DEPS
#     llm_kernels_nvidia_utils llm_kernels_nvidia_kernel_triton)

set(HUMMING_FP8_QUANT_TEST_SRCS humming_fp8_quant_test.cpp)
cc_test(humming_fp8_quant_test SRCS ${HUMMING_FP8_QUANT_TEST_SRCS} DEPS
    llm_kernels_nvidia_utils llm_kernels_nvidia_kernel_triton)

set(MOE_SUM_DISPATCH_TEST_SRCS moe_sum_dispatch_test.cpp)
cc_test(moe_sum_dispatch_test SRCS ${MOE_SUM_DISPATCH_TEST_SRCS} DEPS
    llm_kernels_nvidia_utils llm_kernels_nvidia_kernel_triton)

# if (SM STREQUAL "90" OR SM STREQUAL "90a")
#     file(GLOB_RECURSE FUSED_MOE_GPTQ_INT4_FP8_TEST_SRCS fused_moe_gptq_int4_fp8_test.cpp)
#     cc_test(fused_moe_gptq_int4_fp8_test SRCS ${FUSED_MOE_GPTQ_INT4_FP8_TEST_SRCS} DEPS
#         llm_kernels_nvidia_utils llm_kernels_nvidia_kernel_triton)
# endif()

# set(FLASH_ATTN_CPP_WRAPPER_TEST_SRCS flash_attn_cpp_wrapper_test.cpp)
# cc_test(flash_attn_cpp_wrapper_test SRCS ${FLASH_ATTN_CPP_WRAPPER_TEST_SRCS} DEPS
#     llm_kernels_nvidia_utils llm_kernels_nvidia_kernel_triton LINKS ${TORCH_LIBRARIES})

set(FUSED_MOE_TUNED_CONFIG_TEST_SRCS fused_moe_tuned_config_test.cpp)
cc_test(fused_moe_tuned_config_test SRCS ${FUSED_MOE_TUNED_CONFIG_TEST_SRCS} DEPS
    llm_kernels_nvidia_kernel_triton LINKS ${TORCH_LIBRARIES})

set(TRITON_WRAPPER_TEST_SRCS triton_wrapper_test.cpp)
cc_test(triton_wrapper_test SRCS ${TRITON_WRAPPER_TEST_SRCS} DEPS
    llm_kernels_nvidia_utils llm_kernels_nvidia_kernel_triton LINKS ${TORCH_LIBRARIES})

if(HAS_SM90 OR HAS_SM90A)
    set(KDA_KERNEL_WRAPPER_TEST_SRCS kda_kernel_wrapper_test.cpp)
    cc_test(kda_kernel_wrapper_test SRCS ${KDA_KERNEL_WRAPPER_TEST_SRCS} DEPS
        llm_kernels_nvidia_utils llm_kernels_nvidia_kernel_triton LINKS ${TORCH_LIBRARIES})
endif()

if(ENABLE_DEEPSEEK_DEEPGEMM_FLAG)
    file(GLOB_RECURSE DEEPSEEK_DEEPGEMM_WRAPPER_TEST_SRCS deepseek_deepgemm_wrapper_test.cpp)
    cc_test(deepseek_deepgemm_wrapper_test SRCS ${DEEPSEEK_DEEPGEMM_WRAPPER_TEST_SRCS} DEPS
        llm_kernels_nvidia_utils llm_kernels_nvidia_kernel_triton LINKS ${TORCH_LIBRARIES})

    set(FUSED_BATCHED_MOE_TEST_SRCS fused_batched_moe_test.cpp)
    cc_test(fused_batched_moe_test SRCS ${FUSED_BATCHED_MOE_TEST_SRCS} DEPS
        llm_kernels_nvidia_utils llm_kernels_nvidia_kernel_triton LINKS ${TORCH_LIBRARIES})

    set(LOW_LATENCY_ADAPTER_TEST_SRCS low_latency_adapter_test.cpp)
    cc_test(low_latency_adapter_test SRCS ${LOW_LATENCY_ADAPTER_TEST_SRCS} DEPS
        llm_kernels_nvidia_utils llm_kernels_nvidia_kernel_triton llm_kernels_nvidia_kernel_moe_cutlass_moe
        LINKS ${TORCH_LIBRARIES})
endif()
