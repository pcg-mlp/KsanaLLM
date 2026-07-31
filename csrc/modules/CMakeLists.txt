# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.13)

file(GLOB_RECURSE modules_SRCS
  ${PROJECT_SOURCE_DIR}/csrc/modules/*.cpp
)

list(FILTER modules_SRCS EXCLUDE REGEX ".*test.cpp")

set(kernels_nvidia_LIBS "")
set(kernels_ascend_LIBS "")

if(WITH_CUDA)
  list(APPEND kernels_nvidia_LIBS llm_kernels_nvidia_kernel_asymmetric_gemm)
endif()

if(WITH_ACL)
 list(APPEND kernels_ascend_LIBS atb_plugin_operations)
endif()

add_library(modules STATIC ${modules_SRCS})
target_link_libraries(modules PUBLIC
  -Wl,--whole-archive
  layers ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} ${kernels_nvidia_LIBS}
  -Wl,--no-whole-archive)

# for test
if(WITH_STANDALONE_TEST)
  set(MODELS_TEST_DEPS modules models executor ipc distributed)
  set(MODEL_TEST_MAIN ${PROJECT_SOURCE_DIR}/tests/test.cpp)

  if(WITH_CUDA)
    set(MODELS_TEST_LIBS ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
    cpp_test(module_mla_test SRCS ${PROJECT_SOURCE_DIR}/csrc/modules/attention/multihead_latent_attention_test.cpp ${MODEL_TEST_MAIN} DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
    cpp_test(module_bmm_test SRCS ${PROJECT_SOURCE_DIR}/csrc/modules/basic/bmm_test.cpp ${MODEL_TEST_MAIN} DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
    cpp_test(module_layernorm_test SRCS ${PROJECT_SOURCE_DIR}/csrc/modules/basic/layernorm_test.cpp ${MODEL_TEST_MAIN} DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
    cpp_test(module_dspark_markov_head_test SRCS ${PROJECT_SOURCE_DIR}/csrc/modules/basic/dspark_markov_head_test.cpp ${MODEL_TEST_MAIN} DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
    if(ENABLE_DEEPSEEK_DEEPGEMM_FLAG)
      cpp_test(module_dsv4_mqa_test SRCS ${PROJECT_SOURCE_DIR}/csrc/modules/attention/dsv4_mqa_test.cpp ${MODEL_TEST_MAIN} DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
      cpp_test(module_compressor_test SRCS ${PROJECT_SOURCE_DIR}/csrc/modules/attention/compressor_test.cpp ${MODEL_TEST_MAIN} DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
      cpp_test(module_mhc_test SRCS ${PROJECT_SOURCE_DIR}/csrc/modules/basic/mhc_test.cpp ${MODEL_TEST_MAIN} DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
      cpp_test(module_attn_res_test SRCS ${PROJECT_SOURCE_DIR}/csrc/modules/basic/attn_res_test.cpp ${MODEL_TEST_MAIN} DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
    endif()
  endif()
endif()
