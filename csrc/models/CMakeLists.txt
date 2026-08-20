# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.13)

file(GLOB_RECURSE models_SRCS
  ${PROJECT_SOURCE_DIR}/csrc/models/common/*.cpp
  ${PROJECT_SOURCE_DIR}/csrc/models/communicator/*.cpp
  ${PROJECT_SOURCE_DIR}/csrc/models/draft_model/*.cpp
  ${PROJECT_SOURCE_DIR}/csrc/models/quant/*.cpp
  ${PROJECT_SOURCE_DIR}/csrc/models/weight_method/*.cpp
  ${PROJECT_SOURCE_DIR}/csrc/models/llama/*.cpp
  ${PROJECT_SOURCE_DIR}/csrc/models/qwen/*.cpp
  ${PROJECT_SOURCE_DIR}/csrc/models/bge_reranker_minicpm/*.cpp
  ${PROJECT_SOURCE_DIR}/csrc/models/chatglm/*.cpp
  ${PROJECT_SOURCE_DIR}/csrc/models/gpt_moe/*.cpp
  ${PROJECT_SOURCE_DIR}/csrc/models/common_moe/*.cpp
  ${PROJECT_SOURCE_DIR}/csrc/models/mixtral/*.cpp
  ${PROJECT_SOURCE_DIR}/csrc/models/qwen2_moe/*.cpp
  ${PROJECT_SOURCE_DIR}/csrc/models/qwen3_moe/*.cpp
  ${PROJECT_SOURCE_DIR}/csrc/models/internlm2/*.cpp
  ${PROJECT_SOURCE_DIR}/csrc/models/internlmxcomposer2/*.cpp
  ${PROJECT_SOURCE_DIR}/csrc/models/deepseek_v3/*.cpp
  ${PROJECT_SOURCE_DIR}/csrc/models/deepseek_v4/*.cpp
  ${PROJECT_SOURCE_DIR}/csrc/models/kimi_k3/*.cpp
  ${PROJECT_SOURCE_DIR}/csrc/models/glm4_moe/*.cpp
  ${PROJECT_SOURCE_DIR}/csrc/models/hunyuan_turbo/*.cpp
  ${PROJECT_SOURCE_DIR}/csrc/models/arc_hunyuan_video/*.cpp
  ${PROJECT_SOURCE_DIR}/csrc/models/deepseek_v4/*.cpp)

list(FILTER models_SRCS EXCLUDE REGEX ".*test.cpp")

set(kernels_nvidia_LIBS "")
set(kernels_ascend_LIBS "")

if(WITH_CUDA)
  list(APPEND kernels_nvidia_LIBS llm_kernels_nvidia_kernel_asymmetric_gemm
                                  llm_kernels_nvidia_kernel_gemm_finegrained_mixed_dtype_gemm
                                  llm_kernels_nvidia_kernel_pooling)
endif()

add_library(models STATIC ${models_SRCS})
target_link_libraries(models PUBLIC modules model_base cache_manager
  -Wl,--whole-archive
  layers
  -Wl,--no-whole-archive
  samplers runtime ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} ${kernels_nvidia_LIBS})

# for test
if(WITH_STANDALONE_TEST)
  set(MODELS_TEST_DEPS models runtime executor)
  set(MODEL_TEST_MAIN ${PROJECT_SOURCE_DIR}/tests/test.cpp)

  if(WITH_CUDA)
    set(MODELS_TEST_LIBS ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
  else()
    set(MODELS_TEST_LIBS "")
  endif()

  # 调用 cpp_test
  if(WITH_CUDA)
    cpp_test(fake_weight_test SRCS ${PROJECT_SOURCE_DIR}/csrc/models/base/fake_weight_test.cpp ${MODEL_TEST_MAIN} DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
  endif()
  # cpp_test(expert_para_weight_test SRCS ${PROJECT_SOURCE_DIR}/csrc/models/common_moe/common_moe_weight_test.cpp ${MODEL_TEST_MAIN} DEPS ${MODELS_TEST_DEPS}  LIBS ${MODELS_TEST_LIBS})
  # cpp_test(llama_test SRCS ${PROJECT_SOURCE_DIR}/csrc/models/llama/llama_test.cpp ${MODEL_TEST_MAIN} DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS} ${TORCH_LIBRARIES})
  # cpp_test(simple_decoder_layer_test SRCS ${PROJECT_SOURCE_DIR}/csrc/models/common/simple_decoder_layer_test.cpp ${MODEL_TEST_MAIN} DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
  if(WITH_CUDA)
    cpp_test(qwen_deepstack_test SRCS ${PROJECT_SOURCE_DIR}/csrc/models/qwen/qwen_deepstack_test.cpp ${MODEL_TEST_MAIN} DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
    cpp_test(qwen_embedding_test SRCS ${PROJECT_SOURCE_DIR}/csrc/models/qwen/qwen_embedding_test.cpp ${MODEL_TEST_MAIN} DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
    cpp_test(hunyuan_turbo_test SRCS ${PROJECT_SOURCE_DIR}/csrc/models/hunyuan_turbo/hunyuan_turbo_test.cpp ${MODEL_TEST_MAIN} DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
    cpp_test(chatglm_test SRCS ${PROJECT_SOURCE_DIR}/csrc/models/chatglm/chatglm_test.cpp ${MODEL_TEST_MAIN} DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
    if(DEFINED SM AND "${SM}" STREQUAL "90a")
      cpp_test(mixtral_test SRCS ${PROJECT_SOURCE_DIR}/csrc/models/mixtral/mixtral_test.cpp ${MODEL_TEST_MAIN} DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
      cpp_test(qwen_test SRCS ${PROJECT_SOURCE_DIR}/csrc/models/qwen/qwen_test.cpp ${MODEL_TEST_MAIN} DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
      cpp_test(qwen3_moe_test SRCS ${PROJECT_SOURCE_DIR}/csrc/models/qwen3_moe/qwen3_moe_test.cpp ${MODEL_TEST_MAIN} DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
      cpp_test(bge_reranker_minicpm_test SRCS ${PROJECT_SOURCE_DIR}/csrc/models/bge_reranker_minicpm/bge_reranker_minicpm_test.cpp ${MODEL_TEST_MAIN} DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
      cpp_test(gpt_moe_test SRCS ${PROJECT_SOURCE_DIR}/csrc/models/gpt_moe/gpt_moe_test.cpp ${MODEL_TEST_MAIN} DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
      cpp_test(internlm2_test SRCS ${PROJECT_SOURCE_DIR}/csrc/models/internlm2/internlm2_test.cpp ${MODEL_TEST_MAIN} DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
      cpp_test(internlmxcomposer2_test SRCS ${PROJECT_SOURCE_DIR}/csrc/models/internlmxcomposer2/internlmxcomposer2_test.cpp ${MODEL_TEST_MAIN} DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})

      cpp_test(deepseek_v3_test SRCS ${PROJECT_SOURCE_DIR}/csrc/models/deepseek_v3/deepseek_v3_model_test.cpp ${MODEL_TEST_MAIN} DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
      cpp_test(deepseek_v4_test SRCS ${PROJECT_SOURCE_DIR}/csrc/models/deepseek_v4/deepseek_v4_model_test.cpp ${MODEL_TEST_MAIN} DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
      cpp_test(deepseek_v4_dspark_test SRCS ${PROJECT_SOURCE_DIR}/csrc/models/deepseek_v4/deepseek_v4_dspark_test.cpp ${MODEL_TEST_MAIN} DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
      cpp_test(deepseek_v3_w4afp8_test SRCS ${PROJECT_SOURCE_DIR}/csrc/models/deepseek_v3/deepseek_v3_w4afp8_test.cpp ${MODEL_TEST_MAIN} DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
      cpp_test(deepseek_v4_weight_loader_test SRCS ${PROJECT_SOURCE_DIR}/csrc/models/deepseek_v4/deepseek_v4_weight_loader_test.cpp ${MODEL_TEST_MAIN} DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
      cpp_test(kimi_k3_weight_loader_test SRCS ${PROJECT_SOURCE_DIR}/csrc/models/kimi_k3/kimi_k3_weight_loader_test.cpp ${MODEL_TEST_MAIN} DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
      cpp_test(glm4_moe_test SRCS ${PROJECT_SOURCE_DIR}/csrc/models/glm4_moe/glm4_moe_test.cpp ${MODEL_TEST_MAIN} DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
      # cpp_test(deepseek_v3_dp_test SRCS ${PROJECT_SOURCE_DIR}/csrc/models/deepseek_v3/deepseek_v3_dp_test.cpp ${MODEL_TEST_MAIN} DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
      cpp_test(arc_hunyuan_video_test SRCS ${PROJECT_SOURCE_DIR}/csrc/models/arc_hunyuan_video/arc_hunyuan_video_test.cpp ${MODEL_TEST_MAIN} DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
      cpp_test(arc_hunyuan_video_config_parser_test SRCS ${PROJECT_SOURCE_DIR}/csrc/models/arc_hunyuan_video/arc_hunyuan_video_config_parser_test.cpp ${MODEL_TEST_MAIN} DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
      # cpp_test(quant_int4_weight_test SRCS ${PROJECT_SOURCE_DIR}/csrc/models/quant/quant_int4_weight_test.cpp ${MODEL_TEST_MAIN} DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
    endif()
  endif()
endif()

if(WITH_CUDA)
  file(GLOB_RECURSE quant_test_SRCS
    ${PROJECT_SOURCE_DIR}/csrc/models/quant/quant_fp8_weight_test.cpp)
  message(STATUS "quant_test_SRCS: ${quant_test_SRCS} ")

  if(WITH_STANDALONE_TEST)
    # cpp_test(cutlass_utils_test SRCS ${PROJECT_SOURCE_DIR}/csrc/models/quant/cutlass_utils_test.cpp DEPS models runtime LIBS ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
    # cpp_test(marlin_utils_test SRCS ${PROJECT_SOURCE_DIR}/csrc/models/quant/marlin_utils_test.cpp DEPS models runtime LIBS ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
    # cpp_test(quant_fp8_weight_test SRCS ${quant_test_SRCS} DEPS models runtime LIBS ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
    cpp_test(common_method_test SRCS ${PROJECT_SOURCE_DIR}/csrc/models/weight_method/common_method_test.cpp DEPS models runtime executor LIBS ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
    cpp_test(awq_method_test SRCS ${PROJECT_SOURCE_DIR}/csrc/models/weight_method/awq_method_test.cpp ${MODEL_TEST_MAIN} DEPS models runtime executor LIBS ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
    cpp_test(gptq_method_test SRCS ${PROJECT_SOURCE_DIR}/csrc/models/weight_method/gptq_method_test.cpp ${MODEL_TEST_MAIN} DEPS models runtime executor LIBS ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
    cpp_test(fp8_e4m3_method_test SRCS ${PROJECT_SOURCE_DIR}/csrc/models/weight_method/fp8_e4m3_method_test.cpp ${MODEL_TEST_MAIN} DEPS models runtime executor LIBS ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
    # if(${CUDA_VERSION} VERSION_GREATER_EQUAL "12.8")
    #   cpp_test(w4a8_awq_method_test SRCS ${PROJECT_SOURCE_DIR}/csrc/models/weight_method/w4a8_awq_method_test.cpp DEPS models runtime LIBS ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
    # endif()
  endif()
endif()
