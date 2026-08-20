# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.13)

set(nvidia_kernels_LIBS "")
set(nvidia_kernels_SRCS "")

if(WITH_CUDA)
  add_subdirectory(nvidia)

  file(GLOB_RECURSE nvidia_kernels_SRCS
    ${PROJECT_SOURCE_DIR}/csrc/kernels/nvidia/*.cpp)
  list(FILTER nvidia_kernels_SRCS EXCLUDE REGEX ".*test.cpp")

  list(APPEND nvidia_kernels_LIBS -lcudart llm_kernels_nvidia_kernel_paged_attention
    llm_kernels_nvidia_kernel_embedding llm_kernels_nvidia_kernel_tokenweave
    llm_kernels_nvidia_kernel_layernorm llm_kernels_nvidia_kernel_gemm_wrapper
    llm_kernels_nvidia_kernel_add llm_kernels_nvidia_kernel_add_mul llm_kernels_nvidia_kernel_activation
    llm_kernels_nvidia_kernel_assemble_tokens_hidden llm_kernels_nvidia_kernel_cast
    llm_kernels_nvidia_kernel_concat llm_kernels_nvidia_kernel_identity
    llm_kernels_nvidia_kernel_rotary_embedding llm_kernels_nvidia_kernel_all_reduce
    llm_kernels_nvidia_kernel_permute llm_kernels_nvidia_kernel_alibi
    llm_kernels_nvidia_kernel_samplers llm_kernels_nvidia_kernel_asymmetric_gemm
    llm_kernels_nvidia_kernel_gptq_marlin llm_kernels_nvidia_kernel_weight_only_batched_gemv
    llm_kernels_nvidia_kernel_mixture_of_experts llm_kernels_nvidia_kernel_moe_utils
    llm_kernels_nvidia_kernel_blockwise_gemm llm_kernels_nvidia_utils
    llm_kernels_nvidia_kernel_flash_mla llm_kernels_nvidia_kernel_expand
    llm_kernels_nvidia_kernel_flash_sparse_mla llm_kernels_nvidia_kernel_gather
    llm_kernels_nvidia_kernel_grouped_topk llm_kernels_nvidia_kernel_vllm_topk_softplus_sqrt
    llm_kernels_nvidia_kernel_sglang_topk_softplus_sqrt llm_kernels_nvidia_kernel_machete
    llm_kernels_nvidia_kernel_moe_wna16 llm_kernels_nvidia_kernel_dequant llm_kernels_nvidia_kernel_dequant_fp4
    llm_kernels_nvidia_kernel_marlin_moe llm_kernels_nvidia_kernel_trt_all_reduce
    llm_kernels_nvidia_kernel_quant_fp8 llm_kernels_nvidia_kernel_fused_add_norm
    llm_kernels_nvidia_kernel_split llm_kernels_nvidia_kernel_sglang_elementwise
    llm_kernels_nvidia_kernel_sglang_deepseek_v32 llm_kernels_nvidia_kernel_sglang_deepseek_v4
    llm_kernels_nvidia_kernel_sglang_kimi_k3 llm_kernels_nvidia_kernel_sglang_topk
    llm_kernels_nvidia_kernel_vllm_deepseek_v4 llm_kernels_nvidia_kernel_mhc
    llm_kernels_nvidia_kernel_adjust_mem llm_kernels_nvidia_kernel_gemm
    llm_kernels_nvidia_kernel_moe_cutlass_moe llm_kernels_nvidia_kernel_moe_expert_map
    llm_kernels_nvidia_kernel_moe_per_tensor_quant_by_scale llm_kernels_nvidia_kernel_attention_meta
    llm_kernels_nvidia_kernel_dspark
    llm_kernels_nvidia_kernel_flashinfer_attention
    llm_kernels_nvidia_kernel_sglang_marlin_moe_wna16
    llm_kernels_nvidia_kernel_qknorm_rope
    llm_kernels_nvidia_kernel_sglang_route_radix
    llm_kernels_nvidia_kernel_vllm_single_group_topk
    llm_kernels_nvidia_kernel_hpc_attention
    llm_kernels_nvidia_kernel_hpc_moe)

  if(TARGET llm_kernels_nvidia_kernel_humming)
    list(APPEND nvidia_kernels_LIBS llm_kernels_nvidia_kernel_humming)
  endif()
endif()


message(STATUS "nvidia_kernels_SRCS: ${nvidia_kernels_SRCS}")

set(ascend_kernels_LIBS "")
set(ascend_kernels_SRCS "")
set(ascend_kernels_test_SRCS "")

if(WITH_ACL)
  file(GLOB_RECURSE ascend_kernels_SRCS
    ${PROJECT_SOURCE_DIR}/csrc/kernels/ascend/*.cpp)
  list(FILTER ascend_kernels_SRCS EXCLUDE REGEX ".*test.cpp")
  list(APPEND ascend_kernels_LIBS ${ACL_SHARED_LIBS} atb_plugin_operations)

  file(GLOB_RECURSE ascend_kernels_test_SRCS ${PROJECT_SOURCE_DIR}/csrc/kernels/ascend/*test.cpp)
endif()

message(STATUS "ascend_kernels_SRCS: ${ascend_kernels_SRCS}")

set(zixiao_kernels_LIBS "")
set(zixiao_kernels_SRCS "")
set(zixiao_kernels_test_SRCS "")

if(WITH_TOPS)
  file(GLOB_RECURSE zixiao_kernels_SRCS
    ${PROJECT_SOURCE_DIR}/csrc/kernels/zixiao/*.cpp)
  list(FILTER zixiao_kernels_SRCS EXCLUDE REGEX ".*test.cpp")
endif()

# NOTE(fangtang, kunlun-adapt): Kunlun XPU kernels.
# Three compilation paths:
#   1. *.cpp (excluding *_launcher.cpp) → g++ (normal C++ wrappers)
#   2. *.xpu → xpu-clang --xpu-arch=xpu3 (device kernels)
#   3. *_launcher.cpp → xpu-clang -x xpu --xpu-host-only (host launch with <<<>>>)
set(kunlun_kernels_LIBS "")
set(kunlun_kernels_SRCS "")
set(kunlun_kernels_test_SRCS "")
set(kunlun_xpu_OBJS "")

if(WITH_KUNLUN)
  # --- 1. 普通 .cpp wrapper (g++ 编译) ---
  file(GLOB_RECURSE kunlun_kernels_SRCS
    ${PROJECT_SOURCE_DIR}/csrc/kernels/kunlun/*.cpp)
  list(FILTER kunlun_kernels_SRCS EXCLUDE REGEX ".*test.cpp")
  list(FILTER kunlun_kernels_SRCS EXCLUDE REGEX ".*_launcher\\.cpp")

  # --- 2. .xpu device kernel (xpu-clang) ---
  if(DEFINED ENV{XTDK_DIR} AND EXISTS "$ENV{XTDK_DIR}/bin/clang")
    set(XPU_CLANG "$ENV{XTDK_DIR}/bin/clang")
    message(STATUS "Kunlun: XPU_CLANG = ${XPU_CLANG}")

    file(GLOB kunlun_xpu_SRCS ${PROJECT_SOURCE_DIR}/csrc/kernels/kunlun/*.xpu)
    foreach(xpu_src ${kunlun_xpu_SRCS})
      get_filename_component(kernel_name ${xpu_src} NAME_WE)
      set(_obj ${CMAKE_CURRENT_BINARY_DIR}/kunlun_${kernel_name}.device.o)
      add_custom_command(OUTPUT ${_obj}
        COMMAND ${XPU_CLANG} -std=c++17 -O2
          --target=x86_64-baidu-linux-gnu --xpu-arch=xpu3 -fPIC
          -c ${xpu_src} -o ${_obj}
        DEPENDS ${xpu_src}
        COMMENT "XPU-CLANG: ${kernel_name}.xpu")
      list(APPEND kunlun_xpu_OBJS ${_obj})
    endforeach()

    # --- 3. launcher .cpp (含 <<<>>>, xpu-clang -x xpu --xpu-host-only) ---
    file(GLOB kunlun_launcher_SRCS ${PROJECT_SOURCE_DIR}/csrc/kernels/kunlun/*_launcher.cpp)
    foreach(launcher_src ${kunlun_launcher_SRCS})
      get_filename_component(launcher_name ${launcher_src} NAME_WE)
      set(_obj ${CMAKE_CURRENT_BINARY_DIR}/kunlun_${launcher_name}.o)
      add_custom_command(OUTPUT ${_obj}
        COMMAND ${XPU_CLANG} -std=c++17 -O2
          -x xpu --xpu-host-only -fxpu-launch-return
          --target=x86_64-baidu-linux-gnu -fPIC
          -I${XMLIR_DIR}/xre/include
          -I${PROJECT_SOURCE_DIR}/csrc
          -c ${launcher_src} -o ${_obj}
        DEPENDS ${launcher_src}
        COMMENT "XPU-CLANG (launcher): ${launcher_name}")
      list(APPEND kunlun_xpu_OBJS ${_obj})
    endforeach()

    # --- 打包成 .a ---
    if(kunlun_xpu_OBJS)
      set(_kunlun_ar ${CMAKE_CURRENT_BINARY_DIR}/libkunlun_xpu_kernels.a)
      add_custom_command(OUTPUT ${_kunlun_ar}
        COMMAND ${CMAKE_AR} rcs ${_kunlun_ar} ${kunlun_xpu_OBJS}
        DEPENDS ${kunlun_xpu_OBJS}
        COMMENT "AR: libkunlun_xpu_kernels.a")
      add_custom_target(kunlun_xpu_ar DEPENDS ${_kunlun_ar})
      add_library(kunlun_xpu_kernels STATIC IMPORTED GLOBAL)
      set_target_properties(kunlun_xpu_kernels PROPERTIES IMPORTED_LOCATION ${_kunlun_ar})
      add_dependencies(kunlun_xpu_kernels kunlun_xpu_ar)
      list(APPEND kunlun_kernels_LIBS kunlun_xpu_kernels)
    endif()
  else()
    message(STATUS "Kunlun: XTDK_DIR not set or xpu-clang not found, skipping .xpu kernel compilation")
  endif()
endif()

message(STATUS "kunlun_kernels_SRCS: ${kunlun_kernels_SRCS}")

add_library(kernels STATIC
  ${nvidia_kernels_SRCS}
  ${ascend_kernels_SRCS}
  ${zixiao_kernels_SRCS}
  ${kunlun_kernels_SRCS})
target_link_libraries(kernels PUBLIC utils cache_manager ${Python3_LIBRARIES}
  ${nvidia_kernels_LIBS} ${ascend_kernels_LIBS} ${kunlun_kernels_LIBS})


if(WITH_ACL AND WITH_STANDALONE_TEST)
  message(STATUS "ascend_kernels_test_SRCS: ${ascend_kernels_test_SRCS}")
  cpp_test(kernels_test SRCS ${ascend_kernels_test_SRCS} DEPS kernels)
endif()

# Kunlun kernel wrapper unit tests.
if(WITH_KUNLUN AND WITH_TESTING)
  file(GLOB kunlun_kernels_test_SRCS ${PROJECT_SOURCE_DIR}/csrc/kernels/kunlun/*test.cpp)
  if(kunlun_kernels_test_SRCS)
    add_executable(kunlun_kernel_test ${kunlun_kernels_test_SRCS})
    set(_KUNLUN_RUNTIME_RPATH
      "${_TORCH_LIB_DIR};${XMLIR_DIR};${XMLIR_DIR}/xre/so;${XMLIR_DIR}/xhpc/xblas/dependency_so")
    # libbkcl.so lives in ${XMLIR_DIR}, but libxpuml/libxpucuda are in xre/so
    # and sometimes xhpc/xblas/dependency_so; make both direct search entries.
    target_link_options(kunlun_kernel_test PRIVATE
      -Wl,--disable-new-dtags
      -Wl,-rpath,${XMLIR_DIR}/xre/so
      -Wl,-rpath,${XMLIR_DIR}/xhpc/xblas/dependency_so)
    target_include_directories(kunlun_kernel_test PRIVATE
      ${PROJECT_SOURCE_DIR}/tests
      ${gtest_SOURCE_DIR}/include
      ${gmock_SOURCE_DIR}/include)
    target_link_libraries(kunlun_kernel_test
      -Wl,--start-group
      kernels device utils cache_manager
      loguru fmt
      gtest gmock
      -Wl,--end-group
      -pthread
      -L${_TORCH_LIB_DIR} -ltorch -ltorch_cpu -lc10
      -L${XMLIR_DIR} -L${XMLIR_DIR}/xre/so
      -lxpu_dnn -lxpuapi -lbkcl -lxpurt -lcudart
      -Wl,--no-as-needed -lxpuml -lxpucuda -Wl,--as-needed)
    # libtorch_cpu.so has a versioned DT_NEEDED for libcupti.so.11.8. The file
    # at /usr/local/cuda*/extras/CUPTI has SONAME libcupti.so.11.7 (mismatch).
    # The copy under the nvidia conda package has the correct SONAME 11.8.
    # find_library can't locate it (no .so symlink); search by filename instead.
    find_file(_CUPTI_LIB_FILE libcupti.so.11.8
      HINTS $ENV{CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cuda_cupti/lib
            /data/miniconda3/envs/env-3.10/lib/python3.10/site-packages/nvidia/cuda_cupti/lib
      NO_DEFAULT_PATH)
    if(_CUPTI_LIB_FILE)
      get_filename_component(_CUPTI_LIB_DIR "${_CUPTI_LIB_FILE}" DIRECTORY)
    else()
      set(_CUPTI_LIB_DIR "/data/miniconda3/envs/env-3.10/lib/python3.10/site-packages/nvidia/cuda_cupti/lib")
    endif()
    target_link_directories(kunlun_kernel_test PRIVATE ${XPU_LIB_DIRS} ${_TORCH_LIB_DIR} ${_CUPTI_LIB_DIR})
    set_target_properties(kunlun_kernel_test PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
      BUILD_RPATH "${_KUNLUN_RUNTIME_RPATH}")
  endif()

  # Kunlun xpuml memory query unit test (device-level).
  set(_xpuml_mem_test_src ${PROJECT_SOURCE_DIR}/csrc/device/kunlun/kunlun_xpuml_mem_test.cpp)
  if(EXISTS ${_xpuml_mem_test_src})
    add_executable(kunlun_xpuml_mem_test ${_xpuml_mem_test_src})
    target_include_directories(kunlun_xpuml_mem_test PRIVATE
      ${PROJECT_SOURCE_DIR}/tests
      ${gtest_SOURCE_DIR}/include
      ${gmock_SOURCE_DIR}/include)
    target_link_libraries(kunlun_xpuml_mem_test
      -Wl,--start-group
      device utils
      loguru fmt
      gtest gmock
      -Wl,--end-group
      -pthread
      -L${XMLIR_DIR}/xre/so -lxpurt -lxpuml)
    target_link_directories(kunlun_xpuml_mem_test PRIVATE ${XPU_LIB_DIRS})
    target_link_options(kunlun_xpuml_mem_test PRIVATE
      -Wl,--disable-new-dtags
      -Wl,-rpath,${XMLIR_DIR}/xre/so)
    set_target_properties(kunlun_xpuml_mem_test PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
      BUILD_RPATH "${_KUNLUN_RUNTIME_RPATH}")
  endif()
endif()
