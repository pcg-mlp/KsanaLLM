# Copyright 2025 Tencent Inc.  All rights reserved.

file(GLOB_RECURSE FUSED_MOE_ALL_FILES
  fused_moe_creator.sh
  fused_moe.py
)

ksana_triton_kernel_md5_file(fused_moe FUSED_MOE_MD5_FILE)
set(FUSED_MOE_NEED_REGENERATE TRUE)

if(DEFINED ENV{DISABLE_GENERATE_TRITON_KERNEL})
  set(FUSED_MOE_NEED_REGENERATE FALSE)
  message(STATUS "DISABLE_GENERATE_TRITON_KERNEL is set, skipping fused_moe kernel generation")
endif()

if(EXISTS ${FUSED_MOE_MD5_FILE})
  file(READ ${FUSED_MOE_MD5_FILE} PREV_MD5)

  calculate_files_md5("${FUSED_MOE_ALL_FILES}" CURRENT_MD5)

  if(${PREV_MD5} STREQUAL ${CURRENT_MD5})
    set(FUSED_MOE_NEED_REGENERATE FALSE)
    message(STATUS "Files MD5 unchanged, skipping regeneration")
  else()
    message(STATUS "Files MD5 changed, regenerating")
    message(STATUS "Previous MD5: ${PREV_MD5}")
    message(STATUS "Current MD5: ${CURRENT_MD5}")
  endif()
endif()

if(FUSED_MOE_NEED_REGENERATE)
  message(STATUS "Queue generate fused_moe triton kernel")
  calculate_files_md5("${FUSED_MOE_ALL_FILES}" NEW_MD5)
  ksana_enqueue_triton_gen(
    NAME fused_moe
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    MD5_FILE ${FUSED_MOE_MD5_FILE}
    MD5_VALUE ${NEW_MD5}
    COMMAND bash fused_moe_creator.sh ${CMAKE_CURRENT_SOURCE_DIR} ${TRITON_KERNEL_PATH})
endif()

# for test
file(GLOB_RECURSE fused_moe_test_SRCS
${PROJECT_SOURCE_DIR}/csrc/kernels/nvidia/moe/fused_moe/*test.cpp)
# copy python script for make test
execute_process(COMMAND cp -r ${PROJECT_SOURCE_DIR}/csrc/kernels/nvidia/moe/fused_moe/fused_moe.py ${CMAKE_CURRENT_BINARY_DIR})
execute_process(COMMAND cp -r ${PROJECT_SOURCE_DIR}/csrc/kernels/nvidia/moe/fused_moe/fused_moe.py ${CMAKE_BINARY_DIR})
execute_process(COMMAND cp -r ${PROJECT_SOURCE_DIR}/csrc/kernels/nvidia/moe/common.py ${CMAKE_CURRENT_BINARY_DIR})
execute_process(COMMAND cp -r ${PROJECT_SOURCE_DIR}/csrc/kernels/nvidia/moe/common.py ${CMAKE_BINARY_DIR})
cc_test(fused_moe_config_find_test SRCS ${fused_moe_test_SRCS})

# fused_moe_test.cpp depends on <nlohmann/json.hpp>. The header is fetched by
# csrc/CMakeLists.txt (include external/nlohmann_json) into ${THIRD_PARTY_PATH}/install/nlohmann_json.
# However, that include_directories() call is directory-scoped to csrc/ and does not
# propagate here. Add the include path explicitly for this test target.
# Guard against WITH_TESTING=OFF (cc_test macro skips add_executable when testing is disabled).
if(TARGET fused_moe_config_find_test)
  target_include_directories(fused_moe_config_find_test PRIVATE
    ${THIRD_PARTY_PATH}/install/nlohmann_json/single_include)
endif()

