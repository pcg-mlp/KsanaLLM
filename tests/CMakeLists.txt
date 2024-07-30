# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

# copy python script for make test
execute_process(COMMAND cp -r ${PROJECT_SOURCE_DIR}/src/ksana_llm/python/ksana_llm ${CMAKE_CURRENT_BINARY_DIR})

# copy python script for standalone test
execute_process(COMMAND cp -r ${PROJECT_SOURCE_DIR}/src/ksana_llm/python/ksana_llm ${CMAKE_BINARY_DIR})

set(INTEGRATION_TEST_CONFIG "")
set(INTEGRATION_TEST_MODEL "")
set(INTEGRATION_TEST_CODE "")

if(WITH_CUDA)
  set(INTEGRATION_TEST_CONFIG ${PROJECT_SOURCE_DIR}/tests/model/llama_test.yaml)
  set(INTEGRATION_TEST_CONFIG ${PROJECT_SOURCE_DIR}/examples/ksana_llm.yaml)
  set(INTEGRATION_TEST_MODEL /model/qwen1.5-hf/0.5B-Chat)
  set(INTEGRATION_TEST_CODE ${PROJECT_SOURCE_DIR}/tests/model/qwen2_integration_test.py)
endif()

if(WITH_ACL)
  set(INTEGRATION_TEST_CONFIG ${PROJECT_SOURCE_DIR}/examples/ksana_llm_ascend_llama13b.yaml)
  set(INTEGRATION_TEST_MODEL /model/llama-hf/13B)
  set(INTEGRATION_TEST_CODE ${PROJECT_SOURCE_DIR}/tests/model/llama_integration_test.py)
endif()

if(NOT INTEGRATION_TEST_CONFIG STREQUAL "" AND WITH_STANDALONE_TEST)
  python_test(model_integration_test SRCS ${INTEGRATION_TEST_CODE}
    ARGS --config_file ${INTEGRATION_TEST_CONFIG}
    --tokenizer_dir ${INTEGRATION_TEST_MODEL})
endif()

