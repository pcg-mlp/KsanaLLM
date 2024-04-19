# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

# copy python script for make test
execute_process(COMMAND cp -r ${PROJECT_SOURCE_DIR}/src/ksana_llm/python/ksana_llm ${CMAKE_CURRENT_BINARY_DIR})

# copy python script for standalone test
execute_process(COMMAND cp -r ${PROJECT_SOURCE_DIR}/src/ksana_llm/python/ksana_llm ${CMAKE_BINARY_DIR})

if(WITH_CUDA)
  python_test(model_llama_integration_test SRCS ${PROJECT_SOURCE_DIR}/tests/model/llama_integration_test.py
    ARGS --config_file ${PROJECT_SOURCE_DIR}/tests/model/llama_test.yaml
    --tokenizer_dir /model/llama-hf/13B)
endif()